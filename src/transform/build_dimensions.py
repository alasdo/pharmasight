import pandas as pd
from pathlib import Path
from loguru import logger
import sys


def build_dim_product(output_dir=None):
    """Build product dimension table by joining Drugs@FDA + Orange Book + SDUD product info."""
    if output_dir is None:
        output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load SDUD for the universe of NDCs we care about ──
    logger.info("Loading SDUD cleaned data...")
    sdud = pd.read_parquet("data/validated/sdud_cleaned.parquet",
                           columns=["ndc_11", "product_name", "labeler_code"])
    sdud_products = sdud.drop_duplicates(subset=["ndc_11"]).copy()
    logger.info(f"SDUD unique NDCs: {len(sdud_products):,}")

    # ── Load Drugs@FDA ──
    logger.info("Loading Drugs@FDA...")
    fda = pd.read_parquet("data/raw/drugs_at_fda/drugs_at_fda.parquet")
    logger.info(f"Drugs@FDA records: {len(fda):,}")

    # ── Load Orange Book products ──
    logger.info("Loading Orange Book products...")
    ob_products = pd.read_parquet("data/raw/orange_book/ob_products.parquet")
    logger.info(f"Orange Book products: {len(ob_products):,}")

    # Standardise Orange Book column names
    ob_products.columns = ob_products.columns.str.strip().str.lower().str.replace(" ", "_")
    logger.info(f"Orange Book columns: {list(ob_products.columns)}")

    # ── Load Orange Book patents ──
    logger.info("Loading Orange Book patents...")
    ob_patents = pd.read_parquet("data/raw/orange_book/ob_patent.parquet")
    ob_patents.columns = ob_patents.columns.str.strip().str.lower().str.replace(" ", "_")
    logger.info(f"Orange Book patents: {len(ob_patents):,}")

    # ── Build application number from Orange Book ──
    # Orange Book has Appl_Type (N or A) + Appl_No
    # Drugs@FDA has application_number like "NDA017442" or "ANDA076183"
    if "appl_type" in ob_products.columns and "appl_no" in ob_products.columns:
        ob_products["application_number"] = (
            ob_products["appl_type"].str.strip() + ob_products["appl_no"].str.strip().str.zfill(6)
        )
        # Map N -> NDA, A -> ANDA for matching
        ob_products["application_number"] = ob_products["application_number"].str.replace("^N", "NDA", regex=True)
        ob_products["application_number"] = ob_products["application_number"].str.replace("^A", "ANDA", regex=True)

    # ── Extract key fields from Orange Book products ──
    ob_cols = []
    col_mapping = {}
    for candidate, target in [
        ("ingredient", "ob_ingredient"),
        ("trade_name", "ob_trade_name"),
        ("applicant_full_name", "ob_applicant"),
        ("type", "ob_type"),
        ("te_code", "te_code"),
        ("rld", "is_rld"),
        ("rs", "is_rs"),
        ("approval_date", "ob_approval_date"),
        ("application_number", "application_number"),
        ("appl_no", "appl_no"),
        ("product_no", "product_no"),
        ("df;route", "dosage_form_route"),
        ("strength", "strength"),
    ]:
        if candidate in ob_products.columns:
            col_mapping[candidate] = target
            ob_cols.append(candidate)

    ob_slim = ob_products[ob_cols].rename(columns=col_mapping).copy()

    # ── Extract patent expiry from Orange Book patents ──
    patent_date_col = None
    for candidate in ["patent_expire_date_text", "patent_expire_date"]:
        if candidate in ob_patents.columns:
            patent_date_col = candidate
            break

    if patent_date_col:
        ob_patents["patent_expiry"] = pd.to_datetime(ob_patents[patent_date_col], errors="coerce")

        # Get the latest patent expiry per application
        appl_no_col = "appl_no" if "appl_no" in ob_patents.columns else None
        prod_no_col = "product_no" if "product_no" in ob_patents.columns else None

        if appl_no_col and prod_no_col:
            patent_summary = ob_patents.groupby([appl_no_col, prod_no_col]).agg(
                latest_patent_expiry=("patent_expiry", "max"),
                num_patents=("patent_expiry", "count"),
            ).reset_index()
            logger.info(f"Patent summaries: {len(patent_summary):,}")
        else:
            patent_summary = None
            logger.warning("Could not find appl_no/product_no in patents")
    else:
        patent_summary = None
        logger.warning(f"No patent date column found. Columns: {list(ob_patents.columns)}")

    # ── Join patent info to OB products ──
    if patent_summary is not None and "appl_no" in ob_slim.columns and "product_no" in ob_slim.columns:
        ob_slim = ob_slim.merge(patent_summary, on=["appl_no", "product_no"], how="left")
        has_patent = ob_slim["latest_patent_expiry"].notna().sum()
        logger.info(f"OB products with patent data: {has_patent:,}/{len(ob_slim):,}")

    # ── Extract key fields from Drugs@FDA ──
    fda_slim = fda[[
        "application_number", "application_type", "sponsor_name",
        "brand_name", "generic_name", "active_ingredients",
        "dosage_form", "pharm_class", "approval_date",
    ]].copy()
    fda_slim = fda_slim.rename(columns={"approval_date": "fda_approval_date"})

    # Deduplicate Drugs@FDA to one row per application_number
    fda_slim = fda_slim.drop_duplicates(subset=["application_number"], keep="first")
    logger.info(f"Drugs@FDA unique applications: {len(fda_slim):,}")

    # ── Join OB products with Drugs@FDA via application_number ──
    if "application_number" in ob_slim.columns:
        dim = ob_slim.merge(fda_slim, on="application_number", how="left")
        matched = dim["generic_name"].notna().sum()
        logger.info(f"OB→FDA join: {matched:,}/{len(dim):,} matched ({matched/len(dim)*100:.1f}%)")
    else:
        dim = ob_slim.copy()
        logger.warning("No application_number in OB — skipping FDA join")

    # ── Build the final dimension table ──
    # Use OB ingredient as primary, fall back to FDA generic_name
    if "ob_ingredient" in dim.columns:
        dim["ingredient"] = dim["ob_ingredient"].fillna(dim.get("generic_name"))
    elif "generic_name" in dim.columns:
        dim["ingredient"] = dim["generic_name"]

    # ── Compute generic competition count ──
    if "ingredient" in dim.columns and "application_type" in dim.columns:
        generic_counts = dim[dim["application_type"] == "ANDA"].groupby("ingredient").size().reset_index(name="num_generic_competitors")
        dim = dim.merge(generic_counts, on="ingredient", how="left")
        dim["num_generic_competitors"] = dim["num_generic_competitors"].fillna(0).astype(int)
        logger.info(f"Drugs with generic competition: {(dim['num_generic_competitors'] > 0).sum():,}")

    # ── Save ──
    output_path = output_dir / "dim_product.parquet"
    dim.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1_000_000
    logger.info(f"Saved dim_product: {len(dim):,} rows, {size_mb:.1f}MB → {output_path}")

    # Summary
    logger.info("--- dim_product Summary ---")
    logger.info(f"Total products: {len(dim):,}")
    if "application_type" in dim.columns:
        logger.info(f"By type: {dim['application_type'].value_counts().to_dict()}")
    if "te_code" in dim.columns:
        logger.info(f"With TE code: {dim['te_code'].notna().sum():,}")
    if "latest_patent_expiry" in dim.columns:
        logger.info(f"With patent data: {dim['latest_patent_expiry'].notna().sum():,}")
    if "num_generic_competitors" in dim.columns:
        logger.info(f"Avg generic competitors: {dim['num_generic_competitors'].mean():.1f}")

    return dim


def build_dim_geography(output_dir=None):
    """Build geography dimension table."""
    if output_dir is None:
        output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # US states + DC + territories
    states = {
        "AL": ("Alabama", "hhs4", "East South Central"),
        "AK": ("Alaska", "hhs10", "Pacific"),
        "AZ": ("Arizona", "hhs9", "Mountain"),
        "AR": ("Arkansas", "hhs6", "West South Central"),
        "CA": ("California", "hhs9", "Pacific"),
        "CO": ("Colorado", "hhs8", "Mountain"),
        "CT": ("Connecticut", "hhs1", "New England"),
        "DE": ("Delaware", "hhs3", "South Atlantic"),
        "DC": ("District of Columbia", "hhs3", "South Atlantic"),
        "FL": ("Florida", "hhs4", "South Atlantic"),
        "GA": ("Georgia", "hhs4", "South Atlantic"),
        "HI": ("Hawaii", "hhs9", "Pacific"),
        "ID": ("Idaho", "hhs10", "Mountain"),
        "IL": ("Illinois", "hhs5", "East North Central"),
        "IN": ("Indiana", "hhs5", "East North Central"),
        "IA": ("Iowa", "hhs7", "West North Central"),
        "KS": ("Kansas", "hhs7", "West North Central"),
        "KY": ("Kentucky", "hhs4", "East South Central"),
        "LA": ("Louisiana", "hhs6", "West South Central"),
        "ME": ("Maine", "hhs1", "New England"),
        "MD": ("Maryland", "hhs3", "South Atlantic"),
        "MA": ("Massachusetts", "hhs1", "New England"),
        "MI": ("Michigan", "hhs5", "East North Central"),
        "MN": ("Minnesota", "hhs5", "West North Central"),
        "MS": ("Mississippi", "hhs4", "East South Central"),
        "MO": ("Missouri", "hhs7", "West North Central"),
        "MT": ("Montana", "hhs8", "Mountain"),
        "NE": ("Nebraska", "hhs7", "West North Central"),
        "NV": ("Nevada", "hhs9", "Mountain"),
        "NH": ("New Hampshire", "hhs1", "New England"),
        "NJ": ("New Jersey", "hhs2", "Middle Atlantic"),
        "NM": ("New Mexico", "hhs6", "Mountain"),
        "NY": ("New York", "hhs2", "Middle Atlantic"),
        "NC": ("North Carolina", "hhs4", "South Atlantic"),
        "ND": ("North Dakota", "hhs8", "West North Central"),
        "OH": ("Ohio", "hhs5", "East North Central"),
        "OK": ("Oklahoma", "hhs6", "West South Central"),
        "OR": ("Oregon", "hhs10", "Pacific"),
        "PA": ("Pennsylvania", "hhs3", "Middle Atlantic"),
        "RI": ("Rhode Island", "hhs1", "New England"),
        "SC": ("South Carolina", "hhs4", "South Atlantic"),
        "SD": ("South Dakota", "hhs8", "West North Central"),
        "TN": ("Tennessee", "hhs4", "East South Central"),
        "TX": ("Texas", "hhs6", "West South Central"),
        "UT": ("Utah", "hhs8", "Mountain"),
        "VT": ("Vermont", "hhs1", "New England"),
        "VA": ("Virginia", "hhs3", "South Atlantic"),
        "WA": ("Washington", "hhs10", "Pacific"),
        "WV": ("West Virginia", "hhs3", "South Atlantic"),
        "WI": ("Wisconsin", "hhs5", "East North Central"),
        "WY": ("Wyoming", "hhs8", "Mountain"),
        "PR": ("Puerto Rico", "hhs2", "Territory"),
        "VI": ("Virgin Islands", "hhs2", "Territory"),
        "GU": ("Guam", "hhs9", "Territory"),
    }

    rows = []
    for code, (name, hhs, division) in states.items():
        rows.append({
            "state_code": code,
            "state_name": name,
            "hhs_region": hhs,
            "census_division": division,
        })

    df = pd.DataFrame(rows)

    output_path = output_dir / "dim_geography.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved dim_geography: {len(df)} rows → {output_path}")

    return df


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "all"
    if action in ("all", "product"):
        build_dim_product()
    if action in ("all", "geography"):
        build_dim_geography()
