import pandas as pd
from pathlib import Path
from loguru import logger
import sys


def normalise_drug_name(name):
    """Normalise drug name for matching: lowercase, strip salts/forms."""
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    # Remove common salt forms and suffixes
    for suffix in [" hcl", " hydrochloride", " sodium", " potassium",
                   " calcium", " mesylate", " maleate", " tartrate",
                   " sulfate", " acetate", " phosphate", " succinate",
                   " fumarate", " besylate", " bromide", " citrate",
                   " nitrate", " oxide", " er", " sr", " cr", " xr",
                   " dr", " ec", " hct", " tab", " cap", " sol",
                   " susp", " inj", " oral"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    return name


def build_fact_demand(output_dir=None):
    """Build the core demand fact table by joining SDUD with product dimension."""
    if output_dir is None:
        output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load cleaned SDUD ──
    logger.info("Loading cleaned SDUD...")
    sdud = pd.read_parquet("data/validated/sdud_cleaned.parquet")
    logger.info(f"SDUD rows: {len(sdud):,}")

    # ── Load product dimension ──
    logger.info("Loading dim_product...")
    dim_product = pd.read_parquet("data/processed/dim_product.parquet")

    # Build a lookup from the product dimension keyed on normalised ingredient name
    dim_product["ingredient_norm"] = dim_product["ingredient"].apply(normalise_drug_name)

    product_lookup = dim_product.groupby("ingredient_norm").agg(
        application_type=("application_type", "first"),
        pharm_class=("pharm_class", "first"),
        te_code=("te_code", "first"),
        num_generic_competitors=("num_generic_competitors", "max"),
        latest_patent_expiry=("latest_patent_expiry", "first"),
        num_patents=("num_patents", "max"),
        ob_trade_name=("ob_trade_name", "first"),
    ).reset_index()

    logger.info(f"Product lookup: {len(product_lookup):,} unique ingredients")

    # ── Normalise SDUD product names ──
    sdud["product_norm"] = sdud["product_name"].apply(normalise_drug_name)

    # ── Strategy 1: Exact match on normalised names ──
    fact = sdud.merge(
        product_lookup,
        left_on="product_norm",
        right_on="ingredient_norm",
        how="left",
    )

    exact_matched = fact["application_type"].notna().sum()
    logger.info(f"Exact match: {exact_matched:,}/{len(fact):,} ({exact_matched/len(fact)*100:.1f}%)")

    # ── Strategy 2: Prefix match for truncated names ──
    # SDUD truncates names (e.g., "AMOXICILLI" for "AMOXICILLIN")
    # For unmatched rows, check if SDUD name is a prefix of any ingredient
    unmatched_mask = fact["application_type"].isna()
    unmatched_names = fact.loc[unmatched_mask, "product_norm"].unique()
    logger.info(f"Unmatched unique names to try prefix match: {len(unmatched_names):,}")

    # Build prefix lookup
    ingredient_list = product_lookup["ingredient_norm"].tolist()
    prefix_map = {}
    for sdud_name in unmatched_names:
        if not sdud_name or len(sdud_name) < 4:
            continue
        matches = [ing for ing in ingredient_list if ing.startswith(sdud_name)]
        if len(matches) == 1:
            prefix_map[sdud_name] = matches[0]
        elif len(matches) > 1:
            # Pick the shortest match (most likely the base ingredient)
            prefix_map[sdud_name] = min(matches, key=len)

    logger.info(f"Prefix matches found: {len(prefix_map):,}")

    # Apply prefix matches
    if prefix_map:
        fact["prefix_ingredient"] = fact["product_norm"].map(prefix_map)

        prefix_lookup = product_lookup.set_index("ingredient_norm")

        for col in ["application_type", "pharm_class", "te_code",
                     "num_generic_competitors", "latest_patent_expiry",
                     "num_patents", "ob_trade_name"]:
            mask = fact["application_type"].isna() & fact["prefix_ingredient"].notna()
            if mask.any():
                fact.loc[mask, col] = fact.loc[mask, "prefix_ingredient"].map(
                    prefix_lookup[col].to_dict() if col in prefix_lookup.columns else {}
                )

        # Recount matches
        total_matched = fact["application_type"].notna().sum()
        prefix_added = total_matched - exact_matched
        logger.info(f"After prefix match: {total_matched:,}/{len(fact):,} ({total_matched/len(fact)*100:.1f}%) — prefix added {prefix_added:,}")

        fact = fact.drop(columns=["prefix_ingredient"], errors="ignore")

    # ── Clean up temp columns ──
    fact = fact.drop(columns=["product_norm", "ingredient_norm"], errors="ignore")

    # ── Aggregate: combine FFSU and MCOU utilization types ──
    logger.info("Aggregating across utilization types...")
    agg_cols = {
        "units_reimbursed": "sum",
        "number_of_prescriptions": "sum",
        "total_amount_reimbursed": "sum",
        "medicaid_amount_reimbursed": "sum",
        "non_medicaid_amount_reimbursed": "sum",
        "is_suppressed": "any",
    }

    group_cols = [
        "date", "year", "quarter", "state", "ndc_11", "product_name",
    ]

    product_cols = [
        "application_type", "pharm_class", "te_code",
        "num_generic_competitors", "latest_patent_expiry", "num_patents",
    ]
    for col in product_cols:
        if col in fact.columns:
            agg_cols[col] = "first"

    fact_agg = fact.groupby(group_cols, dropna=False).agg(agg_cols).reset_index()
    logger.info(f"After aggregation: {len(fact_agg):,} rows (from {len(fact):,})")

    # ── Save ──
    output_path = output_dir / "fact_demand.parquet"
    fact_agg.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1_000_000
    logger.info(f"Saved fact_demand: {len(fact_agg):,} rows, {size_mb:.1f}MB → {output_path}")

    # ── Summary ──
    logger.info("--- fact_demand Summary ---")
    logger.info(f"Rows: {len(fact_agg):,}")
    logger.info(f"Unique NDCs: {fact_agg['ndc_11'].nunique():,}")
    logger.info(f"States: {fact_agg['state'].nunique()}")
    logger.info(f"Date range: {fact_agg['date'].min()} to {fact_agg['date'].max()}")
    logger.info(f"Total Rx: {fact_agg['number_of_prescriptions'].sum():,.0f}")

    enriched = fact_agg["application_type"].notna().sum()
    logger.info(f"Rows with product enrichment: {enriched:,}/{len(fact_agg):,} ({enriched/len(fact_agg)*100:.1f}%)")

    top_drugs = (
        fact_agg.groupby("product_name")["number_of_prescriptions"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    logger.info("Top 10 drugs by prescription volume:")
    for name, rx in top_drugs.items():
        logger.info(f"  {name}: {rx:,.0f}")

    return fact_agg


if __name__ == "__main__":
    build_fact_demand()
