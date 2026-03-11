import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys


# ── HHS Region to State Mapping (for FluView join) ──
HHS_TO_STATES = {
    "hhs1": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "hhs2": ["NJ", "NY"],
    "hhs3": ["DE", "DC", "MD", "PA", "VA", "WV"],
    "hhs4": ["AL", "FL", "GA", "KY", "MS", "NC", "SC", "TN"],
    "hhs5": ["IL", "IN", "MI", "MN", "OH", "WI"],
    "hhs6": ["AR", "LA", "NM", "OK", "TX"],
    "hhs7": ["IA", "KS", "MO", "NE"],
    "hhs8": ["CO", "MT", "ND", "SD", "UT", "WY"],
    "hhs9": ["AZ", "CA", "HI", "NV"],
    "hhs10": ["AK", "ID", "OR", "WA"],
}

STATE_TO_HHS = {}
for region, states in HHS_TO_STATES.items():
    for s in states:
        STATE_TO_HHS[s] = region


def build_feat_disease(output_dir=None):
    """Build disease feature table from CDC FluView data."""
    if output_dir is None:
        output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building feat_disease from FluView...")

    # Load FluView
    fluview_dir = Path("data/raw/cdc_fluview")
    files = list(fluview_dir.glob("*.parquet"))
    if not files:
        logger.error("No FluView data found")
        return

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    logger.info(f"FluView records: {len(df):,}")

    # Ensure numeric ILI columns
    for col in ["wili", "ili", "num_ili", "num_patients", "num_providers"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse epiweek to date if not already
    if "date" not in df.columns:
        df["epiweek_str"] = df["epiweek"].astype(str)
        df["year"] = df["epiweek_str"].str[:4].astype(int)
        df["week"] = df["epiweek_str"].str[4:].astype(int)
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + df["week"].astype(str).str.zfill(2) + "1",
            format="%G%V%u",
            errors="coerce",
        )

    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year

    # Use wili (weighted ILI) as primary, fall back to ili
    ili_col = "wili" if "wili" in df.columns else "ili"

    # Aggregate weekly → quarterly per region
    quarterly = df.groupby(["region", "year", "quarter"]).agg(
        ili_rate_mean=(ili_col, "mean"),
        ili_rate_max=(ili_col, "max"),
        ili_rate_std=(ili_col, "std"),
        total_ili_cases=("num_ili", "sum") if "num_ili" in df.columns else (ili_col, "count"),
        total_patients=("num_patients", "sum") if "num_patients" in df.columns else (ili_col, "count"),
        num_weeks=(ili_col, "count"),
    ).reset_index()

    # Create date column (first day of quarter)
    quarterly["date"] = pd.to_datetime(
        quarterly["year"].astype(str) + "-"
        + ((quarterly["quarter"] - 1) * 3 + 1).astype(str).str.zfill(2) + "-01"
    )

    # Flu season flag (Q4 and Q1 are peak)
    quarterly["is_flu_season"] = quarterly["quarter"].isin([1, 4]).astype(int)

    # Year-over-year ILI change
    quarterly = quarterly.sort_values(["region", "year", "quarter"])
    quarterly["ili_rate_yoy_change"] = quarterly.groupby("region")["ili_rate_mean"].pct_change(4)

    # Expand regions to states
    rows = []
    for _, row in quarterly.iterrows():
        region = row["region"]
        states = HHS_TO_STATES.get(region, [])
        for state in states:
            new_row = row.to_dict()
            new_row["state"] = state
            rows.append(new_row)

    feat = pd.DataFrame(rows)
    feat = feat.drop(columns=["region"], errors="ignore")

    output_path = output_dir / "feat_disease.parquet"
    feat.to_parquet(output_path, index=False)
    logger.info(f"Saved feat_disease: {len(feat):,} rows → {output_path}")
    logger.info(f"  States: {feat['state'].nunique()}, Quarters: {feat['date'].nunique()}")

    return feat


def build_feat_supply(output_dir=None):
    """Build supply feature table from shortages, recalls, and product data."""
    if output_dir is None:
        output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building feat_supply...")

    # Load fact_demand to get the universe of NDC × date combinations
    fact = pd.read_parquet(
        "data/processed/fact_demand.parquet",
        columns=["date", "ndc_11", "product_name"],
    )
    fact_keys = fact[["date", "ndc_11"]].drop_duplicates()
    logger.info(f"Demand universe: {len(fact_keys):,} NDC × date combos")

    # ── Shortage features ──
    shortage_path = Path("data/raw/fda_shortages/fda_shortages.parquet")
    if shortage_path.exists():
        shortages = pd.read_parquet(shortage_path)
        logger.info(f"Shortages: {len(shortages):,} records")

        shortage_status = shortages.groupby("status").size()
        logger.info(f"  Shortage status: {shortage_status.to_dict()}")

        # Extract base ingredient: first word of generic_name, lowercased
        current_shortages = shortages[shortages["status"].str.lower().str.contains("current|ongoing", na=False)]
        shortage_first_words = set()
        for name in current_shortages["generic_name"].dropna().unique():
            first_word = name.strip().split()[0].lower()
            if len(first_word) >= 6:  # skip short words like "amino" that match too broadly
                shortage_first_words.add(first_word)

        logger.info(f"  Shortage base ingredients (first word): {len(shortage_first_words)}")
        logger.info(f"  Samples: {sorted(list(shortage_first_words))[:15]}")

        # Get unique SDUD product names
        sdud_names = fact["product_name"].dropna().str.strip().str.lower().unique()
        # Filter to names starting with a letter (skip "0.9% sodium" etc.)
        sdud_names_alpha = [n for n in sdud_names if n and n[0].isalpha()]

        # Match: SDUD name starts with a shortage ingredient
        # OR shortage ingredient starts with the SDUD name (for truncated names)
        shortage_sdud_matches = set()
        for sdud_name in sdud_names_alpha:
            for shortage_word in shortage_first_words:
                # "albuterol" starts with "albuterol" ✓
                # "albutero" (truncated) — "albuterol" starts with "albutero" ✓
                # "carboplat" (truncated) — "carboplatin" starts with "carboplat" ✓
                sdud_first = sdud_name.split()[0]
                # Forward: SDUD starts with shortage word (e.g., "albuterol sulfate" starts with "albuterol")
                # Reverse: shortage word starts with SDUD first word (for truncated names)
                # Require at least 6 chars overlap to avoid false positives
                if len(sdud_first) >= 6 and (
                    sdud_first.startswith(shortage_word) or shortage_word.startswith(sdud_first)
                ):
                    shortage_sdud_matches.add(sdud_name)
                    break

        logger.info(f"  SDUD products matching shortages: {len(shortage_sdud_matches)}")
        if shortage_sdud_matches:
            samples = sorted([m for m in shortage_sdud_matches if len(m) > 3])[:15]
            logger.info(f"  Sample matches: {samples}")
    else:
        shortage_sdud_matches = set()
        logger.warning("No shortage data found")


    # ── Recall features ──
    recall_path = Path("data/raw/fda_recalls/fda_recalls.parquet")
    if recall_path.exists():
        recalls = pd.read_parquet(recall_path)
        logger.info(f"Recalls: {len(recalls):,} records")

        recalls["recall_date"] = pd.to_datetime(recalls["recall_date"], errors="coerce")
        recalls["year"] = recalls["recall_date"].dt.year
        recalls["quarter"] = recalls["recall_date"].dt.quarter
        recalls["date"] = pd.to_datetime(
            recalls["year"].astype(str) + "-"
            + ((recalls["quarter"] - 1) * 3 + 1).astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )

        # Count recalls per quarter by classification
        recall_quarterly = recalls.groupby(["date"]).agg(
            total_recalls=("recall_number", "nunique"),
            class_i_recalls=("classification", lambda x: (x == "Class I").sum()),
            class_ii_recalls=("classification", lambda x: (x == "Class II").sum()),
        ).reset_index()

        logger.info(f"  Quarterly recall features: {len(recall_quarterly):,} quarters")
    else:
        recall_quarterly = None
        logger.warning("No recall data found")

    # ── Product dimension features ──
    dim_product_path = Path("data/processed/dim_product.parquet")
    if dim_product_path.exists():
        dim = pd.read_parquet(dim_product_path)
        logger.info(f"Product dimension: {len(dim):,} products")

        # Patent cliff features
        if "latest_patent_expiry" in dim.columns:
            dim["latest_patent_expiry"] = pd.to_datetime(dim["latest_patent_expiry"], errors="coerce")

        # Build ingredient-level features
        ingredient_features = dim.groupby("ingredient").agg(
            application_type=("application_type", "first"),
            num_generic_competitors=("num_generic_competitors", "max"),
            has_patent=("latest_patent_expiry", lambda x: x.notna().any()),
            latest_patent_expiry=("latest_patent_expiry", "max"),
            num_patents=("num_patents", "max"),
        ).reset_index()

        ingredient_features["ingredient_lower"] = ingredient_features["ingredient"].str.strip().str.lower()
        logger.info(f"  Ingredient features: {len(ingredient_features):,} unique ingredients")
    else:
        ingredient_features = None
        logger.warning("No product dimension found")

    # ── Assemble feat_supply ──
    # Start with the fact_keys and enrich
    feat = fact_keys.copy()

    # Add product name for matching
    feat = feat.merge(
        fact[["date", "ndc_11", "product_name"]].drop_duplicates(),
        on=["date", "ndc_11"],
        how="left",
    )
    feat["product_name_lower"] = feat["product_name"].str.strip().str.lower()

    # Shortage flag (ingredient-based matching)
    feat["shortage_active"] = feat["product_name_lower"].isin(shortage_sdud_matches).astype(int)
    logger.info(f"  Rows with active shortage: {feat['shortage_active'].sum():,}")

    # Generic competition count (from product dimension)
    if ingredient_features is not None:
        feat = feat.merge(
            ingredient_features[["ingredient_lower", "num_generic_competitors", "has_patent", "latest_patent_expiry"]],
            left_on="product_name_lower",
            right_on="ingredient_lower",
            how="left",
        )
        feat = feat.drop(columns=["ingredient_lower"], errors="ignore")
        feat["num_generic_competitors"] = feat["num_generic_competitors"].fillna(0).astype(int)

        # Patent cliff: months until expiry
        if "latest_patent_expiry" in feat.columns:
            feat["months_to_patent_expiry"] = (
                (feat["latest_patent_expiry"] - feat["date"]).dt.days / 30.44
            )
            feat["is_near_patent_cliff"] = (
                (feat["months_to_patent_expiry"] > 0) & (feat["months_to_patent_expiry"] <= 24)
            ).astype(int)

    # Quarterly recall counts (market-wide)
    if recall_quarterly is not None:
        feat = feat.merge(recall_quarterly, on="date", how="left")
        feat["total_recalls"] = feat["total_recalls"].fillna(0).astype(int)
        feat["class_i_recalls"] = feat["class_i_recalls"].fillna(0).astype(int)

    # Clean up
    feat = feat.drop(columns=["product_name", "product_name_lower"], errors="ignore")

    output_path = output_dir / "feat_supply.parquet"
    feat.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1_000_000
    logger.info(f"Saved feat_supply: {len(feat):,} rows, {size_mb:.1f}MB → {output_path}")

    return feat


def build_feat_safety(output_dir=None):
    """Build safety feature table from FAERS adverse event data."""
    if output_dir is None:
        output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building feat_safety from FAERS...")

    faers_dir = Path("data/raw/openfda_faers")
    files = sorted(faers_dir.glob("faers_part_*.parquet"))
    if not files:
        logger.error("No FAERS data found")
        return

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        frames.append(df)
    faers = pd.concat(frames, ignore_index=True)
    logger.info(f"FAERS total records: {len(faers):,}")

    # Parse receive_date
    if "receive_date" in faers.columns:
        faers["receive_date"] = pd.to_datetime(faers["receive_date"], format="%Y%m%d", errors="coerce")
    elif "receivedate" in faers.columns:
        faers["receive_date"] = pd.to_datetime(faers["receivedate"], format="%Y%m%d", errors="coerce")

    faers["year"] = faers["receive_date"].dt.year
    faers["quarter"] = faers["receive_date"].dt.quarter
    faers["date"] = pd.to_datetime(
        faers["year"].astype(str) + "-"
        + ((faers["quarter"] - 1) * 3 + 1).astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )

    # Identify the drug name column
    drug_col = None
    for candidate in ["drug_name", "generic_name", "medicinalproduct"]:
        if candidate in faers.columns:
            drug_col = candidate
            break

    if drug_col is None:
        logger.error(f"No drug name column found. Columns: {list(faers.columns)}")
        return

    faers["drug_name_lower"] = faers[drug_col].str.strip().str.lower()

    # Aggregate quarterly per drug
    serious_col = None
    for candidate in ["is_serious", "serious"]:
        if candidate in faers.columns:
            serious_col = candidate
            break

    agg_dict = {"date": "count"}
    if serious_col:
        faers[serious_col] = pd.to_numeric(faers[serious_col], errors="coerce")

    quarterly = faers.groupby(["drug_name_lower", "date"]).agg(
        adverse_event_count=("date", "size"),
    ).reset_index()

    if serious_col:
        serious_counts = faers[faers[serious_col] == 1].groupby(["drug_name_lower", "date"]).size().reset_index(name="serious_event_count")
        quarterly = quarterly.merge(serious_counts, on=["drug_name_lower", "date"], how="left")
        quarterly["serious_event_count"] = quarterly["serious_event_count"].fillna(0).astype(int)

    # Quarter-over-quarter change (spike detection)
    quarterly = quarterly.sort_values(["drug_name_lower", "date"])
    quarterly["ae_qoq_change"] = quarterly.groupby("drug_name_lower")["adverse_event_count"].pct_change()
    quarterly["ae_spike"] = (quarterly["ae_qoq_change"] > 1.0).astype(int)  # >100% increase

    output_path = output_dir / "feat_safety.parquet"
    quarterly.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1_000_000
    logger.info(f"Saved feat_safety: {len(quarterly):,} rows, {size_mb:.1f}MB → {output_path}")
    logger.info(f"  Unique drugs: {quarterly['drug_name_lower'].nunique():,}")
    logger.info(f"  Adverse event spikes detected: {quarterly['ae_spike'].sum():,}")

    return quarterly


def build_demand_features(output_dir=None):
    """Add lagged demand and rolling statistics to fact_demand."""
    if output_dir is None:
        output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building demand features (lags + rolling stats)...")

    fact = pd.read_parquet("data/processed/fact_demand.parquet")
    logger.info(f"fact_demand: {len(fact):,} rows")

    # Sort for time-series operations
    fact = fact.sort_values(["ndc_11", "state", "date"])

    # Group by drug × state for lag/rolling computations
    group = fact.groupby(["ndc_11", "state"])

    # Lagged demand
    fact["rx_lag_1"] = group["number_of_prescriptions"].shift(1)
    fact["rx_lag_2"] = group["number_of_prescriptions"].shift(2)
    fact["rx_lag_4"] = group["number_of_prescriptions"].shift(4)  # same quarter last year

    # Rolling statistics (4-quarter = 1 year window)
    fact["rx_rolling_mean_4"] = group["number_of_prescriptions"].transform(
        lambda x: x.rolling(4, min_periods=2).mean()
    )
    fact["rx_rolling_std_4"] = group["number_of_prescriptions"].transform(
        lambda x: x.rolling(4, min_periods=2).std()
    )

    # Year-over-year change
    fact["rx_yoy_change"] = group["number_of_prescriptions"].pct_change(4)

    # Reimbursement lag
    fact["reimb_lag_1"] = group["total_amount_reimbursed"].shift(1)

    # Trend: linear slope over last 4 quarters
    def rolling_slope(series, window=4):
        """Compute rolling linear regression slope."""
        x = np.arange(window)
        result = series.rolling(window, min_periods=window).apply(
            lambda y: np.polyfit(x, y, 1)[0] if len(y) == window else np.nan,
            raw=True,
        )
        return result

    fact["rx_trend_4"] = group["number_of_prescriptions"].transform(
        lambda x: rolling_slope(x, 4)
    )

    # Calendar features
    fact["quarter_num"] = fact["date"].dt.quarter
    fact["year_num"] = fact["date"].dt.year
    fact["quarter_sin"] = np.sin(2 * np.pi * fact["quarter_num"] / 4)
    fact["quarter_cos"] = np.cos(2 * np.pi * fact["quarter_num"] / 4)

    # Count NaN features (indicates series start)
    nan_count = fact[["rx_lag_1", "rx_lag_4", "rx_rolling_mean_4"]].isna().sum()
    logger.info(f"  NaN counts (expected for series start): {nan_count.to_dict()}")

    output_path = output_dir / "fact_demand_features.parquet"
    fact.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1_000_000
    logger.info(f"Saved fact_demand_features: {len(fact):,} rows, {size_mb:.1f}MB → {output_path}")
    logger.info(f"  New columns: rx_lag_1, rx_lag_2, rx_lag_4, rx_rolling_mean_4, rx_rolling_std_4, rx_yoy_change, rx_trend_4, quarter_sin, quarter_cos")

    return fact


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "all"

    if action in ("all", "disease"):
        build_feat_disease()
    if action in ("all", "supply"):
        build_feat_supply()
    if action in ("all", "safety"):
        build_feat_safety()
    if action in ("all", "demand"):
        build_demand_features()
