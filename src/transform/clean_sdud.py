import pandas as pd
from pathlib import Path
from loguru import logger
from src.transform.ndc_harmonise import ndc_from_components
import sys


def clean_sdud(raw_dir=None, output_dir=None):
    """Load all raw SDUD files, clean, and save as Parquet."""
    if raw_dir is None:
        raw_dir = Path("data/raw/medicaid_sdud")
    else:
        raw_dir = Path(raw_dir)

    if output_dir is None:
        output_dir = Path("data/validated")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all years
    frames = []
    for f in sorted(raw_dir.glob("sdud_*.csv")):
        logger.info(f"Loading {f.name}...")
        df = pd.read_csv(f, dtype=str, low_memory=False)
        frames.append(df)
        logger.info(f"  {len(df):,} rows")

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Total raw rows: {len(df):,}")

    # Standardise column names to snake_case
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    logger.info(f"Columns: {list(df.columns)}")

    # Build 11-digit NDC from components
    df["ndc_11"] = df.apply(
        lambda row: ndc_from_components(
            row["labeler_code"],
            row["product_code"],
            row["package_size"]
        ),
        axis=1,
    )

    ndc_match = df["ndc_11"].notna().sum()
    logger.info(f"NDC construction: {ndc_match:,}/{len(df):,} ({ndc_match/len(df)*100:.1f}%)")

    # Convert numeric columns
    numeric_cols = [
        "units_reimbursed",
        "number_of_prescriptions",
        "total_amount_reimbursed",
        "medicaid_amount_reimbursed",
        "non_medicaid_amount_reimbursed",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert year and quarter to integers
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce").astype("Int64")

    # Create date column (first day of quarter)
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" +
        ((df["quarter"] - 1) * 3 + 1).astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )

    # Flag suppressed records
    df["is_suppressed"] = df["suppression_used"].str.strip().str.upper() == "TRUE"
    suppressed_count = df["is_suppressed"].sum()
    logger.info(f"Suppressed records: {suppressed_count:,} ({suppressed_count/len(df)*100:.1f}%)")

    # Standardise state codes
    df["state"] = df["state"].str.strip().str.upper()

    # Clean product name
    df["product_name"] = df["product_name"].str.strip()

    # Standardise utilization type
    df["utilization_type"] = df["utilization_type"].str.strip().str.upper()

    # Drop rows with no NDC (shouldn't happen but safety check)
    before = len(df)
    df = df.dropna(subset=["ndc_11"])
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped:,} rows with no NDC")

    # Handle duplicates: keep latest record per state/ndc/quarter/utilization_type
    before = len(df)
    df = df.drop_duplicates(
        subset=["state", "ndc_11", "year", "quarter", "utilization_type"],
        keep="last",
    )
    deduped = before - len(df)
    if deduped > 0:
        logger.info(f"Deduplicated: removed {deduped:,} duplicate rows")

    # Select and reorder final columns
    final_cols = [
        "date", "year", "quarter", "state", "utilization_type",
        "ndc_11", "ndc", "labeler_code", "product_code", "package_size",
        "product_name",
        "units_reimbursed", "number_of_prescriptions",
        "total_amount_reimbursed", "medicaid_amount_reimbursed",
        "non_medicaid_amount_reimbursed",
        "is_suppressed",
    ]
    # Only keep columns that exist
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols]

    # Save as Parquet partitioned by year
    output_path = output_dir / "sdud_cleaned.parquet"
    df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1_000_000
    logger.info(f"Saved cleaned SDUD: {len(df):,} rows, {size_mb:.1f}MB → {output_path}")

    # Print summary stats
    logger.info("--- Summary ---")
    logger.info(f"Years: {sorted(df['year'].dropna().unique().tolist())}")
    logger.info(f"States: {df['state'].nunique()}")
    logger.info(f"Unique NDCs: {df['ndc_11'].nunique():,}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Total prescriptions: {df['number_of_prescriptions'].sum():,.0f}")
    logger.info(f"Total reimbursed: ${df['total_amount_reimbursed'].sum():,.0f}")

    return df


if __name__ == "__main__":
    clean_sdud()
