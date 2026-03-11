import requests
import pandas as pd
from pathlib import Path
from loguru import logger
import sys


DELPHI_BASE = "https://api.delphi.cmu.edu/epidata/fluview/"

HHS_REGION_STATES = {
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


def extract(start_year=2018, end_year=2024, output_dir=None):
    """Extract weekly ILI data from Delphi Epidata API."""
    if output_dir is None:
        output_dir = Path("data/raw/cdc_fluview")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"fluview_{start_year}_{end_year}.parquet"
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1_000_000
        logger.info(f"Already exists: {output_path} ({size_mb:.1f}MB)")
        return

    epiweek_range = f"{start_year}01-{end_year}52"
    regions = ",".join([f"hhs{i}" for i in range(1, 11)])

    logger.info(f"Fetching FluView data: regions={regions}, epiweeks={epiweek_range}")

    resp = requests.get(DELPHI_BASE, params={
        "regions": regions,
        "epiweeks": epiweek_range,
    }, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if data.get("result") != 1:
        logger.error(f"API error: {data.get('message')}")
        return

    epidata = data.get("epidata", [])
    if not epidata:
        logger.error("No data returned")
        return

    df = pd.DataFrame(epidata)
    logger.info(f"Received {len(df):,} records")

    # Parse epiweek to date
    df["epiweek_str"] = df["epiweek"].astype(str)
    df["year"] = df["epiweek_str"].str[:4].astype(int)
    df["week"] = df["epiweek_str"].str[4:].astype(int)

    # Convert epiweek to date (Monday of that week)
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + df["week"].astype(str).str.zfill(2) + "1",
        format="%G%V%u",
        errors="coerce",
    )

    # Expand HHS regions to states for joining with SDUD
    region_to_states = {}
    for region, states in HHS_REGION_STATES.items():
        for state in states:
            region_to_states[state] = region

    df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1_000_000
    logger.info(f"Saved {len(df):,} records → {output_path} ({size_mb:.1f}MB)")

    # Summary
    logger.info("--- FluView Summary ---")
    logger.info(f"Regions: {df['region'].nunique()}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    if "wili" in df.columns:
        logger.info(f"Weighted ILI range: {df['wili'].min():.2f} to {df['wili'].max():.2f}")
    elif "ili" in df.columns:
        logger.info(f"ILI range: {df['ili'].min():.2f} to {df['ili'].max():.2f}")
    logger.info(f"Columns: {list(df.columns)}")


def verify(output_dir=None):
    if output_dir is None:
        output_dir = Path("data/raw/cdc_fluview")
    files = list(output_dir.glob("*.parquet"))
    if not files:
        logger.error("No files found. Run extract first.")
        return
    for f in files:
        df = pd.read_parquet(f)
        logger.info(f"{f.name}: {len(df):,} rows, columns: {list(df.columns)[:8]}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if action == "extract":
        extract()
    elif action == "verify":
        verify()
    else:
        print("Usage: python -m src.extract.cdc_fluview [extract|verify]")
