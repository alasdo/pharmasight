import pandas as pd
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import requests
import os
import sys

load_dotenv()


def extract(output_dir=None):
    """Extract drug recall enforcement reports from OpenFDA."""
    if output_dir is None:
        output_dir = Path("data/raw/fda_recalls")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "fda_recalls.parquet"
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1_000_000
        logger.info(f"Already exists: {output_path} ({size_mb:.1f}MB)")
        return

    api_key = os.getenv("OPENFDA_API_KEY", "")
    base_url = "https://api.fda.gov/drug/enforcement.json"

    all_results = []
    skip = 0
    limit = 100

    logger.info("Fetching FDA drug recall enforcement reports...")

    while skip < 26000:
        params = {"limit": limit, "skip": skip}
        if api_key:
            params["api_key"] = api_key

        try:
            resp = requests.get(base_url, params=params, timeout=30)
            if resp.status_code == 404:
                break
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Error at skip={skip}: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        all_results.extend(results)
        total = data.get("meta", {}).get("results", {}).get("total", 0)

        if len(all_results) % 1000 < limit:
            logger.info(f"  Fetched {len(all_results):,} / {total:,} records")

        skip += limit
        if skip >= total:
            break

    logger.info(f"Total recall records fetched: {len(all_results):,}")

    rows = []
    for r in all_results:
        openfda = r.get("openfda", {})
        rows.append({
            "recall_number": r.get("recall_number"),
            "classification": r.get("classification"),
            "reason": r.get("reason_for_recall"),
            "status": r.get("status"),
            "recall_date": r.get("recall_initiation_date"),
            "report_date": r.get("report_date"),
            "product_description": r.get("product_description"),
            "generic_name": "; ".join(openfda.get("generic_name", [])) or None,
            "brand_name": "; ".join(openfda.get("brand_name", [])) or None,
            "ndc": "; ".join(openfda.get("ndc", [])) or None,
            "manufacturer": "; ".join(openfda.get("manufacturer_name", [])) or None,
            "state": r.get("state"),
            "city": r.get("city"),
            "voluntary_mandated": r.get("voluntary_mandated"),
        })

    df = pd.DataFrame(rows)
    df["recall_date"] = pd.to_datetime(df["recall_date"], errors="coerce")
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")

    df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1_000_000
    logger.info(f"Saved {len(df):,} recall records → {output_path} ({size_mb:.1f}MB)")

    # Summary
    logger.info("--- Recalls Summary ---")
    logger.info(f"Total records: {len(df):,}")
    if "classification" in df.columns:
        logger.info(f"By class: {df['classification'].value_counts().to_dict()}")
    logger.info(f"Date range: {df['recall_date'].min()} to {df['recall_date'].max()}")
    logger.info(f"With NDC: {df['ndc'].notna().sum():,}")


def verify(output_dir=None):
    if output_dir is None:
        output_dir = Path("data/raw/fda_recalls")
    path = output_dir / "fda_recalls.parquet"
    if not path.exists():
        logger.error("File not found. Run extract first.")
        return
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} rows")
    logger.info(f"Classification: {df['classification'].value_counts().to_dict()}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if action == "extract":
        extract()
    elif action == "verify":
        verify()
    else:
        print("Usage: python -m src.extract.fda_recalls [extract|verify]")
