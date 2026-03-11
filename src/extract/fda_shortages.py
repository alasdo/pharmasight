import pandas as pd
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import requests
import os
import sys

load_dotenv()


def extract(output_dir=None):
    """Extract all drug shortage records from OpenFDA."""
    if output_dir is None:
        output_dir = Path("data/raw/fda_shortages")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "fda_shortages.parquet"
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1_000_000
        logger.info(f"Already exists: {output_path} ({size_mb:.1f}MB)")
        return

    api_key = os.getenv("OPENFDA_API_KEY", "")
    base_url = "https://api.fda.gov/drug/shortages.json"

    all_records = []
    skip = 0
    limit = 100

    logger.info("Fetching FDA drug shortages...")

    while True:
        params = {"limit": limit, "skip": skip}
        if api_key:
            params["api_key"] = api_key

        try:
            resp = requests.get(base_url, params=params, timeout=30)
            if resp.status_code == 404:
                logger.info("Reached end of results (404)")
                break
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            if "404" in str(e):
                break
            logger.error(f"HTTP error at skip={skip}: {e}")
            break
        except Exception as e:
            logger.error(f"Error at skip={skip}: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        all_records.extend(results)
        total = data.get("meta", {}).get("results", {}).get("total", 0)

        if len(all_records) % 500 < limit:
            logger.info(f"  Fetched {len(all_records):,} / {total:,} records")

        skip += limit
        if skip >= total or skip >= 26000:
            break

    if not all_records:
        logger.warning("No shortage records returned. The endpoint may require different parameters.")
        logger.info("Trying alternative: searching for all shortage statuses...")

        for status in ["current", "resolved", "discontinued"]:
            skip = 0
            while True:
                params = {
                    "search": f'status:"{status}"',
                    "limit": limit,
                    "skip": skip,
                }
                if api_key:
                    params["api_key"] = api_key

                try:
                    resp = requests.get(base_url, params=params, timeout=30)
                    if resp.status_code == 404:
                        break
                    resp.raise_for_status()
                    data = resp.json()
                except Exception:
                    break

                results = data.get("results", [])
                if not results:
                    break
                all_records.extend(results)
                skip += limit
                total = data.get("meta", {}).get("results", {}).get("total", 0)
                if skip >= total or skip >= 26000:
                    break

            logger.info(f"  Status '{status}': {len(all_records):,} total so far")

    logger.info(f"Total shortage records fetched: {len(all_records):,}")

    if not all_records:
        logger.error("No records retrieved. Check API key and endpoint availability.")
        return

    # Flatten records
    rows = []
    for record in all_records:
        row = {
            "generic_name": record.get("generic_name"),
            "status": record.get("status"),
            "initial_posting_date": record.get("initial_posting_date"),
            "resolved_date": record.get("resolved_date"),
            "shortage_reason": record.get("shortage_reason"),
        }

        products = record.get("products_affected", [])
        if products and isinstance(products, list):
            for product in products:
                r = row.copy()
                if isinstance(product, dict):
                    r["ndc"] = product.get("ndc")
                    r["product_name"] = product.get("name")
                else:
                    r["product_name"] = str(product)
                rows.append(r)
        else:
            rows.append(row)

    df = pd.DataFrame(rows)

    for col in ["initial_posting_date", "resolved_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "initial_posting_date" in df.columns and "resolved_date" in df.columns:
        df["shortage_duration_days"] = (df["resolved_date"] - df["initial_posting_date"]).dt.days

    df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1_000_000
    logger.info(f"Saved {len(df):,} shortage records → {output_path} ({size_mb:.1f}MB)")

    # Summary
    logger.info("--- Shortages Summary ---")
    logger.info(f"Total records: {len(df):,}")
    if "status" in df.columns:
        logger.info(f"By status: {df['status'].value_counts().to_dict()}")
    if "shortage_reason" in df.columns:
        logger.info(f"By reason: {df['shortage_reason'].value_counts().head(5).to_dict()}")


def verify(output_dir=None):
    if output_dir is None:
        output_dir = Path("data/raw/fda_shortages")
    path = output_dir / "fda_shortages.parquet"
    if not path.exists():
        logger.error("File not found. Run extract first.")
        return
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} rows")
    logger.info(f"Columns: {list(df.columns)}")
    if "status" in df.columns:
        logger.info(f"Status: {df['status'].value_counts().to_dict()}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if action == "extract":
        extract()
    elif action == "verify":
        verify()
    else:
        print("Usage: python -m src.extract.fda_shortages [extract|verify]")