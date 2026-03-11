import json
import zipfile
import io
import requests
import pandas as pd
from pathlib import Path
from loguru import logger
import sys


BULK_URL = "https://download.open.fda.gov/drug/drugsfda/drug-drugsfda-0001-of-0001.json.zip"


def extract(output_dir=None):
    """Download and parse Drugs@FDA bulk data."""
    if output_dir is None:
        output_dir = Path("data/raw/drugs_at_fda")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "drugs_at_fda.parquet"

    if output_path.exists():
        size_mb = output_path.stat().st_size / 1_000_000
        logger.info(f"Already exists: {output_path} ({size_mb:.1f}MB)")
        return

    logger.info(f"Downloading Drugs@FDA bulk JSON from {BULK_URL}")
    resp = requests.get(BULK_URL, timeout=120)
    resp.raise_for_status()
    logger.info(f"Downloaded {len(resp.content) / 1_000_000:.1f}MB")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        json_filename = zf.namelist()[0]
        logger.info(f"Extracting {json_filename}...")
        with zf.open(json_filename) as f:
            data = json.load(f)

    results = data.get("results", [])
    logger.info(f"Loaded {len(results):,} drug application records")

    rows = []
    for record in results:
        app_number = record.get("application_number", "")
        sponsor = record.get("sponsor_name", "")
        openfda = record.get("openfda", {})
        brand_names = openfda.get("brand_name", [])
        generic_names = openfda.get("generic_name", [])
        manufacturer = openfda.get("manufacturer_name", [])
        route = openfda.get("route", [])
        substance = openfda.get("substance_name", [])
        ndc_codes = openfda.get("ndc", [])
        pharm_class = openfda.get("pharm_class_epc", [])

        app_type = ""
        if app_number.startswith("N"):
            app_type = "NDA"
        elif app_number.startswith("A"):
            app_type = "ANDA"
        elif app_number.startswith("B"):
            app_type = "BLA"
        else:
            app_type = "OTHER"

        for product in record.get("products", []):
            active_ingredients = [
                ai.get("name", "") for ai in product.get("active_ingredients", [])
            ]

            # Find the original approval submission
            approval_date = None
            for submission in record.get("submissions", []):
                if submission.get("submission_type") == "ORIG":
                    approval_date = submission.get("submission_status_date")
                    break

            # If no ORIG found, take the earliest submission date
            if not approval_date:
                dates = [
                    s.get("submission_status_date")
                    for s in record.get("submissions", [])
                    if s.get("submission_status_date")
                ]
                if dates:
                    approval_date = min(dates)

            rows.append({
                "application_number": app_number,
                "application_type": app_type,
                "sponsor_name": sponsor,
                "brand_name": "; ".join(brand_names) if brand_names else None,
                "generic_name": "; ".join(generic_names) if generic_names else None,
                "active_ingredients": "; ".join(active_ingredients),
                "dosage_form": product.get("dosage_form"),
                "product_route": "; ".join(route) if route else None,
                "pharm_class": "; ".join(pharm_class) if pharm_class else None,
                "manufacturer": "; ".join(manufacturer) if manufacturer else None,
                "ndc_codes": "; ".join(ndc_codes) if ndc_codes else None,
                "approval_date": approval_date,
                "marketing_status": product.get("marketing_status"),
            })

    df = pd.DataFrame(rows)
    df["approval_date"] = pd.to_datetime(df["approval_date"], errors="coerce")

    df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1_000_000
    logger.info(f"Saved {len(df):,} product records → {output_path} ({size_mb:.1f}MB)")

    # Summary
    logger.info("--- Summary ---")
    logger.info(f"Total products: {len(df):,}")
    logger.info(f"NDA (innovator): {(df['application_type'] == 'NDA').sum():,}")
    logger.info(f"ANDA (generic): {(df['application_type'] == 'ANDA').sum():,}")
    logger.info(f"BLA (biologic): {(df['application_type'] == 'BLA').sum():,}")
    logger.info(f"With NDC codes: {df['ndc_codes'].notna().sum():,}")
    logger.info(f"Date range: {df['approval_date'].min()} to {df['approval_date'].max()}")


def verify(output_dir=None):
    """Quick verification of Drugs@FDA data."""
    if output_dir is None:
        output_dir = Path("data/raw/drugs_at_fda")
    else:
        output_dir = Path(output_dir)

    path = output_dir / "drugs_at_fda.parquet"
    if not path.exists():
        logger.error("File not found. Run extract first.")
        return

    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Application types: {df['application_type'].value_counts().to_dict()}")
    logger.info(f"Sample brand names: {df['brand_name'].dropna().head(5).tolist()}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if action == "extract":
        extract()
    elif action == "verify":
        verify()
    else:
        print("Usage: python -m src.extract.drugs_at_fda [extract|verify]")
