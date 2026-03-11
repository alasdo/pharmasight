import requests
import pandas as pd
from pathlib import Path
from loguru import logger
import time
import sys


FR_BASE = "https://www.federalregister.gov/api/v1"

PHARMA_AGENCIES = [
    "food-and-drug-administration",
    "centers-for-medicare-medicaid-services",
    "drug-enforcement-administration",
    "health-and-human-services-department",
]


def extract(start_date="2019-01-01", end_date="2025-12-31", output_dir=None):
    """Extract pharma-relevant Federal Register documents."""
    if output_dir is None:
        output_dir = Path("data/raw/federal_register")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "federal_register.parquet"
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1_000_000
        logger.info(f"Already exists: {output_path} ({size_mb:.1f}MB)")
        return

    all_docs = []

    for agency in PHARMA_AGENCIES:
        logger.info(f"Fetching documents for {agency}...")
        page = 1
        agency_count = 0

        while True:
            params = {
                "conditions[agencies][]": agency,
                "conditions[publication_date][gte]": start_date,
                "conditions[publication_date][lte]": end_date,
                "fields[]": [
                    "document_number", "title", "type", "abstract",
                    "publication_date", "effective_on", "comments_close_on",
                    "html_url",
                ],
                "per_page": 100,
                "page": page,
                "order": "oldest",
            }

            try:
                resp = requests.get(f"{FR_BASE}/documents.json", params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"  Error on page {page}: {e}")
                break

            results = data.get("results", [])
            if not results:
                break

            for doc in results:
                all_docs.append({
                    "document_number": doc.get("document_number"),
                    "title": doc.get("title"),
                    "type": doc.get("type"),
                    "abstract": doc.get("abstract"),
                    "publication_date": doc.get("publication_date"),
                    "effective_date": doc.get("effective_on"),
                    "comments_close": doc.get("comments_close_on"),
                    "url": doc.get("html_url"),
                    "agency": agency,
                })

            agency_count += len(results)
            total_pages = data.get("total_pages", 1)

            if page >= total_pages:
                break
            page += 1
            time.sleep(0.2)

        logger.info(f"  {agency}: {agency_count:,} documents")

    logger.info(f"Total documents fetched: {len(all_docs):,}")

    df = pd.DataFrame(all_docs)
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    df["comments_close"] = pd.to_datetime(df["comments_close"], errors="coerce")

    df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1_000_000
    logger.info(f"Saved {len(df):,} documents → {output_path} ({size_mb:.1f}MB)")

    # Summary
    logger.info("--- Federal Register Summary ---")
    logger.info(f"Total documents: {len(df):,}")
    logger.info(f"By type: {df['type'].value_counts().to_dict()}")
    logger.info(f"By agency: {df['agency'].value_counts().to_dict()}")
    logger.info(f"Date range: {df['publication_date'].min()} to {df['publication_date'].max()}")
    logger.info(f"With abstract: {df['abstract'].notna().sum():,}")


def verify(output_dir=None):
    if output_dir is None:
        output_dir = Path("data/raw/federal_register")
    path = output_dir / "federal_register.parquet"
    if not path.exists():
        logger.error("File not found. Run extract first.")
        return
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} rows")
    logger.info(f"By type: {df['type'].value_counts().to_dict()}")
    logger.info(f"By agency: {df['agency'].value_counts().head().to_dict()}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if action == "extract":
        extract()
    elif action == "verify":
        verify()
    else:
        print("Usage: python -m src.extract.federal_register [extract|verify]")
