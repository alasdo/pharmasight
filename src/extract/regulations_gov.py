import requests
import pandas as pd
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import os
import time
import sys

load_dotenv()


def extract(output_dir=None):
    """Extract FDA drug-related rulemaking dockets from Regulations.gov."""
    if output_dir is None:
        output_dir = Path("data/raw/regulations_gov")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "regulations_gov.parquet"
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1_000_000
        logger.info(f"Already exists: {output_path} ({size_mb:.1f}MB)")
        return

    api_key = os.getenv("REGULATIONS_GOV_API_KEY")
    if not api_key:
        logger.error("REGULATIONS_GOV_API_KEY not found in .env")
        return

    base_url = "https://api.regulations.gov/v4/documents"
    headers = {"X-Api-Key": api_key}

    search_terms = ["drug", "pharmaceutical", "medication", "prescription"]
    all_docs = []

    for term in search_terms:
        logger.info(f"Searching: \"{term}\"...")
        page = 1
        term_count = 0

        while True:
            params = {
                "filter[agencyId]": "FDA",
                "filter[searchTerm]": term,
                "page[size]": 25,
                "page[number]": page,
                "sort": "-postedDate",
            }

            try:
                resp = requests.get(base_url, headers=headers, params=params, timeout=30)

                if resp.status_code == 429:
                    logger.warning("Rate limited. Waiting 60s...")
                    time.sleep(60)
                    continue

                resp.raise_for_status()
                data = resp.json()

            except Exception as e:
                logger.error(f"Error on page {page}: {e}")
                break

            docs = data.get("data", [])
            if not docs:
                break

            for doc in docs:
                attrs = doc.get("attributes", {})
                all_docs.append({
                    "id": doc.get("id"),
                    "document_type": attrs.get("documentType"),
                    "title": attrs.get("title"),
                    "posted_date": attrs.get("postedDate"),
                    "comment_start": attrs.get("commentStartDate"),
                    "comment_end": attrs.get("commentEndDate"),
                    "docket_id": attrs.get("docketId"),
                    "agency_id": attrs.get("agencyId"),
                    "search_term": term,
                })

            term_count += len(docs)

            # Regulations.gov limits to 20 pages max
            total_pages = min(data.get("meta", {}).get("totalPages", 1), 20)
            if page >= total_pages:
                break

            page += 1
            time.sleep(0.5)

        logger.info(f"  \"{term}\": {term_count} documents")

    if not all_docs:
        logger.warning("No documents collected")
        return

    df = pd.DataFrame(all_docs)
    df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
    df["comment_start"] = pd.to_datetime(df["comment_start"], errors="coerce")
    df["comment_end"] = pd.to_datetime(df["comment_end"], errors="coerce")

    # Deduplicate across search terms
    before = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first")
    deduped = before - len(df)
    if deduped > 0:
        logger.info(f"Deduplicated: removed {deduped} duplicates across search terms")

    df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1_000_000
    logger.info(f"Saved {len(df)} documents → {output_path} ({size_mb:.1f}MB)")

    logger.info("--- Regulations.gov Summary ---")
    logger.info(f"Total documents: {len(df)}")
    logger.info(f"By type: {df['document_type'].value_counts().to_dict()}")
    logger.info(f"Date range: {df['posted_date'].min()} to {df['posted_date'].max()}")


def verify(output_dir=None):
    if output_dir is None:
        output_dir = Path("data/raw/regulations_gov")
    path = output_dir / "regulations_gov.parquet"
    if not path.exists():
        logger.error("File not found. Run extract first.")
        return
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows")
    logger.info(f"By type: {df['document_type'].value_counts().to_dict()}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if action == "extract":
        extract()
    elif action == "verify":
        verify()
    else:
        print("Usage: python -m src.extract.regulations_gov [extract|verify]")
