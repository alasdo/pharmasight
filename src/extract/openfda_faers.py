import requests
import json
import zipfile
import io
import pandas as pd
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import os
import sys

load_dotenv()


def flatten_faers_records(records):
    """Flatten deeply nested FAERS JSON into flat rows."""
    rows = []
    for report in records:
        report_id = report.get("safetyreportid")
        receive_date = report.get("receivedate")
        serious = report.get("serious")
        death = report.get("seriousnessdeath")

        patient = report.get("patient", {})
        drugs = patient.get("drug", [])
        reactions = patient.get("reaction", [])

        reaction_terms = [r.get("reactionmeddrapt", "") for r in reactions]

        for drug in drugs:
            openfda = drug.get("openfda", {})
            generic_names = openfda.get("generic_name", [])
            brand_names = openfda.get("brand_name", [])

            rows.append({
                "report_id": report_id,
                "receive_date": receive_date,
                "is_serious": serious,
                "is_death": death,
                "drug_name": drug.get("medicinalproduct"),
                "generic_name": "; ".join(generic_names) if generic_names else None,
                "brand_name": "; ".join(brand_names) if brand_names else None,
                "drug_characterization": drug.get("drugcharacterization"),
                "drug_indication": drug.get("drugindication"),
                "reactions": "; ".join(reaction_terms),
                "num_reactions": len(reaction_terms),
            })
    return rows


def extract(output_dir=None, max_partitions=10):
    """Download FAERS bulk data files."""
    if output_dir is None:
        output_dir = Path("data/raw/openfda_faers")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    existing = list(output_dir.glob("faers_part_*.parquet"))
    if len(existing) >= max_partitions:
        logger.info(f"Already have {len(existing)} partitions. Skipping.")
        return

    logger.info("Fetching FAERS download index...")
    resp = requests.get("https://api.fda.gov/download.json", timeout=30)
    resp.raise_for_status()
    data = resp.json()

    partitions = data["results"]["drug"]["event"]["partitions"]
    logger.info(f"Available partitions: {len(partitions)}")

    # Take the most recent partitions
    partitions_to_download = partitions[-max_partitions:]
    logger.info(f"Downloading {len(partitions_to_download)} most recent partitions...")

    for i, partition in enumerate(partitions_to_download):
        part_num = len(partitions) - max_partitions + i
        output_path = output_dir / f"faers_part_{part_num:04d}.parquet"

        if output_path.exists():
            logger.info(f"  Partition {part_num} already exists, skipping")
            continue

        file_url = partition["file"]
        file_size = partition.get("size_mb", "?")
        logger.info(f"  Downloading partition {i+1}/{len(partitions_to_download)} ({file_size}MB): {file_url}")

        try:
            resp = requests.get(file_url, timeout=180)
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                json_file = zf.namelist()[0]
                with zf.open(json_file) as f:
                    raw = json.load(f)

            records = raw.get("results", [])
            rows = flatten_faers_records(records)
            df = pd.DataFrame(rows)

            df["receive_date"] = pd.to_datetime(df["receive_date"], format="%Y%m%d", errors="coerce")

            df.to_parquet(output_path, index=False)
            logger.info(f"    Saved {len(df):,} records → {output_path.name}")

        except Exception as e:
            logger.error(f"    Failed: {e}")
            continue

    # Final summary
    all_files = sorted(output_dir.glob("faers_part_*.parquet"))
    total_rows = 0
    for f in all_files:
        df = pd.read_parquet(f, columns=["report_id"])
        total_rows += len(df)

    logger.info(f"--- FAERS Summary ---")
    logger.info(f"Partitions downloaded: {len(all_files)}")
    logger.info(f"Total records: {total_rows:,}")


def verify(output_dir=None):
    if output_dir is None:
        output_dir = Path("data/raw/openfda_faers")

    files = sorted(output_dir.glob("faers_part_*.parquet"))
    if not files:
        logger.error("No FAERS files found. Run extract first.")
        return

    total = 0
    for f in files:
        df = pd.read_parquet(f)
        total += len(df)
        logger.info(f"  {f.name}: {len(df):,} rows, {f.stat().st_size/1_000_000:.1f}MB")

    logger.info(f"Total: {len(files)} files, {total:,} records")

    # Sample from first file
    sample = pd.read_parquet(files[0])
    logger.info(f"Columns: {list(sample.columns)}")
    if "drug_characterization" in sample.columns:
        logger.info(f"Drug characterization: {sample['drug_characterization'].value_counts().to_dict()}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if action == "extract":
        extract()
    elif action == "verify":
        verify()
    else:
        print("Usage: python -m src.extract.openfda_faers [extract|verify]")
