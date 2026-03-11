import requests
import pandas as pd
from pathlib import Path
from loguru import logger
import time
import sys


CT_BASE = "https://clinicaltrials.gov/api/v2/studies"


def extract(output_dir=None, max_results=5000):
    """Extract clinical trial data for drug pipeline intelligence."""
    if output_dir is None:
        output_dir = Path("data/raw/clinical_trials")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "clinical_trials.parquet"
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1_000_000
        logger.info(f"Already exists: {output_path} ({size_mb:.1f}MB)")
        return

    all_studies = []

    for phase in ["PHASE3", "PHASE4"]:
        logger.info(f"Fetching {phase} trials...")
        page_token = None
        phase_count = 0

        while phase_count < max_results:
            params = {
                "filter.advanced": f"AREA[Phase]{phase}",
                "filter.overallStatus": "COMPLETED",
                "pageSize": 100,
            }
            if page_token:
                params["pageToken"] = page_token

            try:
                resp = requests.get(CT_BASE, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"Error: {e}")
                break

            studies = data.get("studies", [])
            if not studies:
                break

            for study in studies:
                protocol = study.get("protocolSection", {})
                ident = protocol.get("identificationModule", {})
                status = protocol.get("statusModule", {})
                design = protocol.get("designModule", {})
                sponsor = protocol.get("sponsorCollaboratorsModule", {})
                arms = protocol.get("armsInterventionsModule", {})
                conditions = protocol.get("conditionsModule", {})

                interventions = arms.get("interventions", [])
                drug_names = [
                    i.get("name", "") for i in interventions
                    if i.get("type") == "DRUG"
                ]

                completion_struct = status.get("primaryCompletionDateStruct", {})
                start_struct = status.get("startDateStruct", {})

                all_studies.append({
                    "nct_id": ident.get("nctId"),
                    "title": ident.get("officialTitle") or ident.get("briefTitle"),
                    "status": status.get("overallStatus"),
                    "phase": "|".join(design.get("phases", [])),
                    "primary_completion": completion_struct.get("date"),
                    "start_date": start_struct.get("date"),
                    "sponsor": sponsor.get("leadSponsor", {}).get("name"),
                    "drug_names": "; ".join(drug_names) if drug_names else None,
                    "conditions": "; ".join(conditions.get("conditions", [])),
                    "num_drugs": len(drug_names),
                })

            phase_count += len(studies)
            page_token = data.get("nextPageToken")

            if phase_count % 500 < 100:
                logger.info(f"  {phase}: {phase_count:,} trials so far...")

            if not page_token:
                break

            time.sleep(0.3)

        logger.info(f"  {phase}: {phase_count:,} trials fetched")

    logger.info(f"Total trials: {len(all_studies):,}")

    if not all_studies:
        logger.error("No trials retrieved.")
        return

    df = pd.DataFrame(all_studies)
    if "primary_completion" in df.columns:
        df["primary_completion"] = pd.to_datetime(df["primary_completion"], errors="coerce")
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")

    df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1_000_000
    logger.info(f"Saved {len(df):,} trials → {output_path} ({size_mb:.1f}MB)")

    logger.info("--- ClinicalTrials.gov Summary ---")
    logger.info(f"Total trials: {len(df):,}")
    logger.info(f"By phase: {df['phase'].value_counts().to_dict()}")
    logger.info(f"With drug names: {df['drug_names'].notna().sum():,}")
    logger.info(f"Unique sponsors: {df['sponsor'].nunique():,}")
    if "primary_completion" in df.columns:
        logger.info(f"Date range: {df['primary_completion'].min()} to {df['primary_completion'].max()}")


def verify(output_dir=None):
    if output_dir is None:
        output_dir = Path("data/raw/clinical_trials")
    path = output_dir / "clinical_trials.parquet"
    if not path.exists():
        logger.error("File not found. Run extract first.")
        return
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} rows")
    logger.info(f"By phase: {df['phase'].value_counts().to_dict()}")
    logger.info(f"Top sponsors: {df['sponsor'].value_counts().head(5).to_dict()}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if action == "extract":
        extract()
    elif action == "verify":
        verify()
    else:
        print("Usage: python -m src.extract.clinical_trials [extract|verify]")
