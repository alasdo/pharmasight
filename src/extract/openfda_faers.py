import io
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DEFAULT_OUTPUT_DIR = Path("data/raw/openfda_faers")
DOWNLOAD_INDEX_URL = "https://api.fda.gov/download.json"
INDEX_TIMEOUT = 30
FILE_TIMEOUT = 180

# Target FAERS window
TARGET_START_YEAR = 2019
TARGET_END_YEAR = 2023


def setup_logger() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )


def safe_list(value: Any) -> List[Any]:
    """Normalize a JSON field into a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return [value]


def safe_join(values: Any, sep: str = "; ") -> Optional[str]:
    """Join a possibly nested/list field into a clean string."""
    vals = safe_list(values)
    cleaned = [str(v).strip() for v in vals if v is not None and str(v).strip()]
    return sep.join(cleaned) if cleaned else None


def parse_partition_year_quarter(file_url: str) -> Optional[Tuple[int, int]]:
    """
    Parse year and quarter from FAERS bulk partition file path.

    Expected patterns may look like:
      .../drug-event-0001-of-xxxx.json.zip
      .../faers_ascii_2019q1.zip
      .../2021q4/...
      .../2023Q2/...

    We specifically search for YYYYqN or YYYYQN.
    """
    match = re.search(r"(20\d{2})[qQ]([1-4])", file_url)
    if not match:
        return None

    year = int(match.group(1))
    quarter = int(match.group(2))
    return year, quarter


def fetch_partitions() -> List[Dict[str, Any]]:
    """Fetch FAERS drug/event partitions from the openFDA bulk index."""
    logger.info("Fetching FAERS bulk download index...")
    resp = requests.get(DOWNLOAD_INDEX_URL, timeout=INDEX_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    try:
        partitions = data["results"]["drug"]["event"]["partitions"]
    except KeyError as e:
        raise KeyError(
            "Could not find results['drug']['event']['partitions'] in download index."
        ) from e

    if not isinstance(partitions, list):
        raise TypeError("Partitions payload is not a list.")

    logger.info(f"Total partitions in index: {len(partitions)}")
    return partitions


def select_partitions_2019_2023(
    partitions: List[Dict[str, Any]],
    start_year: int = TARGET_START_YEAR,
    end_year: int = TARGET_END_YEAR,
) -> List[Tuple[int, Dict[str, Any], int, int]]:
    """
    Select only partitions whose filenames indicate a quarter between
    start_year and end_year inclusive.
    """
    selected: List[Tuple[int, Dict[str, Any], int, int]] = []
    unparsable: List[str] = []

    for idx, partition in enumerate(partitions):
        file_url = partition.get("file", "")
        parsed = parse_partition_year_quarter(file_url)

        if parsed is None:
            unparsable.append(file_url)
            continue

        year, quarter = parsed
        if start_year <= year <= end_year:
            selected.append((idx, partition, year, quarter))

    selected.sort(key=lambda x: (x[2], x[3]))  # sort by year, quarter ascending

    logger.info(
        f"Selected {len(selected)} partitions for {start_year}-{end_year} inclusive."
    )

    if not selected:
        logger.warning(
            "No partitions matched YYYYqN pattern in the requested range. "
            "Check the actual file naming in the bulk index."
        )

    if unparsable:
        logger.info(f"Skipped {len(unparsable)} partitions with no parsable YYYYqN pattern.")

    return selected


def flatten_faers_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten nested FAERS JSON into one row per (report, drug)."""
    rows: List[Dict[str, Any]] = []

    for report in records:
        if not isinstance(report, dict):
            continue

        report_id = report.get("safetyreportid")
        receive_date = report.get("receivedate")
        serious = report.get("serious")
        death = report.get("seriousnessdeath")

        patient = report.get("patient") or {}
        if not isinstance(patient, dict):
            patient = {}

        drugs = safe_list(patient.get("drug"))
        reactions = safe_list(patient.get("reaction"))

        reaction_terms: List[str] = []
        for reaction in reactions:
            if isinstance(reaction, dict):
                term = reaction.get("reactionmeddrapt")
                if term is not None and str(term).strip():
                    reaction_terms.append(str(term).strip())

        if not drugs:
            rows.append(
                {
                    "report_id": report_id,
                    "receive_date": receive_date,
                    "is_serious": serious,
                    "is_death": death,
                    "drug_name": None,
                    "generic_name": None,
                    "brand_name": None,
                    "drug_characterization": None,
                    "drug_indication": None,
                    "reactions": "; ".join(reaction_terms) if reaction_terms else None,
                    "num_reactions": len(reaction_terms),
                }
            )
            continue

        for drug in drugs:
            if not isinstance(drug, dict):
                continue

            openfda = drug.get("openfda") or {}
            if not isinstance(openfda, dict):
                openfda = {}

            rows.append(
                {
                    "report_id": report_id,
                    "receive_date": receive_date,
                    "is_serious": serious,
                    "is_death": death,
                    "drug_name": drug.get("medicinalproduct"),
                    "generic_name": safe_join(openfda.get("generic_name")),
                    "brand_name": safe_join(openfda.get("brand_name")),
                    "drug_characterization": drug.get("drugcharacterization"),
                    "drug_indication": drug.get("drugindication"),
                    "reactions": "; ".join(reaction_terms) if reaction_terms else None,
                    "num_reactions": len(reaction_terms),
                }
            )

    return rows


def parse_receive_date(df: pd.DataFrame) -> pd.DataFrame:
    """Parse FAERS receive_date from YYYYMMDD into pandas datetime."""
    if "receive_date" not in df.columns:
        df["receive_date"] = pd.NaT
        return df

    df["receive_date"] = pd.to_datetime(
        df["receive_date"],
        format="%Y%m%d",
        errors="coerce",
    )
    return df


def filter_to_target_years(
    df: pd.DataFrame,
    start_year: int = TARGET_START_YEAR,
    end_year: int = TARGET_END_YEAR,
) -> pd.DataFrame:
    """Keep only rows whose receive_date falls within the target years inclusive."""
    if "receive_date" not in df.columns:
        return df.iloc[0:0].copy()

    start_date = pd.Timestamp(start_year, 1, 1)
    end_date = pd.Timestamp(end_year, 12, 31, 23, 59, 59)

    mask = df["receive_date"].notna() & (df["receive_date"] >= start_date) & (df["receive_date"] <= end_date)
    return df.loc[mask].copy()


def download_partition_zip(file_url: str) -> bytes:
    """Download a partition zip file."""
    resp = requests.get(file_url, timeout=FILE_TIMEOUT)
    resp.raise_for_status()
    return resp.content


def read_partition_json(zip_bytes: bytes) -> Dict[str, Any]:
    """Read the JSON payload inside the ZIP archive."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        if not names:
            raise ValueError("ZIP archive is empty.")

        json_candidates = [name for name in names if name.lower().endswith(".json")]
        target = json_candidates[0] if json_candidates else names[0]

        with zf.open(target) as f:
            return json.load(f)


def build_output_filename(year: int, quarter: int, partition_idx: int, file_url: str) -> str:
    """Create a stable parquet output filename."""
    stem = Path(file_url).stem
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem).strip("_")
    if not stem:
        stem = f"partition_{partition_idx:04d}"
    return f"faers_{year}q{quarter}_part_{partition_idx:04d}_{stem}.parquet"


def extract(
    output_dir: Optional[Path] = None,
    start_year: int = TARGET_START_YEAR,
    end_year: int = TARGET_END_YEAR,
    force_redownload: bool = False,
) -> None:
    """
    Download FAERS partitions for a specific year range (2019-2023),
    flatten them, filter rows by receive_date, and save parquet files.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    partitions = fetch_partitions()
    selected = select_partitions_2019_2023(
        partitions=partitions,
        start_year=start_year,
        end_year=end_year,
    )

    if not selected:
        logger.error("No target partitions selected. Extraction aborted.")
        return

    total_rows_saved = 0
    total_files_saved = 0

    logger.info(
        f"Downloading FAERS data for {start_year}-{end_year} "
        f"({len(selected)} quarter-partitions found)."
    )

    for partition_idx, partition, year, quarter in selected:
        file_url = partition.get("file")
        size_mb = partition.get("size_mb", "?")

        if not file_url:
            logger.warning(f"Skipping partition idx={partition_idx}: missing file URL.")
            continue

        output_name = build_output_filename(year, quarter, partition_idx, file_url)
        output_path = output_dir / output_name

        if output_path.exists() and not force_redownload:
            logger.info(f"Skipping existing file: {output_name}")
            continue

        logger.info(
            f"Downloading {year}Q{quarter} | idx={partition_idx} | size={size_mb}MB | {file_url}"
        )

        try:
            zip_bytes = download_partition_zip(file_url)
            raw = read_partition_json(zip_bytes)

            records = raw.get("results", [])
            if not isinstance(records, list):
                raise TypeError("JSON field 'results' is not a list.")

            rows = flatten_faers_records(records)
            df = pd.DataFrame(rows)

            if df.empty:
                logger.warning(f"{year}Q{quarter}: no rows after flattening.")
                continue

            before_filter = len(df)

            df = parse_receive_date(df)
            df = filter_to_target_years(df, start_year=start_year, end_year=end_year)

            after_filter = len(df)

            if df.empty:
                logger.warning(
                    f"{year}Q{quarter}: 0 rows remain after filtering to {start_year}-{end_year}."
                )
                continue

            df.sort_values(["receive_date", "report_id"], inplace=True, na_position="last")
            df.reset_index(drop=True, inplace=True)

            df.to_parquet(output_path, index=False)

            total_files_saved += 1
            total_rows_saved += after_filter

            logger.info(
                f"Saved {after_filter:,} rows "
                f"(from {before_filter:,}) -> {output_name}"
            )

        except Exception as e:
            logger.exception(f"Failed on {year}Q{quarter} | idx={partition_idx}: {e}")
            continue

    logger.info("--- Extraction Summary ---")
    logger.info(f"Target years: {start_year}-{end_year}")
    logger.info(f"Quarter partitions selected: {len(selected)}")
    logger.info(f"Files saved: {total_files_saved}")
    logger.info(f"Total rows saved: {total_rows_saved:,}")


def verify(
    output_dir: Optional[Path] = None,
    start_year: int = TARGET_START_YEAR,
    end_year: int = TARGET_END_YEAR,
) -> None:
    """Verify saved parquet files and confirm date coverage."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    files = sorted(output_dir.glob("faers_*.parquet"))
    if not files:
        logger.error("No parquet files found. Run extract first.")
        return

    total_rows = 0
    global_min = None
    global_max = None

    logger.info("--- File Summary ---")
    for f in files:
        df = pd.read_parquet(f)

        row_count = len(df)
        total_rows += row_count

        local_min = df["receive_date"].min() if "receive_date" in df.columns and not df.empty else pd.NaT
        local_max = df["receive_date"].max() if "receive_date" in df.columns and not df.empty else pd.NaT

        if pd.notna(local_min):
            global_min = local_min if global_min is None else min(global_min, local_min)
        if pd.notna(local_max):
            global_max = local_max if global_max is None else max(global_max, local_max)

        logger.info(
            f"{f.name} | rows={row_count:,} | size={f.stat().st_size / 1_000_000:.2f}MB"
        )

    logger.info("--- Dataset Summary ---")
    logger.info(f"Files: {len(files)}")
    logger.info(f"Total rows: {total_rows:,}")
    logger.info(f"Min receive_date: {global_min}")
    logger.info(f"Max receive_date: {global_max}")

    if global_min is not None and global_min.year < start_year:
        logger.warning(f"Found records earlier than {start_year}.")
    if global_max is not None and global_max.year > end_year:
        logger.warning(f"Found records later than {end_year}.")

    sample = pd.read_parquet(files[0])
    logger.info(f"Columns: {list(sample.columns)}")


if __name__ == "__main__":
    setup_logger()

    action = sys.argv[1] if len(sys.argv) > 1 else "extract"

    if action == "extract":
        force_redownload = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
        extract(force_redownload=force_redownload)

    elif action == "verify":
        verify()

    else:
        print(
            "Usage:\n"
            "  python -m src.extract.openfda_faers extract [force_redownload]\n"
            "  python -m src.extract.openfda_faers verify\n\n"
            "Examples:\n"
            "  python -m src.extract.openfda_faers extract\n"
            "  python -m src.extract.openfda_faers extract 1\n"
            "  python -m src.extract.openfda_faers verify"
        )