import pandas as pd
from pathlib import Path
from loguru import logger
import yaml
import sys


SDUD_URLS = {
    2019: "https://download.medicaid.gov/data/SDUD2019.csv",
    2020: "https://download.medicaid.gov/data/SDUD2020.csv",
    2021: "https://download.medicaid.gov/data/SDUD2021.csv",
    2022: "https://download.medicaid.gov/data/SDUD2022.csv",
    2023: "https://download.medicaid.gov/data/SDUD2023.csv",
}


def extract(years=None, output_dir=None):
    """Download SDUD CSV files for given years."""
    if output_dir is None:
        output_dir = Path("data/raw/medicaid_sdud")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if years is None:
        years = list(SDUD_URLS.keys())

    for year in years:
        url = SDUD_URLS.get(year)
        if not url:
            logger.warning(f"No URL configured for year {year}")
            continue

        output_path = output_dir / f"sdud_{year}.csv"
        if output_path.exists():
            size_mb = output_path.stat().st_size / 1_000_000
            logger.info(f"Already downloaded: {output_path} ({size_mb:.1f}MB)")
            continue

        logger.info(f"Downloading SDUD {year} from {url}")
        logger.info(f"This file is large (500MB-2GB) — please be patient...")

        try:
            # Stream download to handle large files
            import requests
            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            downloaded = 0

            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1_000_000):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % 50_000_000 < 1_000_000:
                        pct = (downloaded / total) * 100
                        logger.info(f"  {year}: {pct:.0f}% ({downloaded // 1_000_000}MB / {total // 1_000_000}MB)")

            size_mb = output_path.stat().st_size / 1_000_000
            logger.info(f"Saved {output_path} ({size_mb:.1f}MB)")

        except Exception as e:
            logger.error(f"Failed to download {year}: {e}")
            if output_path.exists():
                output_path.unlink()
            continue


def verify(output_dir=None):
    """Quick verification of downloaded files."""
    if output_dir is None:
        output_dir = Path("data/raw/medicaid_sdud")
    else:
        output_dir = Path(output_dir)

    files = sorted(output_dir.glob("sdud_*.csv"))

    if not files:
        logger.error("No SDUD files found. Run extract first.")
        return

    logger.info(f"Found {len(files)} SDUD files:")
    for f in files:
        size_mb = f.stat().st_size / 1_000_000
        # Read just the first few rows to verify structure
        try:
            sample = pd.read_csv(f, nrows=5, dtype=str)
            cols = len(sample.columns)
            logger.info(f"  {f.name}: {size_mb:.1f}MB, {cols} columns")
            if size_mb < 1:
                logger.warning(f"  WARNING: {f.name} seems too small — may be incomplete")
        except Exception as e:
            logger.error(f"  {f.name}: ERROR reading file — {e}")

    # Show column names from first file
    sample = pd.read_csv(files[0], nrows=2, dtype=str)
    logger.info(f"Columns: {list(sample.columns)}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "extract"

    if action == "extract":
        extract()
    elif action == "verify":
        verify()
    else:
        print(f"Usage: python -m src.extract.medicaid_sdud [extract|verify]")
