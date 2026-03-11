import pandas as pd
import requests
import zipfile
import io
from pathlib import Path
from loguru import logger
import sys


OB_URL = "https://www.fda.gov/media/76860/download?attachment"


def extract(output_dir=None):
    """Download and parse Orange Book data files."""
    if output_dir is None:
        output_dir = Path("data/raw/orange_book")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = list(output_dir.glob("*.parquet"))
    if len(existing) >= 2:
        logger.info(f"Already downloaded: {[f.name for f in existing]}")
        return

    logger.info(f"Downloading Orange Book from {OB_URL}")
    resp = requests.get(OB_URL, timeout=60, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    logger.info(f"Downloaded {len(resp.content) / 1_000_000:.1f}MB")

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            logger.info(f"Files in archive: {zf.namelist()}")
            for filename in zf.namelist():
                if filename.endswith(".txt"):
                    with zf.open(filename) as f:
                        try:
                            df = pd.read_csv(f, sep="~", dtype=str, encoding="latin-1")
                        except Exception:
                            f.seek(0)
                            df = pd.read_csv(f, sep="\t", dtype=str, encoding="latin-1")

                    clean_name = Path(filename).stem.lower()
                    out_path = output_dir / f"ob_{clean_name}.parquet"
                    df.to_parquet(out_path, index=False)
                    logger.info(f"  {clean_name}: {len(df):,} rows, {len(df.columns)} cols → {out_path.name}")
    except zipfile.BadZipFile:
        # Might not be a ZIP — could be a direct file
        logger.warning("Not a ZIP file — trying as direct text download")
        content = resp.content.decode("latin-1")
        # Try tilde-separated first
        from io import StringIO
        df = pd.read_csv(StringIO(content), sep="~", dtype=str)
        out_path = output_dir / "ob_products.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"  products: {len(df):,} rows → {out_path.name}")


def verify(output_dir=None):
    """Quick verification of Orange Book data."""
    if output_dir is None:
        output_dir = Path("data/raw/orange_book")
    else:
        output_dir = Path(output_dir)

    files = sorted(output_dir.glob("*.parquet"))
    if not files:
        logger.error("No files found. Run extract first.")
        return

    for f in files:
        df = pd.read_parquet(f)
        logger.info(f"{f.name}: {len(df):,} rows, columns: {list(df.columns)[:8]}...")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if action == "extract":
        extract()
    elif action == "verify":
        verify()
    else:
        print("Usage: python -m src.extract.orange_book [extract|verify]")
