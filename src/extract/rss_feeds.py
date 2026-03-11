import requests
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime
import xml.etree.ElementTree as ET
import re
import sys


WORKING_FEEDS = {
    "biopharmadive": "https://www.biopharmadive.com/feeds/news/",
    "fiercepharma": "https://www.fiercepharma.com/rss/xml",
    "endpts": "https://endpts.com/feed/",
}


def parse_rss(content, feed_name):
    """Parse RSS or Atom XML into a list of entries."""
    entries = []
    try:
        root = ET.fromstring(content)

        # RSS 2.0
        for item in root.iter("item"):
            entries.append({
                "feed_name": feed_name,
                "title": (item.findtext("title") or "").strip(),
                "link": (item.findtext("link") or "").strip(),
                "published": (item.findtext("pubDate") or "").strip(),
                "summary": (item.findtext("description") or "").strip(),
            })

        # Atom
        if not entries:
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall(".//atom:entry", ns):
                link_el = entry.find("atom:link", ns)
                entries.append({
                    "feed_name": feed_name,
                    "title": (entry.findtext("atom:title", "", ns) or "").strip(),
                    "link": link_el.get("href", "") if link_el is not None else "",
                    "published": (entry.findtext("atom:published", "", ns) or entry.findtext("atom:updated", "", ns) or "").strip(),
                    "summary": (entry.findtext("atom:summary", "", ns) or entry.findtext("atom:content", "", ns) or "").strip(),
                })

    except ET.ParseError as e:
        logger.warning(f"  XML parse error for {feed_name}: {e}")

    return entries


def extract(output_dir=None):
    """Extract pharma news from working RSS feeds."""
    if output_dir is None:
        output_dir = Path("data/raw/fda_rss")
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    all_entries = []

    for feed_name, url in WORKING_FEEDS.items():
        logger.info(f"  Fetching {feed_name}...")
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            entries = parse_rss(resp.text, feed_name)
            logger.info(f"    {len(entries)} entries")
            all_entries.extend(entries)
        except Exception as e:
            logger.error(f"    Failed: {e}")

    if not all_entries:
        logger.warning("No entries collected from any feed")
        return

    df = pd.DataFrame(all_entries)
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    df["summary_clean"] = df["summary"].apply(lambda x: re.sub(r"<[^>]+>", "", str(x)).strip() if x else "")

    date_str = datetime.now().strftime("%Y%m%d")
    output_path = output_dir / f"pharma_rss_{date_str}.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} entries → {output_path.name}")

    logger.info("--- RSS Summary ---")
    logger.info(f"Total entries: {len(df)}")
    logger.info(f"By feed: {df['feed_name'].value_counts().to_dict()}")
    if df["published"].notna().any():
        logger.info(f"Date range: {df['published'].min()} to {df['published'].max()}")


def verify(output_dir=None):
    if output_dir is None:
        output_dir = Path("data/raw/fda_rss")
    else:
        output_dir = Path(output_dir)
    files = sorted(output_dir.glob("*.parquet"))
    if not files:
        logger.error("No files found.")
        return
    for f in files:
        df = pd.read_parquet(f)
        logger.info(f"{f.name}: {len(df)} entries")
        logger.info(f"  By feed: {df['feed_name'].value_counts().to_dict()}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if action == "extract":
        extract()
    elif action == "verify":
        verify()
    else:
        print("Usage: python -m src.extract.rss_feeds [extract|verify]")
