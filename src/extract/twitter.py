import requests
import pandas as pd
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from datetime import datetime
import os
import time
import sys

load_dotenv()

# Core keywords — focused on high-signal pharma topics
SEARCH_KEYWORDS = [
    "drug shortage",
    "FDA approval",
    "FDA recall",
    "flu outbreak",
    "drug pricing",
    "generic drug approval",
]


def extract(output_dir=None, max_results_per_keyword=10):
    """Extract recent tweets for pharma-related keywords.

    Default: 6 keywords × 10 tweets = 60 tweets per run.
    At daily runs: ~1,800 tweets/month (well within 10K Basic limit).
    """
    if output_dir is None:
        output_dir = Path("data/raw/twitter")
    output_dir.mkdir(parents=True, exist_ok=True)

    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        logger.error("TWITTER_BEARER_TOKEN not found in .env")
        return

    headers = {
        "Authorization": f"Bearer {bearer_token}",
    }

    base_url = "https://api.x.com/2/tweets/search/recent"

    all_tweets = []
    api_calls = 0

    for keyword in SEARCH_KEYWORDS:
        logger.info(f"Searching: \"{keyword}\"...")

        params = {
            "query": f'"{keyword}" lang:en -is:retweet -is:reply',
            "max_results": max(max_results_per_keyword, 10),
            "tweet.fields": "created_at,public_metrics,author_id",
            "sort_order": "relevancy",
        }

        try:
            resp = requests.get(base_url, headers=headers, params=params, timeout=30)
            api_calls += 1

            if resp.status_code == 429:
                logger.warning("Rate limited. Stopping to preserve quota.")
                break

            if resp.status_code == 403:
                logger.error(f"403 Forbidden: {resp.text[:200]}")
                return

            resp.raise_for_status()
            data = resp.json()

        except Exception as e:
            logger.error(f"Error for \"{keyword}\": {e}")
            continue

        tweets = data.get("data", [])

        for tweet in tweets:
            metrics = tweet.get("public_metrics", {})
            all_tweets.append({
                "keyword": keyword,
                "tweet_id": tweet.get("id"),
                "text": tweet.get("text"),
                "created_at": tweet.get("created_at"),
                "author_id": tweet.get("author_id"),
                "retweet_count": metrics.get("retweet_count", 0),
                "reply_count": metrics.get("reply_count", 0),
                "like_count": metrics.get("like_count", 0),
                "quote_count": metrics.get("quote_count", 0),
                "impression_count": metrics.get("impression_count", 0),
            })

        logger.info(f"  Got {len(tweets)} tweets")
        time.sleep(1)

    if not all_tweets:
        logger.warning("No tweets collected")
        return

    df = pd.DataFrame(all_tweets)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    df["engagement_score"] = (
        df["like_count"]
        + df["retweet_count"] * 2
        + df["reply_count"] * 1.5
        + df["quote_count"] * 3
    )

    date_str = datetime.now().strftime("%Y%m%d")
    output_path = output_dir / f"tweets_{date_str}.parquet"
    df.to_parquet(output_path, index=False)

    logger.info(f"Saved {len(df)} tweets → {output_path.name}")
    logger.info(f"API calls used: {api_calls}")
    logger.info(f"Estimated monthly usage at daily runs: ~{api_calls * 30} calls, ~{len(df) * 30} tweets")
    logger.info(f"By keyword: {df['keyword'].value_counts().to_dict()}")


def verify(output_dir=None):
    if output_dir is None:
        output_dir = Path("data/raw/twitter")
    files = sorted(output_dir.glob("tweets_*.parquet"))
    if not files:
        logger.error("No files found. Run extract first.")
        return
    total = 0
    for f in files:
        df = pd.read_parquet(f)
        total += len(df)
        logger.info(f"{f.name}: {len(df)} tweets")
    logger.info(f"Total across all files: {total}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if action == "extract":
        extract()
    elif action == "verify":
        verify()
    else:
        print("Usage: python -m src.extract.twitter [extract|verify]")
