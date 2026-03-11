import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from transformers import pipeline as hf_pipeline
import spacy
import re
import sys
import torch

# ── Drug Name Dictionary ──
# Built from our dim_product table — we'll load it dynamically
DRUG_NAME_CACHE = None


def load_drug_names():
    """Load known drug names from dim_product for entity matching."""
    global DRUG_NAME_CACHE
    if DRUG_NAME_CACHE is not None:
        return DRUG_NAME_CACHE

    dim_path = Path("data/processed/dim_product.parquet")
    if dim_path.exists():
        dim = pd.read_parquet(dim_path)
        names = set()
        for col in ["ingredient", "ob_trade_name", "brand_name", "generic_name"]:
            if col in dim.columns:
                clean = dim[col].dropna().str.strip().str.lower().unique()
                names.update(clean)
        # Filter to names with 4+ characters
        DRUG_NAME_CACHE = {n for n in names if len(n) >= 4}
        logger.info(f"Loaded {len(DRUG_NAME_CACHE):,} known drug names")
    else:
        DRUG_NAME_CACHE = set()
        logger.warning("No dim_product found — drug matching will be limited")

    return DRUG_NAME_CACHE


# ── Event Categories ──
EVENT_CATEGORIES = {
    "approval": ["approv", "clearance", "authorized", "granted", "nda", "anda", "510(k)",
                  "new drug", "generic approv", "biosimilar"],
    "shortage": ["shortage", "unavailable", "out of stock", "supply disruption",
                 "allocation", "limited supply", "backordered"],
    "recall": ["recall", "withdrawn", "market withdrawal", "voluntary recall",
               "class i recall", "class ii recall"],
    "safety": ["adverse", "side effect", "warning", "black box", "rems",
               "contraindicated", "safety alert", "adverse event", "death"],
    "pricing": ["price", "pricing", "cost", "rebate", "reimbursement", "copay",
                "out-of-pocket", "inflation", "markup", "340b"],
    "manufacturing": ["manufacturing", "cgmp", "gmp", "facility", "inspection",
                      "production", "contamination", "quality"],
    "policy": ["regulation", "proposed rule", "final rule", "guidance",
               "medicaid", "medicare", "formulary", "coverage"],
}


def classify_event(text):
    """Keyword-based event classification. Returns list of matched categories."""
    if not isinstance(text, str):
        return []
    text_lower = text.lower()
    matched = []
    for category, keywords in EVENT_CATEGORIES.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(category)
    return matched if matched else ["other"]


def extract_drug_mentions(text, drug_names=None):
    """Extract drug name mentions from text using dictionary matching."""
    if not isinstance(text, str) or not text.strip():
        return []
    if drug_names is None:
        drug_names = load_drug_names()

    text_lower = text.lower()
    mentions = []
    for name in drug_names:
        if name in text_lower:
            # Avoid partial matches (e.g., "met" matching inside "method")
            pattern = r'\b' + re.escape(name) + r'\b'
            if re.search(pattern, text_lower):
                mentions.append(name)

    return list(set(mentions))


def build_sentiment_pipeline():
    """Initialize the zero-shot sentiment pipeline using a small, fast model."""
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Loading sentiment model (device={'GPU' if device == 0 else 'CPU'})...")

    # Use a small distilled model for speed
    sentiment = hf_pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
        truncation=True,
        max_length=512,
    )
    logger.info("Sentiment model loaded")
    return sentiment


def score_sentiment(texts, sentiment_pipeline, batch_size=64):
    """Score sentiment for a list of texts. Returns list of (label, score) tuples."""
    if not texts:
        return []

    # Clean and truncate texts
    clean_texts = []
    for t in texts:
        if isinstance(t, str) and t.strip():
            clean_texts.append(t[:512])
        else:
            clean_texts.append("neutral")

    results = sentiment_pipeline(clean_texts, batch_size=batch_size)

    scored = []
    for r in results:
        label = r["label"]  # POSITIVE or NEGATIVE
        score = r["score"]
        # Convert to -1 to 1 scale
        if label == "NEGATIVE":
            scored.append(-score)
        else:
            scored.append(score)

    return scored


def process_federal_register(output_dir=None):
    """Process Federal Register documents: event classification + sentiment + drug mentions."""
    if output_dir is None:
        output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing Federal Register corpus...")

    fr_path = Path("data/raw/federal_register/federal_register.parquet")
    if not fr_path.exists():
        logger.error("Federal Register data not found")
        return

    df = pd.read_parquet(fr_path)
    logger.info(f"  Documents: {len(df):,}")

    # Combine title + abstract for analysis
    df["text"] = df["title"].fillna("") + " " + df["abstract"].fillna("")
    df["text"] = df["text"].str.strip()

    # Filter to documents with meaningful text
    df = df[df["text"].str.len() > 20].copy()
    logger.info(f"  With text: {len(df):,}")

    # Load drug names
    drug_names = load_drug_names()

    # 1. Event classification
    logger.info("  Classifying events...")
    df["event_categories"] = df["text"].apply(classify_event)
    df["primary_event"] = df["event_categories"].apply(lambda x: x[0] if x else "other")

    event_counts = df["primary_event"].value_counts()
    logger.info(f"  Event distribution: {event_counts.to_dict()}")

    # 2. Drug mention extraction
    logger.info("  Extracting drug mentions...")
    df["drug_mentions"] = df["text"].apply(lambda t: extract_drug_mentions(t, drug_names))
    df["num_drug_mentions"] = df["drug_mentions"].apply(len)

    docs_with_drugs = (df["num_drug_mentions"] > 0).sum()
    logger.info(f"  Documents mentioning drugs: {docs_with_drugs:,}/{len(df):,}")

    # 3. Sentiment scoring
    logger.info("  Scoring sentiment...")
    sentiment_pipeline = build_sentiment_pipeline()
    df["sentiment_score"] = score_sentiment(df["text"].tolist(), sentiment_pipeline)

    logger.info(f"  Mean sentiment: {df['sentiment_score'].mean():.3f}")
    logger.info(f"  Sentiment std: {df['sentiment_score'].std():.3f}")

    # 4. Aggregate to quarterly features
    logger.info("  Aggregating to quarterly features...")
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df["year"] = df["publication_date"].dt.year
    df["quarter"] = df["publication_date"].dt.quarter
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-"
        + ((df["quarter"] - 1) * 3 + 1).astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )

    # Quarterly aggregation — regulation-level features
    quarterly = df.groupby("date").agg(
        reg_doc_count=("text", "count"),
        reg_rule_count=("type", lambda x: (x == "Rule").sum()),
        reg_proposed_rule_count=("type", lambda x: (x == "Proposed Rule").sum()),
        reg_notice_count=("type", lambda x: (x == "Notice").sum()),
        reg_sentiment_mean=("sentiment_score", "mean"),
        reg_sentiment_std=("sentiment_score", "std"),
        reg_sentiment_min=("sentiment_score", "min"),
        reg_drug_mention_count=("num_drug_mentions", "sum"),
        reg_docs_with_drugs=("num_drug_mentions", lambda x: (x > 0).sum()),
    ).reset_index()

    # Event category counts per quarter
    for cat in EVENT_CATEGORIES.keys():
        quarterly[f"reg_evt_{cat}"] = df.groupby("date")["event_categories"].apply(
            lambda groups: sum(cat in cats for cats in groups)
        ).values if len(quarterly) == df.groupby("date").ngroups else 0

    output_path = output_dir / "feat_regulation.parquet"
    quarterly.to_parquet(output_path, index=False)
    logger.info(f"  Saved feat_regulation: {len(quarterly):,} rows → {output_path}")

    # Also save the document-level data for later analysis
    doc_path = output_dir / "text_regulation_corpus.parquet"
    df[["publication_date", "date", "title", "type", "primary_event",
        "drug_mentions", "num_drug_mentions", "sentiment_score"]].to_parquet(doc_path, index=False)
    logger.info(f"  Saved regulation corpus: {len(df):,} rows → {doc_path}")

    return quarterly


def process_news(output_dir=None):
    """Process RSS feeds + Twitter: sentiment + drug mentions + event classification."""
    if output_dir is None:
        output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing news corpus...")

    all_texts = []

    # RSS feeds
    rss_dir = Path("data/raw/fda_rss")
    for f in rss_dir.glob("*.parquet"):
        df = pd.read_parquet(f)
        df["source_type"] = "rss"
        df["text"] = df["title"].fillna("") + " " + df.get("summary_clean", df.get("summary", "")).fillna("")
        df["published_date"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
        all_texts.append(df[["text", "published_date", "source_type", "feed_name"]].copy())

    # Twitter
    twitter_dir = Path("data/raw/twitter")
    for f in twitter_dir.glob("tweets_*.parquet"):
        df = pd.read_parquet(f)
        df["source_type"] = "twitter"
        df["feed_name"] = "twitter"
        df["text"] = df["text"].fillna("")
        df["published_date"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        all_texts.append(df[["text", "published_date", "source_type", "feed_name"]].copy())

    if not all_texts:
        logger.warning("No news/social data found")
        return

    corpus = pd.concat(all_texts, ignore_index=True)
    corpus = corpus[corpus["text"].str.len() > 10].copy()
    logger.info(f"  News corpus: {len(corpus):,} documents")

    # Load drug names
    drug_names = load_drug_names()

    # 1. Event classification
    logger.info("  Classifying events...")
    corpus["event_categories"] = corpus["text"].apply(classify_event)
    corpus["primary_event"] = corpus["event_categories"].apply(lambda x: x[0] if x else "other")

    # 2. Drug mentions
    logger.info("  Extracting drug mentions...")
    corpus["drug_mentions"] = corpus["text"].apply(lambda t: extract_drug_mentions(t, drug_names))
    corpus["num_drug_mentions"] = corpus["drug_mentions"].apply(len)

    # 3. Sentiment
    logger.info("  Scoring sentiment...")
    sentiment_pipeline = build_sentiment_pipeline()
    corpus["sentiment_score"] = score_sentiment(corpus["text"].tolist(), sentiment_pipeline)

    # 4. Aggregate to quarterly
    corpus["year"] = corpus["published_date"].dt.year
    corpus["quarter"] = corpus["published_date"].dt.quarter
    corpus["date"] = pd.to_datetime(
        corpus["year"].astype(str) + "-"
        + ((corpus["quarter"] - 1) * 3 + 1).astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )

    quarterly = corpus.groupby("date").agg(
        news_doc_count=("text", "count"),
        news_rss_count=("source_type", lambda x: (x == "rss").sum()),
        news_twitter_count=("source_type", lambda x: (x == "twitter").sum()),
        news_sentiment_mean=("sentiment_score", "mean"),
        news_sentiment_std=("sentiment_score", "std"),
        news_sentiment_min=("sentiment_score", "min"),
        news_drug_mention_count=("num_drug_mentions", "sum"),
        news_docs_with_drugs=("num_drug_mentions", lambda x: (x > 0).sum()),
    ).reset_index()

    # Event counts
    for cat in EVENT_CATEGORIES.keys():
        quarterly[f"news_evt_{cat}"] = corpus.groupby("date")["event_categories"].apply(
            lambda groups: sum(cat in cats for cats in groups)
        ).values if len(quarterly) == corpus.groupby("date").ngroups else 0

    output_path = output_dir / "feat_news.parquet"
    quarterly.to_parquet(output_path, index=False)
    logger.info(f"  Saved feat_news: {len(quarterly):,} rows → {output_path}")

    # Save corpus
    doc_path = output_dir / "text_news_corpus.parquet"
    corpus[["published_date", "date", "text", "source_type", "feed_name",
            "primary_event", "drug_mentions", "num_drug_mentions", "sentiment_score"]].to_parquet(doc_path, index=False)
    logger.info(f"  Saved news corpus: {len(corpus):,} rows → {doc_path}")

    return quarterly


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "all"

    if action in ("all", "regulation"):
        process_federal_register()
    if action in ("all", "news"):
        process_news()
