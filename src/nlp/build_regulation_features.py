import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys


def build_drug_level_regulation_features(output_dir=None):
    """Build drug-specific regulation features from the processed corpus."""
    if output_dir is None:
        output_dir = Path("data/processed")

    logger.info("Building drug-level regulation features...")

    # Load the processed regulation corpus
    corpus = pd.read_parquet("data/processed/text_regulation_corpus.parquet")
    logger.info(f"  Regulation corpus: {len(corpus):,} documents")

    # Load drug names from demand data for matching
    fact = pd.read_parquet(
        "data/processed/fact_demand.parquet",
        columns=["product_name"]
    )
    sdud_names = set(fact["product_name"].dropna().str.strip().str.lower().unique())
    logger.info(f"  SDUD drug names: {len(sdud_names):,}")

    # Parse dates
    corpus["publication_date"] = pd.to_datetime(corpus["publication_date"], errors="coerce")
    corpus["year"] = corpus["publication_date"].dt.year
    corpus["quarter"] = corpus["publication_date"].dt.quarter
    corpus["date"] = pd.to_datetime(
        corpus["year"].astype(str) + "-"
        + ((corpus["quarter"] - 1) * 3 + 1).astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )

    # ── Part 1: Market-wide regulation features (join on date only) ──
    # These capture the overall regulatory environment
    logger.info("  Building market-wide features...")

    quarterly_market = corpus.groupby("date").agg(
        reg_total_docs=("title", "count"),
        reg_rule_count=("type", lambda x: (x == "Rule").sum()),
        reg_proposed_count=("type", lambda x: (x == "Proposed Rule").sum()),
        reg_notice_count=("type", lambda x: (x == "Notice").sum()),
    ).reset_index()

    # Ratio features (more informative than raw counts)
    quarterly_market["reg_rule_ratio"] = (
        quarterly_market["reg_rule_count"] / quarterly_market["reg_total_docs"].clip(lower=1)
    )
    quarterly_market["reg_proposed_ratio"] = (
        quarterly_market["reg_proposed_count"] / quarterly_market["reg_total_docs"].clip(lower=1)
    )

    # Event type counts per quarter (only the meaningful categories)
    for cat in ["approval", "manufacturing", "policy", "pricing", "safety", "shortage"]:
        quarterly_market[f"reg_evt_{cat}"] = corpus.groupby("date").apply(
            lambda g: g["primary_event"].eq(cat).sum()
        ).values

    logger.info(f"  Market-wide features: {len(quarterly_market)} quarters, {len(quarterly_market.columns)} columns")

    # ── Part 2: Drug-specific regulation features ──
    # Explode drug_mentions so each drug gets its own row
    logger.info("  Building drug-specific features...")

    docs_with_drugs = corpus[corpus["num_drug_mentions"] > 0].copy()
    logger.info(f"  Documents with drug mentions: {len(docs_with_drugs):,}")

    # Explode the drug_mentions list
    exploded = docs_with_drugs.explode("drug_mentions")
    exploded = exploded.rename(columns={"drug_mentions": "drug_name"})
    exploded["drug_name"] = exploded["drug_name"].str.strip().str.lower()
    logger.info(f"  Exploded drug-doc pairs: {len(exploded):,}")

    # Match to SDUD names using prefix matching (same logic as shortage matching)
    def match_to_sdud(drug_name, sdud_names_set):
        if not isinstance(drug_name, str) or len(drug_name) < 4:
            return None
        # Exact match
        if drug_name in sdud_names_set:
            return drug_name
        # Prefix match: SDUD name starts with regulation drug name
        for sdud_name in sdud_names_set:
            if len(sdud_name) >= 6 and (sdud_name.startswith(drug_name) or drug_name.startswith(sdud_name)):
                return sdud_name
        return None

    # Build a cached lookup (avoid N×M comparison)
    reg_drug_names = exploded["drug_name"].unique()
    drug_match_map = {}
    for reg_name in reg_drug_names:
        match = match_to_sdud(reg_name, sdud_names)
        if match:
            drug_match_map[reg_name] = match

    logger.info(f"  Regulation drugs matched to SDUD: {len(drug_match_map)}/{len(reg_drug_names)}")
    if drug_match_map:
        logger.info(f"  Sample matches: {dict(list(drug_match_map.items())[:10])}")

    exploded["sdud_match"] = exploded["drug_name"].map(drug_match_map)
    matched = exploded[exploded["sdud_match"].notna()].copy()
    logger.info(f"  Matched drug-doc pairs: {len(matched):,}")

    if len(matched) > 0:
        # Aggregate per drug per quarter
        drug_quarterly = matched.groupby(["date", "sdud_match"]).agg(
            reg_drug_doc_count=("title", "count"),
            reg_drug_approval_count=("primary_event", lambda x: (x == "approval").sum()),
            reg_drug_safety_count=("primary_event", lambda x: (x == "safety").sum()),
            reg_drug_manufacturing_count=("primary_event", lambda x: (x == "manufacturing").sum()),
        ).reset_index()
        drug_quarterly = drug_quarterly.rename(columns={"sdud_match": "product_name_lower"})

        logger.info(f"  Drug-quarterly features: {len(drug_quarterly):,} rows")
        logger.info(f"  Unique drugs with regulation features: {drug_quarterly['product_name_lower'].nunique()}")
    else:
        drug_quarterly = pd.DataFrame()
        logger.warning("  No drug-level matches found")

    # ── Save both feature tables ──

    # Market-wide (joins on date)
    market_path = output_dir / "feat_regulation.parquet"
    quarterly_market.to_parquet(market_path, index=False)
    logger.info(f"  Saved market-wide regulation: {len(quarterly_market)} rows -> {market_path}")

    # Drug-specific (joins on date + product_name)
    if len(drug_quarterly) > 0:
        drug_path = output_dir / "feat_regulation_drug.parquet"
        drug_quarterly.to_parquet(drug_path, index=False)
        logger.info(f"  Saved drug-specific regulation: {len(drug_quarterly)} rows -> {drug_path}")

    return quarterly_market, drug_quarterly


if __name__ == "__main__":
    build_drug_level_regulation_features()
