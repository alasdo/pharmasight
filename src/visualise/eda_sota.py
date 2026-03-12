import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


# =========================
# Styling
# =========================
plt.rcParams.update({
    "figure.facecolor": "#0a0a12",
    "axes.facecolor": "#12121a",
    "axes.edgecolor": "#1e1e2e",
    "axes.labelcolor": "#a1a1b5",
    "text.color": "#e2e8f0",
    "xtick.color": "#a1a1b5",
    "ytick.color": "#a1a1b5",
    "grid.color": "#2a2a3a",
    "grid.alpha": 0.35,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#0a0a12",
})

ACCENT = "#4fd1c5"
ACCENT2 = "#f97316"
ACCENT3 = "#a78bfa"
ACCENT4 = "#f43f5e"
ACCENT5 = "#22d3ee"
MUTED = "#94a3b8"

OUTPUT_DIR = Path("figures/eda_sota")


# =========================
# Loading helpers
# =========================
def load_fact():
    logger.info("Loading fact_demand...")
    fact = pd.read_parquet("data/processed/fact_demand.parquet")
    fact["date"] = pd.to_datetime(fact["date"])
    return fact


def load_optional_parquet(path_str: str, label: str):
    path = Path(path_str)
    if not path.exists():
        logger.warning(f"{label} not found at {path} — skipping dependent analyses")
        return None
    logger.info(f"Loading {label}...")
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def quarter_label(d):
    return f"{d.year}-Q{((d.month - 1) // 3) + 1}"


def series_id_from_df(df: pd.DataFrame) -> pd.Series:
    return df["state"].astype(str) + " | " + df["ndc_11"].astype(str)


def safe_divide(a, b):
    return np.where((b != 0) & (~pd.isna(b)), a / b, np.nan)


# =========================
# 1. Panel structure
# =========================
def eda1_panel_structure(fact: pd.DataFrame):
    """
    Understand the structural shape of the forecasting panel:
    span, completeness, intermittency, non-zero history.
    """
    ensure_output_dir()

    df = fact[["state", "ndc_11", "date", "number_of_prescriptions"]].copy()
    df["series_id"] = series_id_from_df(df)

    panel = df.groupby("series_id").agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_periods=("date", "nunique"),
        total_rx=("number_of_prescriptions", "sum"),
        nonzero_periods=("number_of_prescriptions", lambda s: (s > 0).sum()),
    ).reset_index()

    panel["span_quarters"] = (
        (panel["last_date"].dt.year - panel["first_date"].dt.year) * 4
        + (panel["last_date"].dt.quarter - panel["first_date"].dt.quarter)
        + 1
    )
    panel["completeness"] = panel["n_periods"] / panel["span_quarters"].clip(lower=1)
    panel["intermittency"] = 1 - (
        panel["nonzero_periods"] / panel["n_periods"].clip(lower=1)
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(
        panel["span_quarters"].dropna(),
        bins=24,
        color=ACCENT,
        alpha=0.8,
        edgecolor="#1e1e2e",
    )
    axes[0].set_title("Series Span (quarters)", fontweight="bold")
    axes[0].set_xlabel("Span")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.25)

    axes[1].hist(
        panel["completeness"].dropna(),
        bins=30,
        color=ACCENT3,
        alpha=0.8,
        edgecolor="#1e1e2e",
    )
    axes[1].set_title("Panel Completeness", fontweight="bold")
    axes[1].set_xlabel("Observed / possible quarters")
    axes[1].grid(alpha=0.25)

    axes[2].hist(
        panel["intermittency"].dropna(),
        bins=30,
        color=ACCENT2,
        alpha=0.8,
        edgecolor="#1e1e2e",
    )
    axes[2].set_title("Intermittency", fontweight="bold")
    axes[2].set_xlabel("Fraction of observed quarters with zero Rx")
    axes[2].grid(alpha=0.25)

    for ax in axes:
        ax.set_axisbelow(True)

    stats_text = (
        f"series={len(panel):,} | median span={panel['span_quarters'].median():.1f}q | "
        f"median completeness={panel['completeness'].median():.2f} | "
        f"median intermittency={panel['intermittency'].median():.2f}"
    )
    fig.suptitle(stats_text, fontsize=10, color=MUTED, y=0.01)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda1_panel_structure.png")
    plt.close()
    logger.info("Saved eda1_panel_structure.png")


# =========================
# 2. Suppression bias audit
# =========================
def eda2_suppression_bias(fact: pd.DataFrame):
    """
    Measure whether suppression disproportionately affects low-volume rows/states.
    """
    ensure_output_dir()

    if "is_suppressed" not in fact.columns:
        logger.warning("No is_suppressed column — skipping")
        return

    df = fact.copy()

    state_summary = df.groupby("state").agg(
        suppression_rate=("is_suppressed", "mean"),
        total_rx=("number_of_prescriptions", "sum"),
        n_rows=("is_suppressed", "size"),
    ).reset_index()

    rank_source = df["number_of_prescriptions"].fillna(0).rank(method="first")
    df["volume_decile"] = pd.qcut(
        rank_source,
        10,
        labels=[f"D{i}" for i in range(1, 11)],
        duplicates="drop",
    )

    by_decile = (
        df.groupby("volume_decile", observed=False)["is_suppressed"]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(
        np.log1p(state_summary["total_rx"]),
        state_summary["suppression_rate"] * 100,
        s=45,
        alpha=0.75,
        color=ACCENT,
    )
    axes[0].set_title("Suppression vs State Demand Volume", fontweight="bold")
    axes[0].set_xlabel("log(1 + total state prescriptions)")
    axes[0].set_ylabel("Suppression rate (%)")
    axes[0].grid(alpha=0.25)

    axes[1].bar(
        by_decile["volume_decile"].astype(str),
        by_decile["is_suppressed"] * 100,
        color=ACCENT2,
        alpha=0.85,
        edgecolor="#1e1e2e",
    )
    axes[1].set_title("Suppression by Row-Level Volume Decile", fontweight="bold")
    axes[1].set_xlabel("Volume decile")
    axes[1].set_ylabel("Suppression rate (%)")
    axes[1].grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda2_suppression_bias.png")
    plt.close()
    logger.info("Saved eda2_suppression_bias.png")


# =========================
# 3. Forecastability map
# =========================
def eda3_forecastability_map(fact: pd.DataFrame):
    """
    Plot each drug-state series in a forecastability space:
    scale, volatility, seasonality, intermittency.
    """
    ensure_output_dir()

    df = fact[["state", "ndc_11", "date", "number_of_prescriptions"]].copy()
    df["series_id"] = series_id_from_df(df)

    def seasonal_strength(x: pd.Series) -> float:
        x = x.sort_index()
        if len(x) < 8:
            return np.nan
        overall_var = np.var(x.values)
        if overall_var <= 0:
            return np.nan
        quarter_means = x.groupby(x.index.quarter).mean()
        if len(quarter_means) < 2:
            return np.nan
        seasonal_var = np.var(quarter_means.values)
        return seasonal_var / overall_var

    rows = []

    for sid, g in df.groupby("series_id"):
        g = g.sort_values("date")
        x = g.set_index("date")["number_of_prescriptions"].fillna(0)

        if len(x) < 4:
            continue

        mean_rx = x.mean()
        std_rx = x.std()
        cv = std_rx / mean_rx if mean_rx > 0 else np.nan
        intermittency = (x == 0).mean()
        s_strength = seasonal_strength(x)

        rows.append({
            "series_id": sid,
            "mean_rx": mean_rx,
            "cv": cv,
            "intermittency": intermittency,
            "seasonal_strength": s_strength,
        })

    meta = pd.DataFrame(rows)
    meta = meta.replace([np.inf, -np.inf], np.nan).dropna()

    if meta.empty:
        logger.warning("Forecastability map has no valid data — skipping")
        return

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    sc = ax.scatter(
        np.log1p(meta["mean_rx"]),
        meta["cv"].clip(upper=5),
        c=meta["seasonal_strength"].clip(upper=1),
        s=10 + 60 * (1 - meta["intermittency"]),
        alpha=0.55,
    )

    ax.set_title("Forecastability Map", fontweight="bold")
    ax.set_xlabel("log(1 + mean prescriptions)")
    ax.set_ylabel("Coefficient of variation")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Seasonal strength")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda3_forecastability_map.png")
    plt.close()
    logger.info("Saved eda3_forecastability_map.png")


# =========================
# 4. Coverage heatmap
# =========================
def eda4_temporal_coverage_heatmap(fact: pd.DataFrame):
    """
    Heatmap of active drug coverage by state and quarter.
    This is a corrected and cleaner version of your current coverage plot.
    """
    ensure_output_dir()

    active = fact[fact["number_of_prescriptions"] > 0].copy()

    coverage = (
        active.groupby(["state", "date"])["ndc_11"]
        .nunique()
        .reset_index(name="active_ndcs")
    )

    pivot = coverage.pivot(index="state", columns="date", values="active_ndcs").fillna(0)

    if pivot.empty:
        logger.warning("No data for temporal coverage heatmap — skipping")
        return

    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", interpolation="nearest")

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    dates = list(pivot.columns)
    xticks = list(range(0, len(dates), max(1, len(dates) // 12 or 1)))
    ax.set_xticks(xticks)
    ax.set_xticklabels([quarter_label(dates[i]) for i in xticks], rotation=45, fontsize=8)

    cbar = plt.colorbar(im, ax=ax, label="Unique Active NDCs", shrink=0.8)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("Drug Coverage by State and Quarter", fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda4_temporal_coverage_heatmap.png")
    plt.close()
    logger.info("Saved eda4_temporal_coverage_heatmap.png")


# =========================
# 5. Seasonality strength
# =========================
def eda5_seasonality_strength(fact: pd.DataFrame):
    """
    Stronger, more robust seasonality scan using Q1/Q3 ratios.
    """
    ensure_output_dir()

    df = fact.copy()
    df["quarter"] = df["date"].dt.quarter

    national = (
        df.groupby(["date", "quarter", "product_name"])["number_of_prescriptions"]
        .sum()
        .reset_index()
    )

    q1 = national[national["quarter"] == 1].groupby("product_name")["number_of_prescriptions"].mean()
    q3 = national[national["quarter"] == 3].groupby("product_name")["number_of_prescriptions"].mean()

    seasonal_ratio = q1 / q3
    seasonal_ratio = seasonal_ratio.replace([np.inf, -np.inf], np.nan).dropna()
    seasonal_ratio = seasonal_ratio[seasonal_ratio > 0]
    seasonal_ratio = seasonal_ratio[q1.reindex(seasonal_ratio.index) > 1000]

    if seasonal_ratio.empty:
        logger.warning("No valid seasonal ratio data — skipping")
        return

    most_seasonal = seasonal_ratio.sort_values(ascending=False).head(15)
    most_antiseasonal = seasonal_ratio.sort_values(ascending=True).head(15)

    most_seasonal = most_seasonal.replace([np.inf, -np.inf], np.nan).dropna()
    most_antiseasonal = most_antiseasonal.replace([np.inf, -np.inf], np.nan).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(
        most_seasonal.index[::-1],
        most_seasonal.values[::-1],
        color=ACCENT,
        alpha=0.85,
        edgecolor="#1e1e2e",
    )
    axes[0].axvline(1.0, color=MUTED, linestyle="--", linewidth=1)
    axes[0].set_xlabel("Q1 / Q3 Demand Ratio")
    axes[0].set_title("Most Seasonal (Winter-Peaking)", fontweight="bold")
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(
        most_antiseasonal.index[::-1],
        most_antiseasonal.values[::-1],
        color=ACCENT2,
        alpha=0.85,
        edgecolor="#1e1e2e",
    )
    axes[1].axvline(1.0, color=MUTED, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Q1 / Q3 Demand Ratio")
    axes[1].set_title("Most Anti-Seasonal (Summer-Peaking)", fontweight="bold")
    axes[1].grid(axis="x", alpha=0.3)

    for ax in axes:
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda5_seasonality_strength.png")
    plt.close()
    logger.info("Saved eda5_seasonality_strength.png")


# =========================
# 6. Shortage event study
# =========================
def eda6_shortage_event_study(fact: pd.DataFrame):
    """
    Event-study view of demand before and after first shortage quarter.
    More informative than raw shortage time-series snapshots.
    """
    ensure_output_dir()

    supply = load_optional_parquet("data/processed/feat_supply.parquet", "feat_supply")
    if supply is None or "shortage_active" not in supply.columns:
        return

    s = supply.copy()
    s = s[s["shortage_active"] == 1].sort_values(["ndc_11", "date"])

    if s.empty:
        logger.warning("No active shortages found — skipping")
        return

    first_shortage = (
        s.groupby("ndc_11")["date"]
        .min()
        .rename("event_date")
        .reset_index()
    )

    national = (
        fact.groupby(["ndc_11", "date"])["number_of_prescriptions"]
        .sum()
        .reset_index()
    )

    merged = national.merge(first_shortage, on="ndc_11", how="inner")
    merged["event_q"] = (
        (merged["date"].dt.year - merged["event_date"].dt.year) * 4
        + (merged["date"].dt.quarter - merged["event_date"].dt.quarter)
    )

    window = 4
    merged = merged[(merged["event_q"] >= -window) & (merged["event_q"] <= window)].copy()

    baseline = (
        merged[merged["event_q"] == -1][["ndc_11", "number_of_prescriptions"]]
        .rename(columns={"number_of_prescriptions": "baseline_rx"})
    )

    merged = merged.merge(baseline, on="ndc_11", how="inner")
    merged = merged[merged["baseline_rx"] > 0].copy()

    if merged.empty:
        logger.warning("No valid baseline for shortage event study — skipping")
        return

    merged["indexed_rx"] = merged["number_of_prescriptions"] / merged["baseline_rx"] * 100

    summary = merged.groupby("event_q")["indexed_rx"].agg(
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(summary["event_q"], summary["median"], color=ACCENT2, linewidth=2.5)
    ax.fill_between(
        summary["event_q"],
        summary["q25"],
        summary["q75"],
        color=ACCENT2,
        alpha=0.18,
    )

    ax.axvline(0, linestyle="--", color=MUTED, linewidth=1)
    ax.axhline(100, linestyle="--", color=MUTED, linewidth=1)

    ax.set_title("Shortage Event Study", fontweight="bold")
    ax.set_xlabel("Quarters relative to first shortage")
    ax.set_ylabel("Demand index (baseline = 100 at t-1)")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda6_shortage_event_study.png")
    plt.close()
    logger.info("Saved eda6_shortage_event_study.png")


# =========================
# 7. ILI lead-lag scan
# =========================
def eda7_ili_lead_lag_scan(fact: pd.DataFrame):
    """
    Scan which lag of ILI aligns best with demand at state-quarter level.
    """
    ensure_output_dir()

    disease = load_optional_parquet("data/processed/feat_disease.parquet", "feat_disease")
    if disease is None or "ili_rate_mean" not in disease.columns:
        return

    if "state" not in disease.columns or "date" not in disease.columns:
        logger.warning("feat_disease missing state/date columns — skipping")
        return

    rx = (
        fact.groupby(["state", "date"])["number_of_prescriptions"]
        .sum()
        .reset_index()
    )
    ili = (
        disease.groupby(["state", "date"])["ili_rate_mean"]
        .mean()
        .reset_index()
    )

    merged = rx.merge(ili, on=["state", "date"], how="inner").sort_values(["state", "date"])

    if merged.empty:
        logger.warning("No overlapping demand/ILI data — skipping")
        return

    rows = []

    for lag in range(-4, 5):
        tmp = merged.copy()
        tmp["ili_shifted"] = tmp.groupby("state")["ili_rate_mean"].shift(lag)
        tmp = tmp.dropna(subset=["ili_shifted", "number_of_prescriptions"])

        if len(tmp) < 10:
            continue

        corr = tmp["number_of_prescriptions"].corr(tmp["ili_shifted"], method="spearman")
        rows.append({"lag": lag, "spearman_corr": corr})

    res = pd.DataFrame(rows)

    if res.empty:
        logger.warning("No valid lead-lag scan results — skipping")
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(
        res["lag"],
        res["spearman_corr"],
        color=ACCENT3,
        alpha=0.85,
        edgecolor="#1e1e2e",
    )
    ax.axhline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_title("Lead-Lag Scan: Demand vs ILI", fontweight="bold")
    ax.set_xlabel("Lag in quarters (ILI shifted by lag)")
    ax.set_ylabel("Spearman correlation")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda7_ili_lead_lag_scan.png")
    plt.close()
    logger.info("Saved eda7_ili_lead_lag_scan.png")


# =========================
# 8. Competition structure
# =========================
def eda8_generic_competition_advanced(fact: pd.DataFrame):
    """
    Cleaner competition plot with explicit observed=False and better defaults.
    """
    ensure_output_dir()

    if "num_generic_competitors" not in fact.columns:
        logger.warning("No num_generic_competitors column — skipping")
        return

    df = fact.copy()

    df["competition_bin"] = pd.cut(
        df["num_generic_competitors"],
        bins=[-1, 0, 3, 10, 50, 1000],
        labels=["None", "1-3", "4-10", "11-50", "50+"],
    )

    agg = (
        df.groupby(["date", "competition_bin"], observed=False)
        .agg(
            mean_rx=("number_of_prescriptions", "mean"),
            total_rx=("number_of_prescriptions", "sum"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "None": MUTED,
        "1-3": ACCENT,
        "4-10": ACCENT2,
        "11-50": ACCENT3,
        "50+": ACCENT4,
    }

    for comp_bin in ["None", "1-3", "4-10", "11-50", "50+"]:
        g = agg[agg["competition_bin"] == comp_bin].sort_values("date")
        if g.empty:
            continue

        axes[0].plot(
            g["date"],
            g["mean_rx"],
            label=comp_bin,
            color=colors[comp_bin],
            linewidth=2,
        )
        axes[1].plot(
            g["date"],
            g["total_rx"] / 1e6,
            label=comp_bin,
            color=colors[comp_bin],
            linewidth=2,
        )

    axes[0].set_title("Average Demand by Generic Competition Level", fontweight="bold")
    axes[0].set_ylabel("Mean Rx per row")
    axes[0].grid(alpha=0.3)
    axes[0].legend(title="Competitors", fontsize=9, framealpha=0.3)

    axes[1].set_title("Total Volume by Competition Level", fontweight="bold")
    axes[1].set_ylabel("Total Rx (millions)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(title="Competitors", fontsize=9, framealpha=0.3)

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda8_generic_competition_advanced.png")
    plt.close()
    logger.info("Saved eda8_generic_competition_advanced.png")


# =========================
# 9. Archetype gallery
# =========================
def eda9_series_archetypes(fact: pd.DataFrame):
    """
    Curated representative national demand series:
    high-volume stable, high-volume volatile, seasonal.
    """
    ensure_output_dir()

    national = (
        fact.groupby(["product_name", "date"])["number_of_prescriptions"]
        .sum()
        .reset_index()
    )

    stats = national.groupby("product_name")["number_of_prescriptions"].agg(
        mean="mean",
        std="std",
        max="max",
        nonzero=lambda s: (s > 0).sum(),
    ).reset_index()

    stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)

    valid = stats[stats["nonzero"] >= 8].copy()
    if valid.empty:
        logger.warning("No sufficient product histories for archetypes — skipping")
        return

    stable = valid.sort_values(["cv", "mean"], ascending=[True, False]).head(1)
    volatile = valid.sort_values(["cv", "mean"], ascending=[False, False]).head(1)

    fact_q = fact.copy()
    fact_q["quarter"] = fact_q["date"].dt.quarter
    q_national = fact_q.groupby(["product_name", "quarter"])["number_of_prescriptions"].mean().reset_index()
    quarter_spread = q_national.groupby("product_name")["number_of_prescriptions"].agg(
        q_mean="mean",
        q_std="std"
    ).reset_index()
    quarter_spread["seasonality_score"] = quarter_spread["q_std"] / quarter_spread["q_mean"].replace(0, np.nan)
    seasonal_name = quarter_spread.sort_values("seasonality_score", ascending=False)["product_name"].head(1)

    picks = pd.concat([
        stable["product_name"],
        volatile["product_name"],
        seasonal_name,
    ]).drop_duplicates().tolist()

    if not picks:
        logger.warning("No archetype picks found — skipping")
        return

    fig, axes = plt.subplots(len(picks), 1, figsize=(12, 3.5 * len(picks)), sharex=True)
    if len(picks) == 1:
        axes = [axes]

    colors = [ACCENT, ACCENT2, ACCENT3]

    for ax, drug, color in zip(axes, picks, colors):
        g = national[national["product_name"] == drug].sort_values("date")
        ax.plot(g["date"], g["number_of_prescriptions"], color=color, linewidth=2)
        ax.fill_between(g["date"], 0, g["number_of_prescriptions"], color=color, alpha=0.15)
        ax.set_title(drug, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda9_series_archetypes.png")
    plt.close()
    logger.info("Saved eda9_series_archetypes.png")


# =========================
# 10. Data source join coverage
# =========================
def eda10_external_feature_coverage(fact: pd.DataFrame):
    """
    Lightweight coverage audit for optional external feature tables.
    Shows whether source tables are present and how much date overlap exists.
    """
    ensure_output_dir()

    source_specs = [
        ("feat_supply", "data/processed/feat_supply.parquet"),
        ("feat_disease", "data/processed/feat_disease.parquet"),
        ("feat_regulation", "data/processed/feat_regulation.parquet"),
    ]

    fact_dates = set(pd.to_datetime(fact["date"]).dropna().unique())

    rows = []
    for name, path_str in source_specs:
        path = Path(path_str)
        if not path.exists():
            rows.append({
                "source": name,
                "exists": 0,
                "rows": 0,
                "date_overlap_pct": 0,
            })
            continue

        df = pd.read_parquet(path)
        if "date" in df.columns:
            df_dates = set(pd.to_datetime(df["date"]).dropna().unique())
            overlap_pct = (len(fact_dates.intersection(df_dates)) / max(len(fact_dates), 1)) * 100
        else:
            overlap_pct = 0

        rows.append({
            "source": name,
            "exists": 1,
            "rows": len(df),
            "date_overlap_pct": overlap_pct,
        })

    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(
        summary["source"],
        summary["rows"].replace(0, np.nan).fillna(0) / 1e6,
        color=[ACCENT if x == 1 else MUTED for x in summary["exists"]],
        alpha=0.85,
        edgecolor="#1e1e2e",
    )
    axes[0].set_title("External Table Size", fontweight="bold")
    axes[0].set_ylabel("Rows (millions)")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        summary["source"],
        summary["date_overlap_pct"],
        color=[ACCENT3 if x == 1 else MUTED for x in summary["exists"]],
        alpha=0.85,
        edgecolor="#1e1e2e",
    )
    axes[1].set_title("Date Overlap with fact_demand", fontweight="bold")
    axes[1].set_ylabel("Overlap (%)")
    axes[1].grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda10_external_feature_coverage.png")
    plt.close()
    logger.info("Saved eda10_external_feature_coverage.png")


# =========================
# Runner
# =========================
def run_all():
    fact = load_fact()

    eda1_panel_structure(fact)
    eda2_suppression_bias(fact)
    eda3_forecastability_map(fact)
    eda4_temporal_coverage_heatmap(fact)
    eda5_seasonality_strength(fact)
    eda6_shortage_event_study(fact)
    eda7_ili_lead_lag_scan(fact)
    eda8_generic_competition_advanced(fact)
    eda9_series_archetypes(fact)
    eda10_external_feature_coverage(fact)

    logger.info(f"All SOTA EDA figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    ensure_output_dir()

    action = sys.argv[1] if len(sys.argv) > 1 else "all"
    fact = load_fact()

    if action == "all":
        run_all()
    else:
        func = globals().get(action)
        if func:
            if "fact" in func.__code__.co_varnames:
                func(fact)
            else:
                func()
        else:
            print(f"Unknown analysis: {action}")