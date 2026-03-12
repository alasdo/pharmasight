import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from loguru import logger
import json
import sys

# ── Style ──
plt.rcParams.update({
    "figure.facecolor": "#0a0a12",
    "axes.facecolor": "#12121a",
    "axes.edgecolor": "#1e1e2e",
    "axes.labelcolor": "#8888a0",
    "text.color": "#e2e8f0",
    "xtick.color": "#8888a0",
    "ytick.color": "#8888a0",
    "grid.color": "#1e1e2e",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#0a0a12",
})

ACCENT = "#4fd1c5"
ACCENT2 = "#f97316"
ACCENT3 = "#a78bfa"
MUTED = "#8888a0"
BG = "#0a0a12"

OUTPUT_DIR = Path("figures")


def fig1_data_overview():
    """Figure 1: Data source overview — record counts by source."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = {
        "Medicaid SDUD": 25_338_823,
        "FAERS Adverse Events": 26_658_214,
        "FDA Recalls": 17_428,
        "Federal Register": 13_389,
        "ClinicalTrials.gov": 10_000,
        "Drugs@FDA": 50_859,
        "Orange Book": 47_780,
        "Regulations.gov": 1_749,
        "FDA Shortages": 1_683,
        "CDC FluView": 3_650,
        "Twitter/X": 37,
        "RSS Feeds": 59,
    }

    # Sort by count
    sources = dict(sorted(sources.items(), key=lambda x: x[1], reverse=True))

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        list(sources.keys()),
        list(sources.values()),
        color=ACCENT,
        alpha=0.85,
        edgecolor="#1e1e2e",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Records (log scale)")
    ax.set_title("PharmaSight: Data Sources by Record Count", fontweight="bold", pad=15)
    ax.invert_yaxis()

    # Add count labels
    for bar, (name, count) in zip(bars, sources.items()):
        label = f"{count:,.0f}" if count < 1_000_000 else f"{count / 1_000_000:.1f}M"
        ax.text(bar.get_width() * 1.3, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9, color=MUTED)

    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_data_overview.png")
    plt.close()
    logger.info("Saved fig1_data_overview.png")


def fig2_ablation_results():
    """Figure 2: Ablation study — MAE comparison across configurations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = Path("data/results/baseline_results.json")
    if not results_path.exists():
        logger.error("No results file found. Run train_baseline first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    models = [r["model"] + "\n" + r["config"] for r in results]
    maes = [r["mae"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Color map by config
    color_map = {"baseline": MUTED, "A": ACCENT, "B": ACCENT2, "D": ACCENT3}
    colors = [color_map.get(r["config"], MUTED) for r in results]

    # MAE comparison
    bars1 = ax1.bar(models, maes, color=colors, edgecolor="#1e1e2e", width=0.6)
    ax1.set_ylabel("Mean Absolute Error (Rx)")
    ax1.set_title("MAE by Model Configuration", fontweight="bold")
    for bar, mae in zip(bars1, maes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{mae:.1f}", ha="center", fontsize=10, color="#e2e8f0")

    # sMAPE comparison
    smapes = [r.get("smape", r.get("mape", 0)) for r in results]
    bars2 = ax2.bar(models, smapes, color=colors, edgecolor="#1e1e2e", width=0.6)
    ax2.set_ylabel("sMAPE (%)")
    ax2.set_title("sMAPE by Model Configuration", fontweight="bold")
    for bar, m in zip(bars2, smapes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{m:.1f}%", ha="center", fontsize=10, color="#e2e8f0")

    for ax in [ax1, ax2]:
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_ablation_results.png")
    plt.close()
    logger.info("Saved fig2_ablation_results.png")


def fig3_feature_importance():
    """Figure 3: Feature importance for best config (D)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = Path("data/results/baseline_results.json")
    if not results_path.exists():
        return

    with open(results_path) as f:
        results = json.load(f)

    # Find best config (D, then B, then A)
    for target in ["D", "B", "A"]:
        config = [r for r in results if r.get("config") == target]
        if config:
            break

    if not config:
        logger.warning("No model results found")
        return

    importance = config[0].get("feature_importance_top10", {})
    if not importance:
        logger.warning("No feature importance data")
        return

    config_name = config[0]["config"]

    name_map = {
        "rx_rolling_mean_4": "Rolling Mean (4Q)",
        "rx_lag_1": "Lag 1 Quarter",
        "rx_trend_4": "Trend (4Q)",
        "rx_lag_4": "Lag 4 Quarters",
        "rx_lag_2": "Lag 2 Quarters",
        "rx_rolling_std_4": "Rolling Std (4Q)",
        "rx_yoy_change": "Year-over-Year Δ",
        "total_recalls": "Recall Count ★",
        "year_num": "Year",
        "reimb_lag_1": "Reimbursement Lag",
        "quarter_sin": "Quarter (sin)",
        "quarter_cos": "Quarter (cos)",
        "ili_rate_mean": "ILI Rate ★",
        "ili_rate_yoy_change": "ILI YoY Δ ★",
        "shortage_active": "Shortage Active ★",
        "adverse_event_count": "Adverse Events ★",
        "num_generic_competitors": "Generic Competitors ★",
        "reg_rule_count": "Regulation Rules ★★",
        "reg_doc_count": "Regulation Docs ★★",
        "reg_drug_doc_count": "Drug-Specific Reg Docs ★★",
        "reg_drug_approval_count": "Drug Reg Approvals ★★",
        "reg_drug_safety_count": "Drug Reg Safety ★★",
        "reg_drug_manufacturing_count": "Drug Reg Manufacturing ★★",
        "reg_proposed_rule_count": "Proposed Rules ★★",
    }

    features = list(importance.keys())
    gains = list(importance.values())
    labels = [name_map.get(f, f) for f in features]

    # Color: demand=accent, structured external=accent2, NLP=accent3
    def get_color(f):
        if "★★" in name_map.get(f, ""):
            return ACCENT3
        if "★" in name_map.get(f, ""):
            return ACCENT2
        return ACCENT

    colors = [get_color(f) for f in features]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(labels[::-1], gains[::-1], color=colors[::-1],
                   edgecolor="#1e1e2e", alpha=0.85)

    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title(f"Config {config_name}: Top 10 Features by Importance", fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ACCENT, label="Demand features"),
        Patch(facecolor=ACCENT2, label="Structured external ★"),
        Patch(facecolor=ACCENT3, label="NLP regulation ★★"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_feature_importance.png")
    plt.close()
    logger.info("Saved fig3_feature_importance.png")

def fig4_error_by_tier():
    """Figure 4: Error analysis by prescription volume tier."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = Path("data/results/baseline_results.json")
    if not results_path.exists():
        return

    with open(results_path) as f:
        results = json.load(f)

    tiers = ["low (≤p50)", "mid (p50-p90)", "high (p90-p99)", "extreme (>p99)"]
    tier_short = ["Low\n(≤p50)", "Mid\n(p50-p90)", "High\n(p90-p99)", "Extreme\n(>p99)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(tiers))
    width = 0.25

    color_map = {"baseline": MUTED, "A": ACCENT, "B": ACCENT2, "D": ACCENT3}
    for idx, r in enumerate(results):
        label = f"{r['model']} {r['config']}"
        color = color_map.get(r["config"], MUTED)
        width = 0.8 / len(results)

        mae_vals = [r.get(f"mae_{t}", 0) for t in tiers]
        ax1.bar(x + idx * width, mae_vals, width, label=label, color=color,
                edgecolor="#1e1e2e", alpha=0.85)

        mape_vals = [r.get(f"mape_{t}", 0) for t in tiers]
        ax2.bar(x + idx * width, mape_vals, width, label=label, color=color,
                edgecolor="#1e1e2e", alpha=0.85)

    ax1.set_xticks(x + width)
    ax1.set_xticklabels(tier_short)
    ax1.set_ylabel("MAE (Rx)")
    ax1.set_title("MAE by Volume Tier", fontweight="bold")
    ax1.set_yscale("log")
    ax1.legend(fontsize=9, framealpha=0.3)
    ax1.grid(axis="y", alpha=0.3)

    ax2.set_xticks(x + width)
    ax2.set_xticklabels(tier_short)
    ax2.set_ylabel("MAPE (%)")
    ax2.set_title("MAPE by Volume Tier", fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.3)
    ax2.grid(axis="y", alpha=0.3)

    for ax in [ax1, ax2]:
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_error_by_tier.png")
    plt.close()
    logger.info("Saved fig4_error_by_tier.png")


def fig5_demand_timeseries():
    """Figure 5: Example time series — top drugs actual demand over time."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading fact_demand for time series plot...")
    fact = pd.read_parquet("data/processed/fact_demand.parquet",
                           columns=["date", "product_name", "number_of_prescriptions", "state"])

    # Aggregate nationally
    national = fact.groupby(["date", "product_name"])["number_of_prescriptions"].sum().reset_index()

    # Top 6 drugs
    top_drugs = (
        national.groupby("product_name")["number_of_prescriptions"]
        .sum()
        .sort_values(ascending=False)
        .head(6)
        .index.tolist()
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    colors_cycle = [ACCENT, ACCENT2, ACCENT3, "#f43f5e", "#22d3ee", "#facc15"]

    for i, drug in enumerate(top_drugs):
        ax = axes[i]
        drug_data = national[national["product_name"] == drug].sort_values("date")

        ax.plot(drug_data["date"], drug_data["number_of_prescriptions"] / 1e6,
                color=colors_cycle[i], linewidth=2, marker="o", markersize=4)
        ax.fill_between(drug_data["date"], 0,
                        drug_data["number_of_prescriptions"] / 1e6,
                        alpha=0.15, color=colors_cycle[i])

        ax.set_title(drug, fontsize=11, fontweight="bold")
        ax.set_ylabel("Rx (millions)")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

        # Format y axis
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.1f}M"))

    plt.suptitle("National Prescription Volume — Top 6 Drugs", fontweight="bold",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_demand_timeseries.png")
    plt.close()
    logger.info("Saved fig5_demand_timeseries.png")


def fig6_shortage_coverage():
    """Figure 6: Shortage status breakdown."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shortages = pd.read_parquet("data/raw/fda_shortages/fda_shortages.parquet")

    status_counts = shortages["status"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = [ACCENT2 if "Current" in s else ACCENT if "Resolved" in s else MUTED
              for s in status_counts.index]

    wedges, texts, autotexts = ax.pie(
        status_counts.values,
        labels=status_counts.index,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 11, "color": "#e2e8f0"},
        wedgeprops={"edgecolor": "#0a0a12", "linewidth": 2},
    )

    for t in autotexts:
        t.set_fontsize(10)
        t.set_color("#0a0a12")
        t.set_fontweight("bold")

    ax.set_title(f"FDA Drug Shortages (n={len(shortages):,})", fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_shortage_status.png")
    plt.close()
    logger.info("Saved fig6_shortage_status.png")


def fig7_star_schema():
    """Figure 7: Star schema diagram."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Central fact table
    fact_box = plt.Rectangle((3.5, 3), 3, 2, facecolor="#1e1e2e", edgecolor=ACCENT,
                              linewidth=2, zorder=3)
    ax.add_patch(fact_box)
    ax.text(5, 4.3, "fact_demand", ha="center", fontsize=13, fontweight="bold", color=ACCENT, zorder=4)
    ax.text(5, 3.7, "18.7M rows", ha="center", fontsize=10, color=MUTED, zorder=4)

    # Dimension / feature tables
    tables = [
        (0.5, 6, "dim_product\n47,780", ACCENT3),
        (7, 6, "dim_geography\n54 states", ACCENT3),
        (0, 1, "feat_disease\nILI rates", ACCENT2),
        (3.5, 0, "feat_supply\nshortages, patents", ACCENT2),
        (7.5, 1, "feat_safety\nFAERS, recalls", ACCENT2),
    ]

    for x, y, label, color in tables:
        box = plt.Rectangle((x, y), 2.5, 1.2, facecolor="#1e1e2e", edgecolor=color,
                              linewidth=1.5, zorder=3)
        ax.add_patch(box)
        ax.text(x + 1.25, y + 0.6, label, ha="center", va="center",
                fontsize=9, color=color, zorder=4)

    # Arrows
    arrow_kwargs = dict(arrowstyle="-|>", color=MUTED, lw=1.5, zorder=2)
    from matplotlib.patches import FancyArrowPatch
    arrows = [
        ((1.75, 6), (3.5, 5)),      # dim_product → fact
        ((7, 6), (6.5, 5)),          # dim_geography → fact
        ((1.25, 2.2), (3.5, 3)),     # feat_disease → fact
        ((4.75, 1.2), (5, 3)),       # feat_supply → fact
        ((8.75, 2.2), (6.5, 3)),     # feat_safety → fact
    ]
    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, **arrow_kwargs)
        ax.add_patch(arrow)

    ax.set_title("PharmaSight Star Schema", fontweight="bold", fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig7_star_schema.png")
    plt.close()
    logger.info("Saved fig7_star_schema.png")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    action = sys.argv[1] if len(sys.argv) > 1 else "all"

    if action == "all":
        fig1_data_overview()
        fig2_ablation_results()
        fig3_feature_importance()
        fig4_error_by_tier()
        fig5_demand_timeseries()
        fig6_shortage_coverage()
        fig7_star_schema()
    else:
        func = globals().get(action)
        if func:
            func()
        else:
            print(f"Unknown figure: {action}")

    logger.info(f"\nAll figures saved to {OUTPUT_DIR}/")
