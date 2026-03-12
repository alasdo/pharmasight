import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from loguru import logger
import sys

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
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#0a0a12",
})

ACCENT = "#4fd1c5"
ACCENT2 = "#f97316"
ACCENT3 = "#a78bfa"
MUTED = "#8888a0"
OUTPUT_DIR = Path("figures/eda")


def load_fact():
    logger.info("Loading fact_demand...")
    return pd.read_parquet("data/processed/fact_demand.parquet")


def load_features():
    logger.info("Loading fact_demand_features...")
    return pd.read_parquet("data/processed/fact_demand_features.parquet")


def eda1_target_distribution(fact):
    """Histogram of prescription counts — shows the heavy tail."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rx = fact["number_of_prescriptions"].dropna()
    rx_pos = rx[rx > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Raw distribution (clipped)
    ax1.hist(rx_pos.clip(upper=5000), bins=200, color=ACCENT, alpha=0.7, edgecolor="#1e1e2e")
    ax1.set_xlabel("Prescriptions per Quarter")
    ax1.set_ylabel("Count")
    ax1.set_title("Prescription Distribution (clipped at 5,000)", fontweight="bold")
    ax1.axvline(rx_pos.median(), color=ACCENT2, linestyle="--", linewidth=1.5,
                label=f"Median: {rx_pos.median():.0f}")
    ax1.axvline(rx_pos.mean(), color=ACCENT3, linestyle="--", linewidth=1.5,
                label=f"Mean: {rx_pos.mean():.0f}")
    ax1.legend(fontsize=9, framealpha=0.3)
    ax1.grid(alpha=0.3)

    # Log distribution
    ax2.hist(np.log1p(rx_pos), bins=200, color=ACCENT3, alpha=0.7, edgecolor="#1e1e2e")
    ax2.set_xlabel("log(1 + Prescriptions)")
    ax2.set_ylabel("Count")
    ax2.set_title("Log-Transformed Distribution", fontweight="bold")
    ax2.axvline(np.log1p(rx_pos.median()), color=ACCENT2, linestyle="--", linewidth=1.5,
                label=f"Median: {np.log1p(rx_pos.median()):.1f}")
    ax2.legend(fontsize=9, framealpha=0.3)
    ax2.grid(alpha=0.3)

    for ax in [ax1, ax2]:
        ax.set_axisbelow(True)

    stats_text = (f"n={len(rx_pos):,} | zeros={int((rx == 0).sum()):,} | "
                  f"p50={rx_pos.median():.0f} | p99={rx_pos.quantile(0.99):.0f} | "
                  f"max={rx_pos.max():,.0f}")
    fig.suptitle(stats_text, fontsize=10, color=MUTED, y=0.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda1_target_distribution.png")
    plt.close()
    logger.info("Saved eda1_target_distribution.png")


def eda2_temporal_coverage(fact):
    """Heatmap: number of active drugs per state per quarter."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    active = fact[fact["number_of_prescriptions"] > 0].copy()
    coverage = active.groupby(["state", "date"])["ndc_11"].nunique().reset_index()
    coverage_pivot = coverage.pivot(index="state", columns="date", values="ndc_11").fillna(0)

    # Sort states by total coverage
    coverage_pivot = coverage_pivot.loc[coverage_pivot.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(coverage_pivot.values, aspect="auto", cmap="viridis", interpolation="nearest")

    ax.set_yticks(range(len(coverage_pivot.index)))
    ax.set_yticklabels(coverage_pivot.index, fontsize=8)
    dates = coverage_pivot.columns
    ax.set_xticks(range(0, len(dates), 2))
    ax.set_xticklabels(
        [
            f"{d.year}-Q{((d.month - 1) // 3) + 1}" if hasattr(d, "year") else str(d)
            for d in dates[::2]
        ],
        rotation=45,
        fontsize=8,
    )

    plt.colorbar(im, ax=ax, label="Unique Active NDCs", shrink=0.8)
    ax.set_title("Drug Coverage by State and Quarter", fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda2_temporal_coverage.png")
    plt.close()
    logger.info("Saved eda2_temporal_coverage.png")


def eda3_geographic_concentration(fact):
    """Top 20 states by total prescription volume + Lorenz curve."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    state_vol = fact.groupby("state")["number_of_prescriptions"].sum().sort_values(ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart: top 20 states
    top20 = state_vol.head(20)
    ax1.barh(top20.index[::-1], top20.values[::-1] / 1e6, color=ACCENT, alpha=0.85, edgecolor="#1e1e2e")
    ax1.set_xlabel("Total Prescriptions (millions)")
    ax1.set_title("Top 20 States by Rx Volume", fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)
    ax1.set_axisbelow(True)

    # Lorenz curve (inequality)
    sorted_vals = np.sort(state_vol.values)
    cumulative = np.cumsum(sorted_vals) / sorted_vals.sum()
    x = np.arange(1, len(cumulative) + 1) / len(cumulative)

    ax2.plot(x, cumulative, color=ACCENT, linewidth=2, label="Actual")
    ax2.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1, label="Perfect equality")
    ax2.fill_between(x, cumulative, x, alpha=0.15, color=ACCENT)

    # Gini coefficient
    n = len(sorted_vals)
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_vals) / (n * sorted_vals.sum())) - (n + 1) / n
    ax2.set_xlabel("Cumulative Share of States")
    ax2.set_ylabel("Cumulative Share of Prescriptions")
    ax2.set_title(f"Lorenz Curve (Gini = {gini:.3f})", fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.3)
    ax2.grid(alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda3_geographic_concentration.png")
    plt.close()
    logger.info("Saved eda3_geographic_concentration.png")


def eda4_demand_volatility(fact):
    """Coefficient of variation per drug — which drugs are stable vs volatile."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # National aggregation
    national = fact.groupby(["date", "product_name"])["number_of_prescriptions"].sum().reset_index()

    # Only drugs with 8+ quarters of non-zero data
    drug_stats = national[national["number_of_prescriptions"] > 0].groupby("product_name").agg(
        mean_rx=("number_of_prescriptions", "mean"),
        std_rx=("number_of_prescriptions", "std"),
        n_quarters=("number_of_prescriptions", "count"),
    ).reset_index()

    drug_stats = drug_stats[drug_stats["n_quarters"] >= 8].copy()
    drug_stats["cv"] = drug_stats["std_rx"] / drug_stats["mean_rx"]
    drug_stats = drug_stats.sort_values("cv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # CV distribution
    ax1.hist(drug_stats["cv"].clip(upper=2), bins=100, color=ACCENT, alpha=0.7, edgecolor="#1e1e2e")
    ax1.axvline(drug_stats["cv"].median(), color=ACCENT2, linestyle="--", linewidth=1.5,
                label=f"Median CV: {drug_stats['cv'].median():.2f}")
    ax1.set_xlabel("Coefficient of Variation")
    ax1.set_ylabel("Number of Drugs")
    ax1.set_title("Demand Volatility Distribution", fontweight="bold")
    ax1.legend(fontsize=9, framealpha=0.3)
    ax1.grid(alpha=0.3)

    # Most stable vs most volatile
    most_stable = drug_stats.head(10)
    most_volatile = drug_stats.tail(10)

    combined = pd.concat([
        most_stable.assign(category="Most Stable"),
        most_volatile.assign(category="Most Volatile"),
    ])
    colors = [ACCENT if c == "Most Stable" else ACCENT2 for c in combined["category"]]
    labels = [f"{row['product_name']}" for _, row in combined.iterrows()]

    ax2.barh(range(len(combined)), combined["cv"].values, color=colors, alpha=0.85, edgecolor="#1e1e2e")
    ax2.set_yticks(range(len(combined)))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Coefficient of Variation")
    ax2.set_title("Most Stable vs Most Volatile Drugs", fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    ax2.set_axisbelow(True)

    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(facecolor=ACCENT, label="Stable"), Patch(facecolor=ACCENT2, label="Volatile")],
               fontsize=9, framealpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda4_demand_volatility.png")
    plt.close()
    logger.info("Saved eda4_demand_volatility.png")


def eda5_seasonality_strength(fact):
    """Identify drugs with strongest seasonal patterns."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fact = fact.copy()
    fact["quarter"] = fact["date"].dt.quarter

    national = fact.groupby(["date", "quarter", "product_name"])["number_of_prescriptions"].sum().reset_index()

    # For each drug, compute the ratio of Q1 (flu season) to Q3 (summer)
    q1 = national[national["quarter"] == 1].groupby("product_name")["number_of_prescriptions"].mean()
    q3 = national[national["quarter"] == 3].groupby("product_name")["number_of_prescriptions"].mean()

    seasonal_ratio = (q1 / q3).dropna()
    seasonal_ratio = seasonal_ratio[seasonal_ratio > 0]
    seasonal_ratio = seasonal_ratio[q1.reindex(seasonal_ratio.index) > 1000]  # Filter to meaningful drugs

    most_seasonal = seasonal_ratio.sort_values(ascending=False).head(15)
    most_antiseasonal = seasonal_ratio.sort_values(ascending=True).head(15)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.barh(most_seasonal.index[::-1], most_seasonal.values[::-1], color=ACCENT, alpha=0.85, edgecolor="#1e1e2e")
    ax1.axvline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax1.set_xlabel("Q1/Q3 Demand Ratio")
    ax1.set_title("Most Seasonal (Winter Peak)", fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    ax2.barh(most_antiseasonal.index[::-1], most_antiseasonal.values[::-1], color=ACCENT2, alpha=0.85, edgecolor="#1e1e2e")
    ax2.axvline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax2.set_xlabel("Q1/Q3 Demand Ratio")
    ax2.set_title("Most Anti-Seasonal (Summer Peak)", fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    for ax in [ax1, ax2]:
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda5_seasonality.png")
    plt.close()
    logger.info("Saved eda5_seasonality.png")


def eda6_suppression_analysis(fact):
    """Suppression rates by state — data quality indicator."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if "is_suppressed" not in fact.columns:
        logger.warning("No is_suppressed column — skipping")
        return

    supp_rate = fact.groupby("state").agg(
        total=("is_suppressed", "count"),
        suppressed=("is_suppressed", "sum"),
    ).reset_index()
    supp_rate["rate"] = supp_rate["suppressed"] / supp_rate["total"] * 100
    supp_rate = supp_rate.sort_values("rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 12))
    colors = [ACCENT2 if r > 50 else ACCENT if r < 20 else MUTED for r in supp_rate["rate"]]
    ax.barh(supp_rate["state"], supp_rate["rate"], color=colors, alpha=0.85, edgecolor="#1e1e2e")
    ax.set_xlabel("Suppression Rate (%)")
    ax.set_title("Data Suppression Rate by State", fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda6_suppression.png")
    plt.close()
    logger.info("Saved eda6_suppression.png")


def eda7_shortage_impact(fact):
    """Before/during demand for drugs with active shortages."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    supply_path = Path("data/processed/feat_supply.parquet")
    if not supply_path.exists():
        logger.warning("No feat_supply — skipping shortage impact")
        return

    supply = pd.read_parquet(supply_path)
    shortage_drugs = supply[supply["shortage_active"] == 1]["ndc_11"].unique()

    if len(shortage_drugs) == 0:
        logger.warning("No shortage drugs found")
        return

    # National demand for shortage drugs
    shortage_demand = fact[fact["ndc_11"].isin(shortage_drugs)].copy()
    national = shortage_demand.groupby(["date", "product_name"])["number_of_prescriptions"].sum().reset_index()

    # Top 6 shortage drugs by volume
    top_shortage = (
        national.groupby("product_name")["number_of_prescriptions"]
        .sum().sort_values(ascending=False).head(6).index.tolist()
    )

    if not top_shortage:
        logger.warning("No top shortage drugs found")
        return

    colors = [ACCENT, ACCENT2, ACCENT3, "#f43f5e", "#22d3ee", "#facc15"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, drug in enumerate(top_shortage):
        ax = axes[i]
        drug_data = national[national["product_name"] == drug].sort_values("date")

        ax.plot(drug_data["date"], drug_data["number_of_prescriptions"] / 1e3,
                color=colors[i], linewidth=2, marker="o", markersize=4)
        ax.fill_between(drug_data["date"], 0,
                        drug_data["number_of_prescriptions"] / 1e3,
                        alpha=0.15, color=colors[i])

        # Mark shortage periods
        shortage_dates = supply[(supply["ndc_11"].isin(
            fact[fact["product_name"] == drug]["ndc_11"].unique()
        )) & (supply["shortage_active"] == 1)]["date"].unique()

        for sd in shortage_dates:
            ax.axvline(sd, color=ACCENT2, alpha=0.3, linewidth=1)

        ax.set_title(drug, fontsize=10, fontweight="bold")
        ax.set_ylabel("Rx (thousands)")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Demand for Drugs with Active Shortages (orange lines = shortage quarters)",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda7_shortage_impact.png")
    plt.close()
    logger.info("Saved eda7_shortage_impact.png")


def eda8_generic_competition(fact):
    """Demand trajectory for drugs with varying generic competition levels."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if "num_generic_competitors" not in fact.columns:
        logger.warning("No num_generic_competitors column — skipping")
        return

    # Bin by competition level
    fact = fact.copy()
    fact["competition_bin"] = pd.cut(
        fact["num_generic_competitors"],
        bins=[-1, 0, 3, 10, 50, 1000],
        labels=["None", "1-3", "4-10", "11-50", "50+"]
    )

    agg = fact.groupby(["date", "competition_bin"]).agg(
        mean_rx=("number_of_prescriptions", "mean"),
        total_rx=("number_of_prescriptions", "sum"),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"None": MUTED, "1-3": ACCENT, "4-10": ACCENT2, "11-50": ACCENT3, "50+": "#f43f5e"}

    for comp_bin in ["None", "1-3", "4-10", "11-50", "50+"]:
        data = agg[agg["competition_bin"] == comp_bin].sort_values("date")
        if len(data) > 0:
            ax1.plot(data["date"], data["mean_rx"], label=comp_bin,
                     color=colors[comp_bin], linewidth=2)
            ax2.plot(data["date"], data["total_rx"] / 1e6, label=comp_bin,
                     color=colors[comp_bin], linewidth=2)

    ax1.set_ylabel("Mean Rx per Drug-State-Quarter")
    ax1.set_title("Average Demand by Generic Competition Level", fontweight="bold")
    ax1.legend(title="Generic Competitors", fontsize=9, framealpha=0.3)
    ax1.grid(alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    ax2.set_ylabel("Total Rx (millions)")
    ax2.set_title("Total Volume by Competition Level", fontweight="bold")
    ax2.legend(title="Generic Competitors", fontsize=9, framealpha=0.3)
    ax2.grid(alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    for ax in [ax1, ax2]:
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda8_generic_competition.png")
    plt.close()
    logger.info("Saved eda8_generic_competition.png")


def eda9_predictability_ranking():
    """Rank drugs by model prediction accuracy — which are easiest/hardest to forecast."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = Path("data/results/baseline_results.json")
    if not results_path.exists():
        logger.warning("No results — need to run predictions first. Skipping.")
        return

    # We need per-drug errors. Check if prediction file exists
    # This requires regenerating predictions — skip if not available
    logger.info("eda9 requires per-drug predictions — generating from saved model...")

    try:
        from src.model.train_baseline import (
            load_and_prepare, filter_forecastable, split_train_val_test,
            fill_features, FEATURES_D, TARGET
        )
        import lightgbm as lgb

        model_path = Path("data/models/lgbm_config_d.txt")
        if not model_path.exists():
            model_path = Path("data/models/lgbm_config_b.txt")
        if not model_path.exists():
            logger.warning("No saved model found — skipping eda9")
            return

        df = load_and_prepare()
        df = filter_forecastable(df)
        train, val, test = split_train_val_test(df)

        available = [f for f in FEATURES_D if f in train.columns]
        fill_features(train, val, test, available)

        # Clean inf
        for dataset in [train, val, test]:
            for col in available:
                dataset.loc[:, col] = dataset[col].replace([np.inf, -np.inf], 0)

        test_clean = test[test[TARGET] > 0].copy()
        X_test = test_clean[available].values

        model = lgb.Booster(model_file=str(model_path))
        y_pred = np.expm1(model.predict(X_test))
        y_pred = np.maximum(y_pred, 0)

        test_clean["predicted"] = y_pred
        test_clean["abs_error"] = np.abs(test_clean[TARGET] - test_clean["predicted"])
        test_clean["pct_error"] = test_clean["abs_error"] / test_clean[TARGET] * 100

        # Per-drug accuracy
        drug_accuracy = test_clean.groupby("product_name").agg(
            mae=("abs_error", "mean"),
            median_ae=("abs_error", "median"),
            mape=("pct_error", "mean"),
            mean_rx=(TARGET, "mean"),
            n_obs=(TARGET, "count"),
        ).reset_index()

        drug_accuracy = drug_accuracy[drug_accuracy["n_obs"] >= 10]

        # Most and least predictable
        most_predictable = drug_accuracy.nsmallest(15, "mape")
        least_predictable = drug_accuracy.nlargest(15, "mape")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        ax1.barh(most_predictable["product_name"].values[::-1],
                 most_predictable["mape"].values[::-1],
                 color=ACCENT, alpha=0.85, edgecolor="#1e1e2e")
        ax1.set_xlabel("MAPE (%)")
        ax1.set_title("15 Most Predictable Drugs", fontweight="bold")
        ax1.grid(axis="x", alpha=0.3)

        ax2.barh(least_predictable["product_name"].values[::-1],
                 least_predictable["mape"].values[::-1].clip(max=200),
                 color=ACCENT2, alpha=0.85, edgecolor="#1e1e2e")
        ax2.set_xlabel("MAPE (%, clipped at 200)")
        ax2.set_title("15 Least Predictable Drugs", fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)

        for ax in [ax1, ax2]:
            ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "eda9_predictability_ranking.png")
        plt.close()
        logger.info("Saved eda9_predictability_ranking.png")

    except Exception as e:
        logger.error(f"eda9 failed: {e}")


def eda10_flu_drug_correlation(fact):
    """Correlation between ILI rates and flu-related drug demand."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    disease_path = Path("data/processed/feat_disease.parquet")
    if not disease_path.exists():
        logger.warning("No feat_disease — skipping")
        return

    disease = pd.read_parquet(disease_path)

    # Identify likely flu-related drugs by name
    flu_keywords = ["oseltam", "tamiflu", "amoxicil", "azithrom", "zithrom",
                    "augmentin", "levaquin", "cephalex", "doxycycl"]

    flu_drugs = fact[fact["product_name"].str.lower().str.contains("|".join(flu_keywords), na=False)]
    flu_national = flu_drugs.groupby("date")["number_of_prescriptions"].sum().reset_index()

    ili_national = disease.groupby("date")["ili_rate_mean"].mean().reset_index()

    merged = flu_national.merge(ili_national, on="date", how="inner").sort_values("date")

    if len(merged) < 4:
        logger.warning("Not enough data for flu correlation plot")
        return

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.set_xlabel("Quarter")
    ax1.set_ylabel("Flu-Related Rx (millions)", color=ACCENT)
    ax1.plot(merged["date"], merged["number_of_prescriptions"] / 1e6,
             color=ACCENT, linewidth=2, marker="o", markersize=5, label="Flu Drug Demand")
    ax1.tick_params(axis="y", labelcolor=ACCENT)
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean ILI Rate (%)", color=ACCENT2)
    ax2.plot(merged["date"], merged["ili_rate_mean"],
             color=ACCENT2, linewidth=2, marker="s", markersize=5, linestyle="--", label="ILI Rate")
    ax2.tick_params(axis="y", labelcolor=ACCENT2)

    # Correlation
    corr = merged["number_of_prescriptions"].corr(merged["ili_rate_mean"])
    ax1.set_title(f"Flu Drug Demand vs ILI Rate (r = {corr:.3f})", fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, framealpha=0.3, loc="upper left")

    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda10_flu_correlation.png")
    plt.close()
    logger.info("Saved eda10_flu_correlation.png")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    action = sys.argv[1] if len(sys.argv) > 1 else "all"

    fact = load_fact()

    if action == "all":
        eda1_target_distribution(fact)
        eda2_temporal_coverage(fact)
        eda3_geographic_concentration(fact)
        eda4_demand_volatility(fact)
        eda5_seasonality_strength(fact)
        eda6_suppression_analysis(fact)
        eda7_shortage_impact(fact)
        eda8_generic_competition(fact)
        eda10_flu_drug_correlation(fact)
    else:
        func = globals().get(action)
        if func:
            if "fact" in func.__code__.co_varnames:
                func(fact)
            else:
                func()
        else:
            print(f"Unknown analysis: {action}")

    logger.info(f"\nAll EDA figures saved to {OUTPUT_DIR}/")
