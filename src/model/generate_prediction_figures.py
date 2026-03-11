import pandas as pd
import numpy as np
import lightgbm as lgb
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
OUTPUT_DIR = Path("figures")

# Import the data loading from train_baseline
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.model.train_baseline import (
    load_and_prepare, filter_forecastable, split_train_val_test,
    fill_features, FEATURES_B, TARGET
)


def get_predictions():
    """Load data, train Config B, and return test predictions."""
    logger.info("Loading and preparing data...")
    df = load_and_prepare()
    df = filter_forecastable(df)
    train, val, test = split_train_val_test(df)

    available = [f for f in FEATURES_B if f in train.columns]
    fill_features(train, val, test, available)

    train_clean = train[train[TARGET] > 0].copy()
    val_clean = val[val[TARGET] > 0].copy()
    test_clean = test[test[TARGET] > 0].copy()

    X_train = train_clean[available].values
    y_train = np.log1p(train_clean[TARGET].values)
    X_val = val_clean[available].values
    y_val = np.log1p(val_clean[TARGET].values)
    X_test = test_clean[available].values
    y_test = test_clean[TARGET].values

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=available)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=available, reference=train_data)

    model = lgb.train(
        params, train_data,
        num_boost_round=3000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)],
    )

    y_pred = np.expm1(model.predict(X_test))
    y_pred = np.maximum(y_pred, 0)

    # Build results dataframe
    results_df = test_clean[["date", "state", "ndc_11", "product_name", "quarter_num", TARGET]].copy()
    results_df["predicted"] = y_pred
    results_df["residual"] = results_df[TARGET] - results_df["predicted"]
    results_df["abs_error"] = np.abs(results_df["residual"])
    results_df["pct_error"] = np.where(
        results_df[TARGET] > 0,
        results_df["abs_error"] / results_df[TARGET] * 100,
        0
    )

    # Add naive predictions (same quarter last year)
    history = pd.concat([train_clean, val_clean], ignore_index=True)
    results_df["lag_date"] = results_df["date"] - pd.DateOffset(years=1)
    naive_lookup = history[["ndc_11", "state", "date", TARGET]].rename(
        columns={"date": "lag_date", TARGET: "naive_pred"}
    )
    results_df = results_df.merge(naive_lookup, on=["ndc_11", "state", "lag_date"], how="left")

    # Fallback: series mean
    series_mean = history.groupby(["ndc_11", "state"])[TARGET].mean().reset_index().rename(
        columns={TARGET: "series_mean"}
    )
    results_df = results_df.merge(series_mean, on=["ndc_11", "state"], how="left")
    results_df["naive_pred"] = results_df["naive_pred"].fillna(results_df["series_mean"])
    results_df["naive_pred"] = results_df["naive_pred"].fillna(0)
    results_df = results_df.drop(columns=["lag_date", "series_mean"], errors="ignore")

    logger.info(f"Predictions generated: {len(results_df):,} rows")
    return results_df, model, available


def fig8_actual_vs_predicted(results_df):
    """Scatter plot: actual vs predicted prescriptions."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_actual = results_df[TARGET].values
    y_pred = results_df["predicted"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Log-log scatter (full range)
    sample_idx = np.random.RandomState(42).choice(len(y_actual), min(50000, len(y_actual)), replace=False)

    ax1.scatter(y_actual[sample_idx], y_pred[sample_idx], s=1, alpha=0.15, color=ACCENT)
    max_val = max(y_actual.max(), y_pred.max())
    ax1.plot([1, max_val], [1, max_val], "--", color=ACCENT2, linewidth=1.5, label="Perfect prediction")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Actual Prescriptions")
    ax1.set_ylabel("Predicted Prescriptions")
    ax1.set_title("Actual vs Predicted (log scale)", fontweight="bold")
    ax1.legend(fontsize=9, framealpha=0.3)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(1, max_val * 1.5)
    ax1.set_ylim(1, max_val * 1.5)

    # Linear scale (zoom into p99)
    p99 = np.percentile(y_actual, 99)
    mask = y_actual <= p99
    ax2.scatter(y_actual[sample_idx][mask[sample_idx]], y_pred[sample_idx][mask[sample_idx]],
                s=1, alpha=0.15, color=ACCENT)
    ax2.plot([0, p99], [0, p99], "--", color=ACCENT2, linewidth=1.5, label="Perfect prediction")
    ax2.set_xlabel("Actual Prescriptions")
    ax2.set_ylabel("Predicted Prescriptions")
    ax2.set_title(f"Actual vs Predicted (≤ p99 = {p99:,.0f})", fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.3)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig8_actual_vs_predicted.png")
    plt.close()
    logger.info("Saved fig8_actual_vs_predicted.png")


def fig9_top_drug_predictions(results_df):
    """Time series overlay: actual vs naive vs LightGBM for top drugs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Aggregate to national level per drug per quarter
    national = results_df.groupby(["date", "product_name"]).agg(
        actual=(TARGET, "sum"),
        predicted=("predicted", "sum"),
        naive=("naive_pred", "sum"),
    ).reset_index()

    # Top 6 drugs by total volume
    top_drugs = (
        national.groupby("product_name")["actual"]
        .sum()
        .sort_values(ascending=False)
        .head(6)
        .index.tolist()
    )

    colors = [ACCENT, ACCENT2, ACCENT3, "#f43f5e", "#22d3ee", "#facc15"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, drug in enumerate(top_drugs):
        ax = axes[i]
        drug_data = national[national["product_name"] == drug].sort_values("date")

        # Actual
        ax.plot(drug_data["date"], drug_data["actual"] / 1e3,
                color=colors[i], linewidth=2.5, marker="o", markersize=6, label="Actual", zorder=3)

        # LightGBM predicted
        ax.plot(drug_data["date"], drug_data["predicted"] / 1e3,
                color="#22d3ee", linewidth=2, marker="s", markersize=5,
                linestyle="--", label="LightGBM", zorder=2)

        # Naive predicted
        ax.plot(drug_data["date"], drug_data["naive"] / 1e3,
                color=MUTED, linewidth=1.5, marker="^", markersize=4,
                linestyle=":", label="Naive", zorder=1)

        ax.set_title(drug, fontsize=11, fontweight="bold")
        ax.set_ylabel("Rx (thousands)")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=7, framealpha=0.3, loc="best")

    plt.suptitle("Top 6 Drugs: Actual vs LightGBM vs Naive (National, Test Period)",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig9_top_drug_predictions.png")
    plt.close()
    logger.info("Saved fig9_top_drug_predictions.png")


def fig10_residual_distribution(results_df):
    """Residual distribution and bias analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    residuals = results_df["residual"].values
    pct_errors = results_df["pct_error"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Raw residual distribution (clipped for visibility)
    clipped = np.clip(residuals, -1000, 1000)
    ax1.hist(clipped, bins=100, color=ACCENT, alpha=0.7, edgecolor="#1e1e2e")
    ax1.axvline(0, color=ACCENT2, linestyle="--", linewidth=1.5, label="Zero error")
    ax1.axvline(np.median(residuals), color=ACCENT3, linestyle="--", linewidth=1.5,
                label=f"Median: {np.median(residuals):.1f}")
    ax1.set_xlabel("Residual (Actual - Predicted)")
    ax1.set_ylabel("Count")
    ax1.set_title("Residual Distribution (clipped ±1000)", fontweight="bold")
    ax1.legend(fontsize=9, framealpha=0.3)
    ax1.grid(alpha=0.3)

    # Percentage error distribution
    clipped_pct = np.clip(pct_errors, 0, 50)
    ax2.hist(clipped_pct, bins=100, color=ACCENT3, alpha=0.7, edgecolor="#1e1e2e")
    ax2.axvline(np.median(pct_errors), color=ACCENT2, linestyle="--", linewidth=1.5,
                label=f"Median: {np.median(pct_errors):.1f}%")
    ax2.set_xlabel("Absolute Percentage Error (%)")
    ax2.set_ylabel("Count")
    ax2.set_title("Percentage Error Distribution (clipped 0-50%)", fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.3)
    ax2.grid(alpha=0.3)

    for ax in [ax1, ax2]:
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig10_residual_distribution.png")
    plt.close()
    logger.info("Saved fig10_residual_distribution.png")


def fig11_error_by_state(results_df):
    """Map-style bar chart: MAE by state."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    state_errors = results_df.groupby("state").agg(
        mae=("abs_error", "mean"),
        median_ae=("abs_error", "median"),
        n_series=(TARGET, "count"),
        mean_rx=(TARGET, "mean"),
    ).reset_index()

    # Sort by MAE
    state_errors = state_errors.sort_values("mae", ascending=True)

    # Top and bottom 15 states
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # Lowest MAE (best performing)
    bottom15 = state_errors.head(15)
    ax1.barh(bottom15["state"], bottom15["mae"], color=ACCENT, alpha=0.85, edgecolor="#1e1e2e")
    ax1.set_xlabel("Mean Absolute Error (Rx)")
    ax1.set_title("15 Best-Performing States", fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)
    ax1.invert_yaxis()

    # Highest MAE (worst performing)
    top15 = state_errors.tail(15)
    ax2.barh(top15["state"], top15["mae"], color=ACCENT2, alpha=0.85, edgecolor="#1e1e2e")
    ax2.set_xlabel("Mean Absolute Error (Rx)")
    ax2.set_title("15 Worst-Performing States", fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    ax2.invert_yaxis()

    for ax in [ax1, ax2]:
        ax.set_axisbelow(True)

    plt.suptitle("Model Performance by State (Config B)", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig11_error_by_state.png")
    plt.close()
    logger.info("Saved fig11_error_by_state.png")


def fig12_learning_curve(model):
    """Training vs validation loss curve."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # LightGBM stores eval results
    evals = model.evals_result_
    if not evals:
        logger.warning("No eval results stored in model")
        return

    val_key = list(evals.keys())[0]
    metric_key = list(evals[val_key].keys())[0]
    val_loss = evals[val_key][metric_key]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(len(val_loss)), val_loss, color=ACCENT, linewidth=1.5, label="Validation MAE")

    # Mark best iteration
    best_iter = np.argmin(val_loss)
    ax.axvline(best_iter, color=ACCENT2, linestyle="--", linewidth=1,
               label=f"Best iteration: {best_iter}")
    ax.scatter([best_iter], [val_loss[best_iter]], color=ACCENT2, s=50, zorder=5)

    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("Validation MAE (log-space)")
    ax.set_title("Learning Curve — Config B", fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.3)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig12_learning_curve.png")
    plt.close()
    logger.info("Saved fig12_learning_curve.png")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Generating prediction figures...")
    results_df, model, features = get_predictions()

    fig8_actual_vs_predicted(results_df)
    fig9_top_drug_predictions(results_df)
    fig10_residual_distribution(results_df)
    fig11_error_by_state(results_df)
    fig12_learning_curve(model)

    logger.info(f"\nAll prediction figures saved to {OUTPUT_DIR}/")
