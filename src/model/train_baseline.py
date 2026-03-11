import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import sys


OUTPUT_DIR = Path("data/models")
RESULTS_DIR = Path("data/results")

# Proper 3-way temporal split
TRAIN_END = "2022-07-01"    # Train through 2022-Q3
VAL_START = "2022-10-01"    # Validate on 2022-Q4 + 2023-Q1
VAL_END = "2023-01-01"
TEST_START = "2023-04-01"   # Test on 2023-Q2, Q3, Q4

# Minimum history
MIN_QUARTERS = 8
MIN_MEAN_RX = 10

# Feature sets for ablation
FEATURES_A = [
    "rx_lag_1", "rx_lag_2", "rx_lag_4",
    "rx_rolling_mean_4", "rx_rolling_std_4",
    "rx_yoy_change", "rx_trend_4", "reimb_lag_1",
    "quarter_sin", "quarter_cos", "year_num",
]

FEATURES_B = FEATURES_A + [
    "num_generic_competitors", "num_patents",
    "months_to_patent_expiry", "is_near_patent_cliff",
    "shortage_active", "total_recalls", "class_i_recalls",
    "ili_rate_mean", "ili_rate_max", "ili_rate_std", "is_flu_season",
    "ili_rate_yoy_change",
    "adverse_event_count", "serious_event_count", "ae_spike",
]

TARGET = "number_of_prescriptions"


def mape(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    """Symmetric MAPE — handles scale better."""
    mask = (y_true > 0) | (y_pred > 0)
    if mask.sum() == 0:
        return np.nan
    denom = np.maximum((np.abs(y_true[mask]) + np.abs(y_pred[mask])) / 2, 1e-8)
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom) * 100


def evaluate(y_true, y_pred, label=""):
    """Compute all metrics including percentile-based tier analysis."""
    mask = y_true > 0
    y_t = y_true[mask]
    y_p = y_pred[mask]

    if len(y_t) == 0:
        logger.warning(f"  {label}: no non-zero targets to evaluate")
        return {}

    results = {
        "mae": mean_absolute_error(y_t, y_p),
        "rmse": np.sqrt(mean_squared_error(y_t, y_p)),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "median_ae": float(np.median(np.abs(y_t - y_p))),
        "n_test": int(len(y_t)),
    }

    # RMSE by volume tier
    percentiles = np.percentile(y_t, [50, 90, 99])
    tier_labels = ["low (≤p50)", "mid (p50-p90)", "high (p90-p99)", "extreme (>p99)"]
    tier_bounds = [0, percentiles[0], percentiles[1], percentiles[2], np.inf]

    for i in range(len(tier_labels)):
        tier_mask = (y_t >= tier_bounds[i]) & (y_t < tier_bounds[i + 1])
        n_tier = int(tier_mask.sum())
        if n_tier > 0:
            tier_mae = mean_absolute_error(y_t[tier_mask], y_p[tier_mask])
            tier_rmse = np.sqrt(mean_squared_error(y_t[tier_mask], y_p[tier_mask]))
            tier_mape_val = mape(y_t[tier_mask], y_p[tier_mask])
            results[f"mae_{tier_labels[i]}"] = tier_mae
            results[f"rmse_{tier_labels[i]}"] = tier_rmse
            results[f"mape_{tier_labels[i]}"] = tier_mape_val
            results[f"n_{tier_labels[i]}"] = n_tier

    if label:
        logger.info(f"  {label}")
        logger.info(f"    MAE: {results['mae']:.1f} | RMSE: {results['rmse']:.1f} | MAPE: {results['mape']:.1f}% | sMAPE: {results['smape']:.1f}% | Median AE: {results['median_ae']:.1f}")
        for tier in tier_labels:
            if f"mae_{tier}" in results:
                logger.info(
                    f"    {tier:20s}: MAE={results[f'mae_{tier}']:>8.1f}  "
                    f"RMSE={results[f'rmse_{tier}']:>10.1f}  "
                    f"MAPE={results.get(f'mape_{tier}', 0):>6.1f}%  "
                    f"n={results[f'n_{tier}']:>8,}"
                )

    return results


def load_and_prepare():
    logger.info("Loading data...")

    df = pd.read_parquet("data/processed/fact_demand_features.parquet")
    logger.info(f"  fact_demand_features: {len(df):,} rows")

    # Join feat_supply
    supply_path = Path("data/processed/feat_supply.parquet")
    if supply_path.exists():
        supply = pd.read_parquet(supply_path)
        supply = supply.drop_duplicates(subset=["date", "ndc_11"], keep="first")
        overlap_cols = [c for c in supply.columns if c in df.columns and c not in ["date", "ndc_11"]]
        if overlap_cols:
            logger.info(f"  Dropping overlapping columns from supply: {overlap_cols}")
            supply = supply.drop(columns=overlap_cols)
        df = df.merge(supply, on=["date", "ndc_11"], how="left")
        logger.info(f"  Joined feat_supply")

    # Join feat_disease
    disease_path = Path("data/processed/feat_disease.parquet")
    if disease_path.exists():
        disease = pd.read_parquet(disease_path)
        disease = disease.drop_duplicates(subset=["date", "state"], keep="first")
        df = df.merge(disease, on=["date", "state"], how="left")
        logger.info(f"  Joined feat_disease")

    # Join feat_safety (fix #5: safe string handling)
    safety_path = Path("data/processed/feat_safety.parquet")
    if safety_path.exists():
        safety = pd.read_parquet(safety_path)
        df["product_name_lower"] = df["product_name"].astype(str).str.strip().str.lower()
        safety["drug_name_lower"] = safety["drug_name_lower"].astype(str).str.strip().str.lower()
        safety = safety.drop_duplicates(subset=["date", "drug_name_lower"], keep="first")
        df = df.merge(safety, left_on=["date", "product_name_lower"],
                      right_on=["date", "drug_name_lower"], how="left")
        df = df.drop(columns=["product_name_lower", "drug_name_lower"], errors="ignore")
        logger.info(f"  Joined feat_safety")

    logger.info(f"  Final shape: {df.shape}")
    return df


def filter_forecastable(df):
    logger.info("Filtering to forecastable series...")

    series_stats = df[df[TARGET] > 0].groupby(["ndc_11", "state"]).agg(
        non_zero_quarters=(TARGET, "count"),
        mean_rx=(TARGET, "mean"),
    ).reset_index()

    good_series = series_stats[
        (series_stats["non_zero_quarters"] >= MIN_QUARTERS) &
        (series_stats["mean_rx"] >= MIN_MEAN_RX)
    ]

    logger.info(f"  Total series: {len(series_stats):,}")
    logger.info(f"  Forecastable: {len(good_series):,}")

    df = df.merge(good_series[["ndc_11", "state"]], on=["ndc_11", "state"], how="inner")
    logger.info(f"  Filtered rows: {len(df):,}")
    return df


def split_train_val_test(df):
    """Proper 3-way temporal split."""
    train = df[df["date"] <= TRAIN_END].copy()
    val = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].copy()
    test = df[df["date"] >= TEST_START].copy()

    logger.info(f"  Train: {len(train):,} rows ({train['date'].min()} to {train['date'].max()})")
    logger.info(f"  Val:   {len(val):,} rows ({val['date'].min()} to {val['date'].max()})")
    logger.info(f"  Test:  {len(test):,} rows ({test['date'].min()} to {test['date'].max()})")

    return train, val, test

def seasonal_naive(train, val, test):
    """Baseline: predict same quarter last year using exact date lag."""
    logger.info("Running Seasonal Naive baseline...")
    test = test.copy()

    # Use train + val for the naive lookup (all data before test)
    history = pd.concat([train, val], ignore_index=True)
    history_nonzero = history[history[TARGET] > 0]

    # Exact lag-4 match: for each test date, look up the value exactly 1 year prior
    test["lag_date"] = test["date"] - pd.DateOffset(years=1)
    lookup = history_nonzero[["ndc_11", "state", "date", TARGET]].rename(
        columns={"date": "lag_date", TARGET: "naive_pred"}
    )
    test = test.merge(lookup, on=["ndc_11", "state", "lag_date"], how="left")

    # Fallback: series mean if no same-quarter-last-year value
    series_mean = (
        history_nonzero
        .groupby(["ndc_11", "state"])[TARGET]
        .mean()
        .reset_index()
        .rename(columns={TARGET: "series_mean"})
    )
    test = test.merge(series_mean, on=["ndc_11", "state"], how="left")
    test["naive_pred"] = test["naive_pred"].fillna(test["series_mean"])
    test["naive_pred"] = test["naive_pred"].fillna(0)

    # Evaluate
    results = evaluate(test[TARGET].values, test["naive_pred"].values, "Seasonal Naive")
    results["model"] = "Seasonal Naive"
    results["config"] = "baseline"

    return results



def fill_features(train, val, test, available):
    """Fill NaN features consistently across all splits (fix #4: explicit copies)."""
    rate_features = {"ili_rate_mean", "ili_rate_max", "ili_rate_std", "ili_rate_yoy_change",
                     "rx_yoy_change", "rx_trend_4", "months_to_patent_expiry", "ae_qoq_change"}

    # Compute medians from training set only
    medians = {}
    for col in available:
        if col in rate_features:
            med = train[col].median()
            medians[col] = med if pd.notna(med) else 0
        else:
            medians[col] = 0

    # Apply to all splits
    for dataset in [train, val, test]:
        for col in available:
            dataset.loc[:, col] = dataset[col].fillna(medians[col])


def train_lightgbm(train, val, test, features, config_name, use_log_target=True):
    logger.info(f"Training LightGBM — Config {config_name} (log_target={use_log_target})...")

    available = [f for f in features if f in train.columns]
    missing = [f for f in features if f not in train.columns]
    if missing:
        logger.warning(f"  Missing features (skipped): {missing}")
    logger.info(f"  Using {len(available)} features")

    # Fill NaNs consistently
    fill_features(train, val, test, available)

    # Consistent zero-exclusion
    train_clean = train[train[TARGET] > 0].copy()
    val_clean = val[val[TARGET] > 0].copy()
    test_clean = test[test[TARGET] > 0].copy()

    logger.info(f"  Train: {len(train_clean):,} | Val: {len(val_clean):,} | Test: {len(test_clean):,}")

    if len(train_clean) < 100:
        logger.error("  Not enough training data!")
        return None, None

    X_train = train_clean[available].values
    y_train = train_clean[TARGET].values
    X_val = val_clean[available].values
    y_val = val_clean[TARGET].values
    X_test = test_clean[available].values
    y_test = test_clean[TARGET].values

    # Log-transform target to handle heavy tail
    if use_log_target:
        y_train_model = np.log1p(y_train)
        y_val_model = np.log1p(y_val)
    else:
        y_train_model = y_train
        y_val_model = y_val

    # Reverted parameters — closer to the earlier working config
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

    train_data = lgb.Dataset(X_train, label=y_train_model, feature_name=available)
    val_data = lgb.Dataset(X_val, label=y_val_model, feature_name=available, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=3000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    # Predict on TEST set
    y_pred = model.predict(X_test)

    # Inverse log transform
    if use_log_target:
        y_pred = np.expm1(y_pred)

    y_pred = np.maximum(y_pred, 0)

    # Diagnostic: check prediction distribution
    logger.info(f"  Prediction diagnostics:")
    logger.info(f"    y_test  — mean: {y_test.mean():.1f}, median: {np.median(y_test):.1f}, max: {y_test.max():.1f}")
    logger.info(f"    y_pred  — mean: {y_pred.mean():.1f}, median: {np.median(y_pred):.1f}, max: {y_pred.max():.1f}")

    # Evaluate
    results = evaluate(y_test, y_pred, f"LightGBM Config {config_name}")
    results["model"] = "LightGBM"
    results["config"] = config_name
    results["num_features"] = len(available)
    results["best_iteration"] = model.best_iteration
    results["log_target"] = use_log_target

    # Feature importance
    importance = dict(zip(available, model.feature_importance(importance_type="gain")))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    results["feature_importance"] = importance

    logger.info(f"  Best iteration: {model.best_iteration}")
    logger.info(f"  Top 10 features by gain:")
    for feat, gain in list(importance.items())[:10]:
        logger.info(f"    {feat:30s} {gain:>12,.0f}")

    model_path = OUTPUT_DIR / f"lgbm_config_{config_name.lower()}.txt"
    model.save_model(str(model_path))
    logger.info(f"  Model saved: {model_path}")

    return results, model



def run_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare()
    df = filter_forecastable(df)

    train, val, test = split_train_val_test(df)

    all_results = []

    # 1. Seasonal Naive (uses train+val for lookup)
    naive_results = seasonal_naive(train, val, test)
    all_results.append(naive_results)

    # 2. LightGBM Config A
    lgbm_a_results, model_a = train_lightgbm(train, val, test, FEATURES_A, "A")
    if lgbm_a_results:
        all_results.append(lgbm_a_results)

    # 3. LightGBM Config B
    lgbm_b_results, model_b = train_lightgbm(train, val, test, FEATURES_B, "B")
    if lgbm_b_results:
        all_results.append(lgbm_b_results)

    # ── Results Summary ──
    logger.info("\n" + "=" * 80)
    logger.info("ABLATION RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"  {'Model':<20s} {'Config':<10s} | {'MAE':>8s} | {'RMSE':>8s} | {'MAPE':>6s} | {'sMAPE':>6s} | {'MedAE':>8s} | {'N':>8s}")
    logger.info(f"  {'-'*80}")

    for r in all_results:
        logger.info(
            f"  {r['model']:<20s} {r['config']:<10s} | "
            f"{r['mae']:>8.1f} | {r['rmse']:>8.1f} | "
            f"{r.get('mape', 0):>5.1f}% | {r.get('smape', 0):>5.1f}% | "
            f"{r.get('median_ae', 0):>8.1f} | {r.get('n_test', 0):>8,}"
        )

    if lgbm_a_results and lgbm_b_results:
        mae_impr = (lgbm_a_results["mae"] - lgbm_b_results["mae"]) / lgbm_a_results["mae"] * 100
        rmse_impr = (lgbm_a_results["rmse"] - lgbm_b_results["rmse"]) / lgbm_a_results["rmse"] * 100
        logger.info(f"\n  Config B vs A: MAE {mae_impr:+.1f}% | RMSE {rmse_impr:+.1f}%")

    # Save results
    results_path = RESULTS_DIR / "baseline_results.json"
    clean_results = []
    for r in all_results:
        clean = {k: v for k, v in r.items() if k != "feature_importance"}
        if "feature_importance" in r:
            clean["feature_importance_top10"] = dict(list(r["feature_importance"].items())[:10])
        clean_results.append(clean)

    with open(results_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    logger.info(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    run_all()
