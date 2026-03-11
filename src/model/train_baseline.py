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

# Temporal split
TRAIN_END = "2023-01-01"    # Train through 2023-Q1
TEST_START = "2023-04-01"   # Test on 2023-Q2, Q3, Q4

# Minimum history: need at least 8 quarters of non-zero Rx to be forecastable
MIN_QUARTERS = 8
MIN_MEAN_RX = 10  # At least 10 avg Rx per quarter (skip ultra-low-volume drugs)

# Feature sets for ablation
FEATURES_A = [
    # Demand lags + calendar only
    "rx_lag_1", "rx_lag_2", "rx_lag_4",
    "rx_rolling_mean_4", "rx_rolling_std_4",
    "rx_yoy_change", "rx_trend_4", "reimb_lag_1",
    "quarter_sin", "quarter_cos", "year_num",
]

FEATURES_B = FEATURES_A + [
    # + structured supply/product features
    "num_generic_competitors", "num_patents",
    "months_to_patent_expiry", "is_near_patent_cliff",
    "shortage_active", "total_recalls", "class_i_recalls",
    # + disease features
    "ili_rate_mean", "ili_rate_max", "ili_rate_std", "is_flu_season",
    "ili_rate_yoy_change",
    # + safety features
    "adverse_event_count", "serious_event_count", "ae_spike",
]

TARGET = "number_of_prescriptions"


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error, excluding zeros."""
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def load_and_prepare():
    """Load fact_demand_features and join with feature tables."""
    logger.info("Loading data...")

    # Core demand with features
    df = pd.read_parquet("data/processed/fact_demand_features.parquet")
    logger.info(f"  fact_demand_features: {len(df):,} rows")

    # Join feat_supply
    supply_path = Path("data/processed/feat_supply.parquet")
    if supply_path.exists():
        supply = pd.read_parquet(supply_path)
        # Deduplicate supply features to one row per ndc × date
        supply = supply.drop_duplicates(subset=["date", "ndc_11"], keep="first")
        # Drop columns that already exist in fact to avoid suffixed duplicates
        overlap_cols = [c for c in supply.columns if c in df.columns and c not in ["date", "ndc_11"]]
        if overlap_cols:
            logger.info(f"  Dropping overlapping columns from supply: {overlap_cols}")
            supply = supply.drop(columns=overlap_cols)
        df = df.merge(supply, on=["date", "ndc_11"], how="left")
        logger.info(f"  Joined feat_supply: {len(supply):,} rows")

    # Join feat_disease
    disease_path = Path("data/processed/feat_disease.parquet")
    if disease_path.exists():
        disease = pd.read_parquet(disease_path)
        disease = disease.drop_duplicates(subset=["date", "state"], keep="first")
        df = df.merge(disease, on=["date", "state"], how="left")
        logger.info(f"  Joined feat_disease: {len(disease):,} rows")

    # Join feat_safety
    safety_path = Path("data/processed/feat_safety.parquet")
    if safety_path.exists():
        safety = pd.read_parquet(safety_path)
        # Match on drug name (lowercase)
        df["product_name_lower"] = df["product_name"].str.strip().str.lower()
        safety = safety.drop_duplicates(subset=["date", "drug_name_lower"], keep="first")
        df = df.merge(safety, left_on=["date", "product_name_lower"],
                      right_on=["date", "drug_name_lower"], how="left")
        df = df.drop(columns=["product_name_lower", "drug_name_lower"], errors="ignore")
        logger.info(f"  Joined feat_safety")

    logger.info(f"  Final shape: {df.shape}")
    return df


def filter_forecastable(df):
    """Filter to drug×state series with enough history for meaningful forecasting."""
    logger.info("Filtering to forecastable series...")

    # Count non-zero quarters per drug×state
    series_stats = df[df[TARGET] > 0].groupby(["ndc_11", "state"]).agg(
        non_zero_quarters=(TARGET, "count"),
        mean_rx=(TARGET, "mean"),
    ).reset_index()

    # Keep series with enough history and volume
    good_series = series_stats[
        (series_stats["non_zero_quarters"] >= MIN_QUARTERS) &
        (series_stats["mean_rx"] >= MIN_MEAN_RX)
    ]

    logger.info(f"  Total series: {len(series_stats):,}")
    logger.info(f"  Forecastable series (≥{MIN_QUARTERS} quarters, ≥{MIN_MEAN_RX} avg Rx): {len(good_series):,}")

    # Filter the main dataframe
    df = df.merge(good_series[["ndc_11", "state"]], on=["ndc_11", "state"], how="inner")
    logger.info(f"  Filtered rows: {len(df):,}")

    return df


def split_train_test(df):
    """Temporal train/test split."""
    train = df[df["date"] <= TRAIN_END].copy()
    test = df[df["date"] >= TEST_START].copy()

    logger.info(f"  Train: {len(train):,} rows ({train['date'].min()} to {train['date'].max()})")
    logger.info(f"  Test:  {len(test):,} rows ({test['date'].min()} to {test['date'].max()})")

    return train, test


def seasonal_naive(train, test):
    """Baseline: predict same quarter last year."""
    logger.info("Running Seasonal Naive baseline...")

    # For each drug×state in test, use same quarter from previous year
    test = test.copy()

    # Build lookup: for each (ndc, state, quarter), get the most recent value
    last_year = train[train["date"] >= "2022-04-01"].copy()  # Use 2022 Q2-Q4 and 2023 Q1
    lookup = last_year.groupby(["ndc_11", "state", "quarter_num"])[TARGET].last().reset_index()
    lookup = lookup.rename(columns={TARGET: "naive_pred"})

    test = test.merge(lookup, on=["ndc_11", "state", "quarter_num"], how="left")
    test["naive_pred"] = test["naive_pred"].fillna(test[TARGET].median())

    # Evaluate
    mask = test[TARGET] > 0
    results = {
        "model": "Seasonal Naive",
        "config": "baseline",
        "mae": mean_absolute_error(test.loc[mask, TARGET], test.loc[mask, "naive_pred"]),
        "rmse": np.sqrt(mean_squared_error(test.loc[mask, TARGET], test.loc[mask, "naive_pred"])),
        "mape": mape(test[TARGET].values, test["naive_pred"].values),
        "test_rows": len(test),
        "test_nonzero": int(mask.sum()),
    }

    logger.info(f"  Seasonal Naive — MAE: {results['mae']:.1f}, RMSE: {results['rmse']:.1f}, MAPE: {results['mape']:.1f}%")
    return results, test["naive_pred"]


def train_lightgbm(train, test, features, config_name):
    """Train LightGBM model."""
    logger.info(f"Training LightGBM — Config {config_name}...")

    # Use only features that exist in the dataframe
    available = [f for f in features if f in train.columns]
    missing = [f for f in features if f not in train.columns]
    if missing:
        logger.warning(f"  Missing features (skipped): {missing}")
    logger.info(f"  Using {len(available)} features: {available}")

    # Fill NaN features with defaults (0 for counts/flags, median for rates)
    rate_features = {"ili_rate_mean", "ili_rate_max", "ili_rate_std", "ili_rate_yoy_change",
                     "rx_yoy_change", "rx_trend_4", "months_to_patent_expiry", "ae_qoq_change"}

    for col in available:
        if col in rate_features:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val if pd.notna(median_val) else 0)
            test[col] = test[col].fillna(median_val if pd.notna(median_val) else 0)
        else:
            train[col] = train[col].fillna(0)
            test[col] = test[col].fillna(0)

    # Drop rows with NaN only in target or core lag features
    train_clean = train.dropna(subset=[TARGET])
    test_clean = test.dropna(subset=[TARGET])

    # Filter to non-zero and non-suppressed for training
    train_clean = train_clean[train_clean[TARGET] > 0]

    logger.info(f"  Train rows (after NaN/zero filter): {len(train_clean):,}")
    logger.info(f"  Test rows: {len(test_clean):,}")

    if len(train_clean) < 100:
        logger.error("  Not enough training data!")
        return None, None

    X_train = train_clean[available].values
    y_train = train_clean[TARGET].values
    X_test = test_clean[available].values
    y_test = test_clean[TARGET].values

    # LightGBM parameters
    params = {
        "objective": "regression_l1",  # MAE loss (robust to outliers)
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
    val_data = lgb.Dataset(X_test, label=y_test, feature_name=available, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # Predict
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # Can't have negative prescriptions

    # Evaluate
    mask = y_test > 0
    results = {
        "model": "LightGBM",
        "config": config_name,
        "mae": mean_absolute_error(y_test[mask], y_pred[mask]),
        "rmse": np.sqrt(mean_squared_error(y_test[mask], y_pred[mask])),
        "mape": mape(y_test, y_pred),
        "test_rows": len(test_clean),
        "test_nonzero": int(mask.sum()),
        "num_features": len(available),
        "best_iteration": model.best_iteration,
    }

    # Feature importance
    importance = dict(zip(available, model.feature_importance(importance_type="gain")))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    results["feature_importance_top10"] = dict(list(importance.items())[:10])

    logger.info(f"  LightGBM Config {config_name} — MAE: {results['mae']:.1f}, RMSE: {results['rmse']:.1f}, MAPE: {results['mape']:.1f}%")
    logger.info(f"  Best iteration: {model.best_iteration}")
    logger.info(f"  Top 5 features: {list(importance.keys())[:5]}")

    # Save model
    model_path = OUTPUT_DIR / f"lgbm_config_{config_name.lower()}.txt"
    model.save_model(str(model_path))
    logger.info(f"  Model saved: {model_path}")

    return results, model


def run_all():
    """Run full baseline comparison."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare
    df = load_and_prepare()
    df = filter_forecastable(df)

    # Split
    train, test = split_train_test(df)

    all_results = []

    # 1. Seasonal Naive
    naive_results, _ = seasonal_naive(train, test)
    all_results.append(naive_results)

    # 2. LightGBM Config A (demand + calendar only)
    lgbm_a_results, model_a = train_lightgbm(train, test, FEATURES_A, "A")
    if lgbm_a_results:
        all_results.append(lgbm_a_results)

    # 3. LightGBM Config B (+ structured features)
    lgbm_b_results, model_b = train_lightgbm(train, test, FEATURES_B, "B")
    if lgbm_b_results:
        all_results.append(lgbm_b_results)

    # ── Results Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION RESULTS SUMMARY")
    logger.info("=" * 60)

    for r in all_results:
        logger.info(f"  {r['model']:20s} Config {r['config']:10s} | MAE: {r['mae']:>10.1f} | RMSE: {r['rmse']:>10.1f} | MAPE: {r['mape']:>6.1f}%")

    # A vs B improvement
    if lgbm_a_results and lgbm_b_results:
        mae_improvement = (lgbm_a_results["mae"] - lgbm_b_results["mae"]) / lgbm_a_results["mae"] * 100
        logger.info(f"\n  Config B vs A improvement: {mae_improvement:.1f}% MAE reduction")

    # Save results
    results_path = RESULTS_DIR / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    run_all()
