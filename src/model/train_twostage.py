import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error

from src.model.train_baseline import (
    FEATURES_A,
    FEATURES_B,
    FEATURES_D,
    OUTPUT_DIR,
    RESULTS_DIR,
    TARGET,
    evaluate,
    fill_features,
    filter_forecastable,
    split_train_val_test,
)


# ============================================================
# Memory helpers
# ============================================================
def downcast_dataframe(df: pd.DataFrame, categorical_threshold: float = 0.05) -> pd.DataFrame:
    """
    Reduce dataframe memory usage by:
    - downcasting float64 -> float32
    - downcasting int64 -> smaller ints
    - converting low-cardinality object columns to category
    """
    for col in df.columns:
        col_data = df[col]

        if pd.api.types.is_float_dtype(col_data):
            df[col] = pd.to_numeric(col_data, downcast="float")
        elif pd.api.types.is_integer_dtype(col_data):
            df[col] = pd.to_numeric(col_data, downcast="integer")
        elif pd.api.types.is_object_dtype(col_data):
            nunique = col_data.nunique(dropna=False)
            if len(df) > 0 and (nunique / len(df)) <= categorical_threshold:
                try:
                    df[col] = col_data.astype("category")
                except Exception:
                    pass

    return df


def log_mem(df: pd.DataFrame, name: str):
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"  {name}: {len(df):,} rows | {df.shape[1]} cols | {mem_mb:,.1f} MB")


def dedupe_on_keys(df: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    """
    Ensure one row per merge key to prevent merge explosion.
    """
    if not all(k in df.columns for k in keys):
        missing = [k for k in keys if k not in df.columns]
        logger.warning(f"  {name}: cannot dedupe, missing keys {missing}")
        return df

    dup_count = df.duplicated(subset=keys).sum()
    if dup_count > 0:
        logger.warning(f"  {name}: found {dup_count:,} duplicate key rows on {keys}; keeping last")
        df = df.drop_duplicates(subset=keys, keep="last")

    return df


def safe_prepare_merge_frame(
    df_right: pd.DataFrame,
    keys: list[str],
    name: str,
    keep_only_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Prepare a right-side frame before merge:
    - keep selected columns if provided
    - dedupe on merge keys
    - downcast dtypes
    """
    if keep_only_cols is not None:
        keep_only_cols = [c for c in keep_only_cols if c in df_right.columns]
        df_right = df_right[keep_only_cols].copy()
    else:
        df_right = df_right.copy()

    df_right = dedupe_on_keys(df_right, keys, name)
    df_right = downcast_dataframe(df_right)
    log_mem(df_right, name)
    return df_right


def safe_left_merge(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    on: list[str],
    name: str,
) -> pd.DataFrame:
    """
    Memory-safer left merge:
    - drops right-side columns already present in left (except keys)
    - dedupes right-side keys
    - downcasts after merge
    """
    if df_right is None:
        logger.warning(f"  {name}: dataframe is None, skipping")
        return df_left

    missing_left = [k for k in on if k not in df_left.columns]
    missing_right = [k for k in on if k not in df_right.columns]

    if missing_left:
        logger.warning(f"  {name}: left side missing keys {missing_left}; skipping")
        return df_left
    if missing_right:
        logger.warning(f"  {name}: right side missing keys {missing_right}; skipping")
        return df_left

    df_right = dedupe_on_keys(df_right, on, name)

    overlapping = [c for c in df_right.columns if c in df_left.columns and c not in on]
    if overlapping:
        logger.info(f"  {name}: dropping overlapping columns from right: {overlapping}")
        df_right = df_right.drop(columns=overlapping)

    add_cols = [c for c in df_right.columns if c not in on]
    logger.info(f"  Merging {name} on {on} | adding {len(add_cols)} columns")

    df_merged = df_left.merge(df_right, on=on, how="left")
    df_merged = downcast_dataframe(df_merged)
    log_mem(df_merged, f"post-{name}")

    del df_right
    gc.collect()

    return df_merged


# ============================================================
# Data loading (memory-safe alternative to baseline loader)
# ============================================================
def load_and_prepare_twostage() -> pd.DataFrame:
    """
    Memory-safe loader for two-stage training.
    Rebuilds the training matrix without using the very memory-heavy
    baseline loader path that crashes during reg_drug merge.
    """
    logger.info("Loading data (memory-safe twostage loader)...")

    processed_dir = Path("data/processed")

    fact_path = processed_dir / "fact_demand_features.parquet"
    if not fact_path.exists():
        raise FileNotFoundError(f"Missing required file: {fact_path}")

    df = pd.read_parquet(fact_path)
    logger.info(f"  fact_demand_features: {len(df):,} rows")

    # Standardize date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Standardize product lower column if needed
    if "product_name_lower" not in df.columns and "product_name" in df.columns:
        df["product_name_lower"] = (
            df["product_name"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

    # Optional: treat state as category to save memory
    for cat_col in ["state", "ndc_11", "product_name_lower"]:
        if cat_col in df.columns:
            try:
                df[cat_col] = df[cat_col].astype("category")
            except Exception:
                pass

    df = downcast_dataframe(df)
    log_mem(df, "fact_demand_features")

    # --------------------------------------------------------
    # Supply features
    # Expected keys: ["date", "ndc_11"]
    # --------------------------------------------------------
    supply_path = processed_dir / "feat_supply.parquet"
    if supply_path.exists():
        supply = pd.read_parquet(supply_path)
        if "date" in supply.columns:
            supply["date"] = pd.to_datetime(supply["date"])

        supply = safe_prepare_merge_frame(
            supply,
            keys=["date", "ndc_11"],
            name="feat_supply",
        )

        df = safe_left_merge(df, supply, on=["date", "ndc_11"], name="feat_supply")
        del supply
        gc.collect()
    else:
        logger.warning("  feat_supply not found — skipping")

    # --------------------------------------------------------
    # Disease features
    # Expected keys: ["date", "state"]
    # --------------------------------------------------------
    disease_path = processed_dir / "feat_disease.parquet"
    if disease_path.exists():
        disease = pd.read_parquet(disease_path)
        if "date" in disease.columns:
            disease["date"] = pd.to_datetime(disease["date"])

        disease = safe_prepare_merge_frame(
            disease,
            keys=["date", "state"],
            name="feat_disease",
        )

        df = safe_left_merge(df, disease, on=["date", "state"], name="feat_disease")
        del disease
        gc.collect()
    else:
        logger.warning("  feat_disease not found — skipping")

    # --------------------------------------------------------
    # Safety features
    # Expected keys: ["date", "drug_name_lower"]
    # feat_safety uses "drug_name_lower", fact uses "product_name_lower"
    # --------------------------------------------------------
    safety_path = processed_dir / "feat_safety.parquet"
    if safety_path.exists():
        safety = pd.read_parquet(safety_path)
        if "date" in safety.columns:
            safety["date"] = pd.to_datetime(safety["date"])

        # Rename to match fact table's key
        if "drug_name_lower" in safety.columns and "product_name_lower" not in safety.columns:
            safety = safety.rename(columns={"drug_name_lower": "product_name_lower"})

        safety = safe_prepare_merge_frame(
            safety,
            keys=["date", "product_name_lower"],
            name="feat_safety",
        )

        df = safe_left_merge(
            df,
            safety,
            on=["date", "product_name_lower"],
            name="feat_safety",
        )
        del safety
        gc.collect()
    else:
        logger.warning("  feat_safety not found — skipping")

    # --------------------------------------------------------
    # Regulation features: market-level
    # Expected key: ["date"]
    # --------------------------------------------------------
    reg_market_path = processed_dir / "feat_regulation_market.parquet"
    if not reg_market_path.exists():
        # fallback if your project stores it under a generic name
        alt_market = processed_dir / "feat_regulation.parquet"
        reg_market_path = alt_market if alt_market.exists() else reg_market_path

    if reg_market_path.exists():
        reg_market = pd.read_parquet(reg_market_path)
        if "date" in reg_market.columns:
            reg_market["date"] = pd.to_datetime(reg_market["date"])

        reg_market = safe_prepare_merge_frame(
            reg_market,
            keys=["date"],
            name="feat_regulation_market",
        )

        df = safe_left_merge(df, reg_market, on=["date"], name="feat_regulation_market")
        del reg_market
        gc.collect()
    else:
        logger.warning("  feat_regulation_market not found — skipping")

    # --------------------------------------------------------
    # Regulation features: drug-level
    # Expected keys: ["date", "product_name_lower"]
    # This is the merge that previously crashed.
    # We make it safer by:
    # - deduping keys
    # - dropping overlapping columns
    # - downcasting both sides first
    # --------------------------------------------------------
    reg_drug_path = processed_dir / "feat_regulation_drug.parquet"
    if reg_drug_path.exists():
        reg_drug = pd.read_parquet(reg_drug_path)
        if "date" in reg_drug.columns:
            reg_drug["date"] = pd.to_datetime(reg_drug["date"])

        if "product_name_lower" not in reg_drug.columns and "product_name" in reg_drug.columns:
            reg_drug["product_name_lower"] = (
                reg_drug["product_name"]
                .astype(str)
                .str.lower()
                .str.strip()
            )

        # Keep only columns that are not absurdly wide if present.
        # This avoids pulling raw text or unused giant columns into the model matrix.
        likely_keep = [
            "date",
            "product_name_lower",
            "reg_drug_doc_count",
            "reg_drug_approval_count",
            "reg_drug_safety_count",
            "reg_drug_manufacturing_count",
        ]
        existing_keep = [c for c in likely_keep if c in reg_drug.columns]

        # If none of the curated names exist, still keep all columns,
        # but this path is less ideal.
        reg_drug = safe_prepare_merge_frame(
            reg_drug,
            keys=["date", "product_name_lower"],
            name="feat_regulation_drug",
            keep_only_cols=existing_keep if existing_keep else None,
        )

        df = safe_left_merge(
            df,
            reg_drug,
            on=["date", "product_name_lower"],
            name="feat_regulation_drug",
        )
        del reg_drug
        gc.collect()
    else:
        logger.warning("  feat_regulation_drug not found — skipping")

    logger.info("Finished loading and preparing data")
    log_mem(df, "final_model_matrix")
    return df


# ============================================================
# Training
# ============================================================
def train_twostage(train, val, test, features, config_name):
    """Two-stage model: (1) classify demand occurrence, (2) predict magnitude."""
    logger.info(f"Training Two-Stage LightGBM — Config {config_name}...")

    available = [f for f in features if f in train.columns]
    missing = [f for f in features if f not in train.columns]
    if missing:
        logger.warning(f"  Missing features (skipped): {missing}")
    logger.info(f"  Using {len(available)} features")

    fill_features(train, val, test, available)

    # Clean inf
    for dataset in [train, val, test]:
        for col in available:
            dataset.loc[:, col] = dataset[col].replace([np.inf, -np.inf], 0)

    # --------------------------------------------------------
    # STAGE 1: Binary classification
    # --------------------------------------------------------
    logger.info("  Stage 1: Demand occurrence classification...")

    train_s1 = train.dropna(subset=[TARGET]).copy()
    val_s1 = val.dropna(subset=[TARGET]).copy()
    test_s1 = test.dropna(subset=[TARGET]).copy()

    train_s1["has_demand"] = (train_s1[TARGET] > 0).astype(np.int8)
    val_s1["has_demand"] = (val_s1[TARGET] > 0).astype(np.int8)
    test_s1["has_demand"] = (test_s1[TARGET] > 0).astype(np.int8)

    logger.info(
        f"    Train: {len(train_s1):,} rows, {train_s1['has_demand'].mean() * 100:.1f}% positive"
    )
    logger.info(
        f"    Val:   {len(val_s1):,} rows, {val_s1['has_demand'].mean() * 100:.1f}% positive"
    )
    logger.info(
        f"    Test:  {len(test_s1):,} rows, {test_s1['has_demand'].mean() * 100:.1f}% positive"
    )

    n_neg = int((train_s1["has_demand"] == 0).sum())
    n_pos = int((train_s1["has_demand"] == 1).sum())
    scale_pos = n_neg / max(n_pos, 1)

    params_s1 = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale_pos,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    X_train_s1 = train_s1[available].to_numpy(dtype=np.float32, copy=False)
    X_val_s1 = val_s1[available].to_numpy(dtype=np.float32, copy=False)
    X_test_s1 = test_s1[available].to_numpy(dtype=np.float32, copy=False)

    y_train_s1 = train_s1["has_demand"].to_numpy(dtype=np.int8, copy=False)
    y_val_s1 = val_s1["has_demand"].to_numpy(dtype=np.int8, copy=False)

    ds1_train = lgb.Dataset(X_train_s1, label=y_train_s1, feature_name=available, free_raw_data=True)
    ds1_val = lgb.Dataset(X_val_s1, label=y_val_s1, feature_name=available, reference=ds1_train, free_raw_data=True)

    model_s1 = lgb.train(
        params_s1,
        ds1_train,
        num_boost_round=1000,
        valid_sets=[ds1_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
    )

    prob_demand = model_s1.predict(X_test_s1)
    val_probs = model_s1.predict(X_val_s1)

    best_threshold = 0.5
    best_f1 = -1.0
    val_actual = val_s1["has_demand"].to_numpy(dtype=np.int8, copy=False)

    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (val_probs >= threshold).astype(np.int8)
        tp = ((preds == 1) & (val_actual == 1)).sum()
        fp = ((preds == 1) & (val_actual == 0)).sum()
        fn = ((preds == 0) & (val_actual == 1)).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    logger.info(f"    Best threshold: {best_threshold:.2f} (F1: {best_f1:.3f})")

    pred_has_demand = (prob_demand >= best_threshold).astype(np.int8)

    actual_has_demand = test_s1["has_demand"].to_numpy(dtype=np.int8, copy=False)
    tp = int(((pred_has_demand == 1) & (actual_has_demand == 1)).sum())
    fp = int(((pred_has_demand == 1) & (actual_has_demand == 0)).sum())
    fn = int(((pred_has_demand == 0) & (actual_has_demand == 1)).sum())
    tn = int(((pred_has_demand == 0) & (actual_has_demand == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / len(actual_has_demand)

    logger.info(f"    Stage 1 Test: Acc={accuracy:.3f} Prec={precision:.3f} Rec={recall:.3f}")
    logger.info(f"    TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}")

    # --------------------------------------------------------
    # STAGE 2: Magnitude regression on positive rows only
    # --------------------------------------------------------
    logger.info("  Stage 2: Demand magnitude prediction...")

    train_s2 = train_s1[train_s1[TARGET] > 0].copy()
    val_s2 = val_s1[val_s1[TARGET] > 0].copy()

    logger.info(f"    Train S2: {len(train_s2):,} | Val S2: {len(val_s2):,}")

    X_train_s2 = train_s2[available].to_numpy(dtype=np.float32, copy=False)
    X_val_s2 = val_s2[available].to_numpy(dtype=np.float32, copy=False)

    y_train_s2 = np.log1p(train_s2[TARGET].to_numpy(dtype=np.float32, copy=False))
    y_val_s2 = np.log1p(val_s2[TARGET].to_numpy(dtype=np.float32, copy=False))

    params_s2 = {
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

    ds2_train = lgb.Dataset(X_train_s2, label=y_train_s2, feature_name=available, free_raw_data=True)
    ds2_val = lgb.Dataset(X_val_s2, label=y_val_s2, feature_name=available, reference=ds2_train, free_raw_data=True)

    model_s2 = lgb.train(
        params_s2,
        ds2_train,
        num_boost_round=3000,
        valid_sets=[ds2_val],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    # --------------------------------------------------------
    # Combine stages
    # --------------------------------------------------------
    logger.info("  Combining stages...")

    magnitude_pred = np.expm1(model_s2.predict(X_test_s1))
    magnitude_pred = np.maximum(magnitude_pred, 0)

    final_pred = np.where(pred_has_demand == 1, magnitude_pred, 0)
    ungated_pred = magnitude_pred

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    y_test_all = test_s1[TARGET].to_numpy(dtype=np.float32, copy=False)

    logger.info("  --- Full Test Set (including zeros) ---")
    mae_all = mean_absolute_error(y_test_all, final_pred)
    logger.info(f"    Two-Stage MAE (all rows): {mae_all:.1f}")

    nonzero_mask = y_test_all > 0
    y_test_nz = y_test_all[nonzero_mask]
    final_pred_nz = final_pred[nonzero_mask]
    ungated_pred_nz = ungated_pred[nonzero_mask]

    logger.info("  --- Non-Zero Test Set (comparable to single-stage) ---")
    results_gated = evaluate(y_test_nz, final_pred_nz, f"Two-Stage Gated Config {config_name}")
    _ = evaluate(y_test_nz, ungated_pred_nz, f"Two-Stage Ungated Config {config_name}")

    correctly_gated = int((pred_has_demand[nonzero_mask] == 1).sum())
    incorrectly_gated = int((pred_has_demand[nonzero_mask] == 0).sum())

    logger.info(
        f"    Non-zero rows correctly passed by gate: "
        f"{correctly_gated:,}/{len(y_test_nz):,} ({correctly_gated / len(y_test_nz) * 100:.1f}%)"
    )
    logger.info(f"    Non-zero rows incorrectly blocked by gate: {incorrectly_gated:,} (forced to 0)")

    zero_mask = y_test_all == 0
    false_positives_gated = final_pred[zero_mask].sum()
    false_positives_ungated = ungated_pred[zero_mask].sum()

    logger.info(f"    False positive Rx on zero rows (gated): {false_positives_gated:,.0f}")
    logger.info(f"    False positive Rx on zero rows (ungated): {false_positives_ungated:,.0f}")

    results_gated["model"] = "Two-Stage LightGBM"
    results_gated["config"] = config_name
    results_gated["num_features"] = len(available)
    results_gated["s1_best_iteration"] = model_s1.best_iteration
    results_gated["s2_best_iteration"] = model_s2.best_iteration
    results_gated["s1_threshold"] = best_threshold
    results_gated["s1_accuracy"] = accuracy
    results_gated["s1_precision"] = precision
    results_gated["s1_recall"] = recall
    results_gated["mae_all_rows"] = mae_all

    importance = dict(zip(available, model_s2.feature_importance(importance_type="gain")))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    results_gated["feature_importance"] = importance

    logger.info(f"  S2 Best iteration: {model_s2.best_iteration}")
    logger.info("  Top 10 features (Stage 2) by gain:")
    for feat, gain in list(importance.items())[:10]:
        logger.info(f"    {feat:30s} {gain:>12,.0f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_s1.save_model(str(OUTPUT_DIR / f"twostage_s1_config_{config_name.lower()}.txt"))
    model_s2.save_model(str(OUTPUT_DIR / f"twostage_s2_config_{config_name.lower()}.txt"))

    # Explicit cleanup
    del X_train_s1, X_val_s1, X_test_s1, X_train_s2, X_val_s2
    del ds1_train, ds1_val, ds2_train, ds2_val
    gc.collect()

    return results_gated


# ============================================================
# Runner
# ============================================================
def run_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_twostage()
    df = filter_forecastable(df)
    logger.info(f"After filter_forecastable: {len(df):,} rows")

    train, val, test = split_train_val_test(df)
    del df
    gc.collect()

    all_results = []

    ts_d_results = train_twostage(train, val, test, FEATURES_D, "D")
    if ts_d_results:
        all_results.append(ts_d_results)

    results_path = RESULTS_DIR / "twostage_results.json"
    clean_results = []
    for r in all_results:
        clean = {k: v for k, v in r.items() if k != "feature_importance"}
        if "feature_importance" in r:
            clean["feature_importance_top10"] = dict(list(r["feature_importance"].items())[:10])
        clean_results.append(clean)

    with open(results_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)

    logger.info(f"Results saved: {results_path}")


if __name__ == "__main__":
    run_all()