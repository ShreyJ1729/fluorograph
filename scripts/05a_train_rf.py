#!/usr/bin/env python3
"""
scripts/05a_train_rf.py

Train XGBoost on one-hot features with fixed hyperparams (no tuning).
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm


def load_onehot_split(name: str):
    """Load a onehot npz split. Returns X, em_max, ex_max."""
    d = np.load(f"features/onehot_{name}.npz", allow_pickle=True)
    X = d["X"].astype(np.float32)
    em_max = d["em_max"].astype(np.float64)
    ex_max = d["ex_max"].astype(np.float64)
    return X, em_max, ex_max


def compute_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(float(mean_squared_error(y_true, y_pred)))),
    }


N_ESTIMATORS = 500


def train_xgb_with_progress(X_train, y_train, target_name):
    """Train XGBoost with a tqdm progress bar over boosting rounds."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.3,
        "tree_method": "hist",
        "verbosity": 0,
    }

    model = None
    pbar = tqdm(total=N_ESTIMATORS, desc=f"XGB {target_name}", unit="round")
    for i in range(N_ESTIMATORS):
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1,
            xgb_model=model,
            verbose_eval=False,
        )
        pbar.update(1)
    pbar.close()
    return model


def main():
    print("=== 05a: XGBoost baseline (one-hot, no tuning) ===\n")

    X_train, em_train, ex_train = load_onehot_split("train")
    X_val, em_val, ex_val = load_onehot_split("val")
    X_test, em_test, ex_test = load_onehot_split("test")

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}\n")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    results = {}

    for target_name, y_train, y_val, y_test in [
        ("em_max", em_train, em_val, em_test),
        ("ex_max", ex_train, ex_val, ex_test),
    ]:
        model = train_xgb_with_progress(X_train, y_train, target_name)

        # Save model
        short = target_name.replace("_max", "")
        model_path = f"models/xgboost_{short}.json"
        model.save_model(model_path)
        print(f"Saved {model_path}")

        # Evaluate
        split_metrics = {}
        for split_name, X_s, y_s in [("val", X_val, y_val), ("test", X_test, y_test)]:
            dmat = xgb.DMatrix(X_s)
            pred = model.predict(dmat)
            split_metrics[split_name] = compute_metrics(y_s, pred)

        results[target_name] = split_metrics

    # Save results
    metrics_path = "results/metrics.json"
    all_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            all_metrics = json.load(f)
    all_metrics["xgb_baseline"] = results
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved {metrics_path}\n")

    # Print metrics
    print(f"{'Target':<10} {'Split':<8} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
    print("-" * 46)
    for target in ["em_max", "ex_max"]:
        for split in ["val", "test"]:
            m = results[target][split]
            print(f"{target:<10} {split:<8} {m['r2']:>8.4f} {m['mae']:>8.2f} {m['rmse']:>8.2f}")


if __name__ == "__main__":
    main()
