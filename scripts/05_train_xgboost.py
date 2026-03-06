#!/usr/bin/env python3
"""
scripts/05_train_xgboost.py

Train XGBoost models on one-hot or other feature sets.

--mode g0: FPredX replication (one-hot only, 100 Optuna trials)
--mode g1: Ablation across 6 feature combinations (US-007)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np
import numpy.typing as npt
import optuna
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]


def load_onehot_split(
    name: str,
) -> tuple[Float32Array, npt.NDArray[Any], FloatArray, FloatArray]:
    """Load a onehot npz split. Returns X, names, em_max, ex_max."""
    path = f"features/onehot_{name}.npz"
    d = np.load(path, allow_pickle=True)
    X: Float32Array = d["X"].astype(np.float32)
    names: npt.NDArray[Any] = d["names"]
    em_max: FloatArray = d["em_max"].astype(np.float64)
    ex_max: FloatArray = d["ex_max"].astype(np.float64)
    return X, names, em_max, ex_max


def _train_xgb_with_progress(
    params: dict[str, Any],
    dtrain: xgb.DMatrix,
    n_rounds: int,
    pbar_pos: int = 1,
) -> xgb.Booster:
    """Train XGBoost one round at a time with a tqdm bar."""
    pbar = tqdm(
        total=n_rounds,
        desc="    XGB rounds",
        unit="rnd",
        leave=False,
        position=pbar_pos,
    )
    model: xgb.Booster | None = None
    xgb_params = {
        k: v
        for k, v in params.items()
        if k not in ("n_estimators",)
    }
    xgb_params.setdefault("tree_method", "hist")
    xgb_params.setdefault("verbosity", 0)
    for _ in range(n_rounds):
        model = xgb.train(
            xgb_params, dtrain, num_boost_round=1,
            xgb_model=model, verbose_eval=False,
        )
        pbar.update(1)
    pbar.close()
    assert model is not None
    return model


def optimize_xgb(
    X_train: npt.NDArray[Any],
    y_train: FloatArray,
    n_trials: int = 100,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Run Optuna to find best XGBoost hyperparams by minimizing 5-fold CV MAE."""
    trial_pbar = tqdm(
        total=n_trials, desc="  Optuna trials", unit="trial", position=0
    )
    best_mae: float = float("inf")

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_mae
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        params: dict[str, Any] = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 0.5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "tree_method": "hist",
            "verbosity": 0,
        }
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        maes: list[float] = []
        for train_idx, val_idx in kf.split(X_train):
            Xtr, Xvl = X_train[train_idx], X_train[val_idx]
            ytr, yvl = y_train[train_idx], y_train[val_idx]
            dtrain_fold = xgb.DMatrix(Xtr, label=ytr)
            booster = _train_xgb_with_progress(params, dtrain_fold, n_estimators)
            dval = xgb.DMatrix(Xvl)
            pred = booster.predict(dval)
            maes.append(float(mean_absolute_error(yvl, pred)))
        trial_mae = float(np.mean(maes))
        best_mae = min(best_mae, trial_mae)
        trial_pbar.set_postfix(best_mae=f"{best_mae:.2f}")
        trial_pbar.update(1)
        return trial_mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    trial_pbar.close()
    best = dict(study.best_params)
    # Re-insert n_estimators into the params dict for downstream use
    return best


def train_final_model(
    X_train: npt.NDArray[Any],
    y_train: FloatArray,
    params: dict[str, Any],
) -> xgb.Booster:
    """Train final XGBoost model on full training set with given params."""
    n_estimators = int(params.get("n_estimators", 500))
    dtrain = xgb.DMatrix(X_train, label=y_train)
    print(f"  Training final model ({n_estimators} rounds)...")
    model = _train_xgb_with_progress(params, dtrain, n_estimators, pbar_pos=0)
    return model


def compute_metrics(
    y_true: FloatArray, y_pred: npt.NDArray[Any]
) -> dict[str, float]:
    """Compute R², MAE, RMSE."""
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(float(mean_squared_error(y_true, y_pred))))
    return {"r2": r2, "mae": mae, "rmse": rmse}


def run_g0() -> None:
    """G0: FPredX replication with one-hot + XGBoost."""
    print("=== G0: FPredX Replication — One-hot XGBoost ===\n")

    # Load splits
    X_train, _, em_train, ex_train = load_onehot_split("train")
    X_val, _, em_val, ex_val = load_onehot_split("val")
    X_test, _, em_test, ex_test = load_onehot_split("test")

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}\n")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    results: dict[str, Any] = {}
    best_params_all: dict[str, dict[str, Any]] = {}

    targets: list[tuple[str, FloatArray, FloatArray, FloatArray]] = [
        ("em_max", em_train, em_val, em_test),
        ("ex_max", ex_train, ex_val, ex_test),
    ]

    for target_name, y_train, y_val, y_test in targets:
        print(f"--- Optimizing {target_name} (100 Optuna trials, 5-fold CV MAE) ---")
        params = optimize_xgb(X_train, y_train, n_trials=100, n_folds=5)
        best_params_all[target_name] = params
        print(f"Best params for {target_name}: {params}\n")

        model = train_final_model(X_train, y_train, params)

        # Save model
        short = target_name.replace("_max", "")
        model_path = f"models/xgboost_{short}.json"
        model.save_model(model_path)
        print(f"Saved model to {model_path}")

        # Evaluate on val and test
        split_metrics: dict[str, dict[str, float]] = {}
        for split_name, X_s, y_s in [
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            pred: npt.NDArray[Any] = model.predict(xgb.DMatrix(X_s))
            split_metrics[split_name] = compute_metrics(y_s, pred)

        results[target_name] = split_metrics

    # Save best params
    with open("models/xgboost_params.json", "w") as f:
        json.dump(best_params_all, f, indent=2)
    print("\nSaved models/xgboost_params.json")

    # Load or init metrics.json
    metrics_path = "results/metrics.json"
    all_metrics: dict[str, Any] = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            all_metrics = json.load(f)
    all_metrics["g0"] = results
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("Saved results/metrics.json\n")

    # Print metrics table
    print("=== G0 Metrics ===")
    print(f"{'Target':<10} {'Split':<8} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
    print("-" * 46)
    for target in ["em_max", "ex_max"]:
        for split in ["val", "test"]:
            m = results[target][split]
            print(
                f"{target:<10} {split:<8} {m['r2']:>8.4f} {m['mae']:>8.2f} {m['rmse']:>8.2f}"
            )

    # G0 gate: R² within 0.05 of FPredX published (em ~0.88, ex ~0.84)
    em_test_r2: float = results["em_max"]["test"]["r2"]
    ex_test_r2: float = results["ex_max"]["test"]["r2"]
    em_ref, ex_ref = 0.88, 0.84
    em_pass = abs(em_test_r2 - em_ref) <= 0.05
    ex_pass = abs(ex_test_r2 - ex_ref) <= 0.05

    print("\n=== G0 Gate ===")
    print(
        f"em_max test R²: {em_test_r2:.4f} (ref {em_ref}, "
        f"delta {abs(em_test_r2 - em_ref):.4f}) → {'PASS' if em_pass else 'FAIL'}"
    )
    print(
        f"ex_max test R²: {ex_test_r2:.4f} (ref {ex_ref}, "
        f"delta {abs(ex_test_r2 - ex_ref):.4f}) → {'PASS' if ex_pass else 'FAIL'}"
    )

    overall = em_pass and ex_pass
    print(f"\nG0 Overall: {'PASS' if overall else 'FAIL'}")
    if not overall:
        print(
            "NOTE: G0 FAIL — check dataset size, MAFFT flags, "
            "hyperparameter space, data leakage."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost models for FluoroGraph"
    )
    parser.add_argument(
        "--mode",
        choices=["g0", "g1"],
        required=True,
        help="g0: FPredX replication; g1: ablation across 6 feature sets (US-007)",
    )
    args = parser.parse_args()

    if args.mode == "g0":
        run_g0()
    elif args.mode == "g1":
        print("G1 mode not yet implemented (US-007)")
        sys.exit(1)


if __name__ == "__main__":
    main()
