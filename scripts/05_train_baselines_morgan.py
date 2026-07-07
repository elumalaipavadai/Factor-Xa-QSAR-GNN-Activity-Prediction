#!/usr/bin/env python
"""
05_train_baselines_morgan.py

Step 05 of the Factor Xa portfolio workflow.

Purpose
-------
Train classical Morgan fingerprint baseline models for Factor Xa pKi prediction.

Models
------
1. RandomForest
2. XGBoost
3. LightGBM

Inputs
------
data/processed/fxa_04_modeling_dataset.csv
data/features/fxa_04_X_morgan_2048.npy
data/features/fxa_04_y_pKi.npy
data/splits/fxa_04_scaffold_train.npy
data/splits/fxa_04_scaffold_valid.npy
data/splits/fxa_04_scaffold_test.npy
data/splits/fxa_04_random_train.npy
data/splits/fxa_04_random_valid.npy
data/splits/fxa_04_random_test.npy

Outputs
-------
results/metrics/fxa_05_baseline_metrics.csv
results/metrics/fxa_05_baseline_metrics.json
results/metrics/fxa_05_baseline_summary.json
results/tables/fxa_05_baseline_predictions.csv
results/figures/fxa_05_scaffold_test_pred_vs_actual_best.png
models/fxa_05_best_scaffold_morgan_model.joblib
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

MODELING_CSV = ROOT / "data" / "processed" / "fxa_04_modeling_dataset.csv"
X_PATH = ROOT / "data" / "features" / "fxa_04_X_morgan_2048.npy"
Y_PATH = ROOT / "data" / "features" / "fxa_04_y_pKi.npy"

SPLIT_DIR = ROOT / "data" / "splits"

SPLITS = {
    "scaffold": {
        "train": SPLIT_DIR / "fxa_04_scaffold_train.npy",
        "valid": SPLIT_DIR / "fxa_04_scaffold_valid.npy",
        "test": SPLIT_DIR / "fxa_04_scaffold_test.npy",
    },
    "random": {
        "train": SPLIT_DIR / "fxa_04_random_train.npy",
        "valid": SPLIT_DIR / "fxa_04_random_valid.npy",
        "test": SPLIT_DIR / "fxa_04_random_test.npy",
    },
}

METRICS_DIR = ROOT / "results" / "metrics"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"
MODELS_DIR = ROOT / "models"

for directory in [METRICS_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Settings
# =============================================================================

TARGET_COL = "target_pKi"
SMILES_COL = "model_smiles"
RANDOM_STATE = 42


# =============================================================================
# Helpers
# =============================================================================

def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def assert_required_files_exist(paths: List[Path]):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def load_splits() -> Dict[str, Dict[str, np.ndarray]]:
    loaded = {}

    for split_type, split_paths in SPLITS.items():
        loaded[split_type] = {}
        for split_name, path in split_paths.items():
            loaded[split_type][split_name] = np.load(path).astype(int)

    return loaded


def assert_y_matches_dataframe(df: pd.DataFrame, y: np.ndarray):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    y_from_csv = pd.to_numeric(df[TARGET_COL], errors="coerce").values.astype(np.float32)

    if len(y_from_csv) != len(y):
        raise ValueError("CSV target length does not match saved y vector length.")

    if not np.allclose(y_from_csv, y, equal_nan=False):
        max_abs_diff = float(np.max(np.abs(y_from_csv - y)))
        raise ValueError(
            f"Saved y vector does not match CSV {TARGET_COL}. "
            f"Max absolute difference: {max_abs_diff}"
        )


def make_models() -> Dict[str, object]:
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    }

    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
    else:
        print("WARNING: xgboost is not installed. Skipping XGBoost.")

    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        print("WARNING: lightgbm is not installed. Skipping LightGBM.")

    return models


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | None]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    if spearmanr is not None:
        rho, pval = spearmanr(y_true, y_pred)
        spearman_r = float(rho) if not np.isnan(rho) else None
        spearman_p = float(pval) if not np.isnan(pval) else None
    else:
        spearman_r = None
        spearman_p = None

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
):
    plt.figure(figsize=(6.0, 6.0))
    plt.scatter(y_true, y_pred, alpha=0.7)

    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Actual pKi")
    plt.ylabel("Predicted pKi")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 90)
    print("STEP 05: MORGAN FINGERPRINT BASELINE MODELS")
    print("=" * 90)

    required = [
        MODELING_CSV,
        X_PATH,
        Y_PATH,
        *[path for split_paths in SPLITS.values() for path in split_paths.values()],
    ]

    assert_required_files_exist(required)

    df = pd.read_csv(MODELING_CSV)
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    print(f"Loaded dataframe: {df.shape}")
    print(f"Loaded Morgan X: {X.shape}, dtype={X.dtype}")
    print(f"Loaded y: {y.shape}, dtype={y.dtype}")

    if len(df) != X.shape[0] or len(df) != len(y):
        raise ValueError("Mismatch among dataframe rows, X rows, and y length.")

    assert_y_matches_dataframe(df, y)
    print("Target consistency check passed.")

    split_indices = load_splits()

    for split_type, split_dict in split_indices.items():
        print(
            f"{split_type} split sizes: "
            f"train={len(split_dict['train'])}, "
            f"valid={len(split_dict['valid'])}, "
            f"test={len(split_dict['test'])}"
        )

    all_metric_rows = []
    all_prediction_rows = []

    best_scaffold_valid_rmse = np.inf
    best_scaffold_model_name = None
    best_scaffold_model = None
    best_scaffold_test_predictions = None

    for split_type, split_dict in split_indices.items():
        train_idx = split_dict["train"]
        valid_idx = split_dict["valid"]
        test_idx = split_dict["test"]

        X_train = X[train_idx]
        y_train = y[train_idx]

        models = make_models()

        for model_name, model in models.items():
            print("\n" + "-" * 90)
            print(f"Training {model_name} on {split_type} split")
            print("-" * 90)

            model.fit(X_train, y_train)

            for split_name, idx in split_dict.items():
                y_true = y[idx]
                y_pred = model.predict(X[idx])

                metrics = regression_metrics(y_true, y_pred)

                row = {
                    "split_type": split_type,
                    "split": split_name,
                    "model": model_name,
                    "n": int(len(idx)),
                    **metrics,
                }

                all_metric_rows.append(row)

                print(
                    f"{split_type:8s} | {split_name:5s} | {model_name:12s} | "
                    f"n={len(idx):4d} | "
                    f"RMSE={metrics['rmse']:.3f} | "
                    f"MAE={metrics['mae']:.3f} | "
                    f"R2={metrics['r2']:.3f} | "
                    f"Spearman={metrics['spearman_r']:.3f}"
                )

                for local_rank, row_idx in enumerate(idx):
                    df_row = df.iloc[int(row_idx)]

                    all_prediction_rows.append(
                        {
                            "split_type": split_type,
                            "split": split_name,
                            "model": model_name,
                            "row_index": int(row_idx),
                            "molecule_chembl_id": df_row["molecule_chembl_id"]
                            if "molecule_chembl_id" in df.columns
                            else None,
                            "model_smiles": df_row[SMILES_COL]
                            if SMILES_COL in df.columns
                            else None,
                            "target_pKi": float(y_true[local_rank]),
                            "predicted_pKi": float(y_pred[local_rank]),
                            "residual": float(y_pred[local_rank] - y_true[local_rank]),
                            "abs_error": float(abs(y_pred[local_rank] - y_true[local_rank])),
                        }
                    )

            scaffold_valid_row = [
                r for r in all_metric_rows
                if r["split_type"] == "scaffold"
                and r["split"] == "valid"
                and r["model"] == model_name
            ]

            if split_type == "scaffold" and len(scaffold_valid_row) == 1:
                valid_rmse = scaffold_valid_row[0]["rmse"]

                if valid_rmse < best_scaffold_valid_rmse:
                    best_scaffold_valid_rmse = valid_rmse
                    best_scaffold_model_name = model_name
                    best_scaffold_model = model

                    test_idx_scaffold = split_indices["scaffold"]["test"]
                    best_scaffold_test_predictions = model.predict(X[test_idx_scaffold])

    metrics_df = pd.DataFrame(all_metric_rows)
    predictions_df = pd.DataFrame(all_prediction_rows)

    metrics_csv = METRICS_DIR / "fxa_05_baseline_metrics.csv"
    metrics_json = METRICS_DIR / "fxa_05_baseline_metrics.json"
    predictions_csv = TABLES_DIR / "fxa_05_baseline_predictions.csv"

    metrics_df.to_csv(metrics_csv, index=False)
    predictions_df.to_csv(predictions_csv, index=False)

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=2, default=json_default)

    print("\nSaved Step 05 outputs:")
    print(metrics_csv)
    print(metrics_json)
    print(predictions_csv)

    if best_scaffold_model is None:
        raise RuntimeError("No best scaffold model was selected.")

    best_model_path = MODELS_DIR / "fxa_05_best_scaffold_morgan_model.joblib"
    joblib.dump(best_scaffold_model, best_model_path)

    print(f"\nBest scaffold-validation model: {best_scaffold_model_name}")
    print(f"Best scaffold-validation RMSE: {best_scaffold_valid_rmse:.6f}")
    print(f"Saved best model: {best_model_path}")

    scaffold_test_idx = split_indices["scaffold"]["test"]
    y_test = y[scaffold_test_idx]
    y_pred_test = best_scaffold_test_predictions

    best_test_metrics = regression_metrics(y_test, y_pred_test)

    pred_fig = FIGURES_DIR / "fxa_05_scaffold_test_pred_vs_actual_best.png"
    plot_pred_vs_actual(
        y_true=y_test,
        y_pred=y_pred_test,
        title=f"Step 05 Best Morgan Baseline: {best_scaffold_model_name}",
        out_path=pred_fig,
    )

    print(f"Saved figure: {pred_fig}")

    summary = {
        "script": "05_train_baselines_morgan.py",
        "feature_set": "Morgan_2048",
        "models": sorted(metrics_df["model"].unique().tolist()),
        "selection_rule": "Best scaffold validation RMSE",
        "best_scaffold_valid_model": best_scaffold_model_name,
        "best_scaffold_valid_rmse": best_scaffold_valid_rmse,
        "best_scaffold_test_metrics": best_test_metrics,
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "metrics_json": str(metrics_json),
            "predictions_csv": str(predictions_csv),
            "best_model": str(best_model_path),
            "figure": str(pred_fig),
        },
    }

    summary_json = METRICS_DIR / "fxa_05_baseline_summary.json"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)

    print(f"Saved summary: {summary_json}")

    print("\nFinal Step 05 scaffold test rows:")
    print(
        metrics_df[
            (metrics_df["split_type"] == "scaffold")
            & (metrics_df["split"] == "test")
        ].sort_values("rmse").to_string(index=False)
    )

    print("\n" + "=" * 90)
    print("STEP 05 COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()