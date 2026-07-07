#!/usr/bin/env python
"""
05b_benchmark_morgan_rdkit2d_combined_features.py

Step 05b of the Factor Xa portfolio workflow.

Purpose
-------
Benchmark classical ML baselines across three molecular feature sets:

1. Morgan fingerprints only
2. RDKit 2D descriptors only
3. Morgan fingerprints + RDKit 2D descriptors

Important leakage fix
---------------------
Raw RDKit 2D descriptor values are computed molecule-by-molecule for all molecules.
That is not leakage.

But imputation and zero-variance feature removal are NOT done globally.
They are fit only on the training split using a sklearn Pipeline:

    SimpleImputer(strategy="median")
    VarianceThreshold(0.0)
    Model

This prevents validation/test molecules from influencing preprocessing.

Models
------
1. Random Forest
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
data/features/fxa_05b_X_rdkit2d_raw.npy
data/features/fxa_05b_X_morgan_plus_rdkit2d_raw.npy
data/features/fxa_05b_rdkit2d_descriptor_names.txt
data/features/fxa_05b_rdkit2d_raw_summary.json

results/metrics/fxa_05b_feature_set_metrics.csv
results/metrics/fxa_05b_feature_set_metrics.json
results/metrics/fxa_05b_feature_set_summary.json

results/tables/fxa_05b_feature_set_predictions.csv

results/figures/fxa_05b_scaffold_valid_rmse_by_feature_set.png
results/figures/fxa_05b_scaffold_test_rmse_by_feature_set.png
results/figures/fxa_05b_best_scaffold_test_pred_vs_actual.png

models/fxa_05b_<split>_<feature_set>_<model>.joblib
models/fxa_05b_best_scaffold_feature_model.joblib

Important
---------
Use scaffold split as the headline result.
Use random split only as comparison.
Select best feature/model combination by scaffold validation RMSE, not test RMSE.

How to run:
-----------
python -m py_compile .\scripts\05b_benchmark_morgan_rdkit2d_combined_features.py
python .\scripts\05b_benchmark_morgan_rdkit2d_combined_features.py *> .\scripts\05b.log

"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


# =============================================================================
# Warning cleanup
# =============================================================================

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
)

warnings.filterwarnings(
    "ignore",
    message="Skipping features without any observed values.*",
)


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

MODELING_CSV = ROOT / "data" / "processed" / "fxa_04_modeling_dataset.csv"
X_MORGAN_PATH = ROOT / "data" / "features" / "fxa_04_X_morgan_2048.npy"
Y_PATH = ROOT / "data" / "features" / "fxa_04_y_pKi.npy"

FEATURE_DIR = ROOT / "data" / "features"
SPLIT_DIR = ROOT / "data" / "splits"

METRICS_DIR = ROOT / "results" / "metrics"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"
MODELS_DIR = ROOT / "models"

for directory in [FEATURE_DIR, METRICS_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Settings
# =============================================================================

RANDOM_SEED = 42
TARGET_COL = "target_pKi"
SMILES_COL = "model_smiles"


# =============================================================================
# Helper functions
# =============================================================================

def safe_token(text: str) -> str:
    """Make a safe lowercase filename token."""
    return (
        text.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("+", "plus")
        .replace("/", "_")
    )


def smiles_to_mol(smiles: str):
    """Convert SMILES to RDKit Mol."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def get_rdkit_descriptor_functions():
    """
    Return RDKit 2D descriptor functions.

    Descriptors._descList is widely used for descriptor benchmarking and gives
    a stable list of name/function pairs in this workflow.
    """
    return Descriptors._descList


def compute_rdkit2d_raw_descriptors(
    smiles_list: List[str],
) -> Tuple[np.ndarray, List[str], Dict[str, object]]:
    """
    Compute raw RDKit 2D descriptor matrix.

    This function does NOT impute missing values.
    This function does NOT drop zero-variance descriptors.
    This function does NOT fit any preprocessing statistics.

    Those preprocessing steps are done later inside sklearn Pipeline and fit
    only on the training split to avoid leakage.
    """

    descriptor_functions = get_rdkit_descriptor_functions()
    descriptor_names = [name for name, _ in descriptor_functions]

    rows = []
    invalid_count = 0
    descriptor_failure_count = 0

    print(f"Computing raw RDKit 2D descriptors: {len(descriptor_functions)} descriptors")

    for smi in smiles_list:
        mol = smiles_to_mol(smi)

        if mol is None:
            invalid_count += 1
            rows.append([np.nan] * len(descriptor_functions))
            continue

        values = []

        for _, func in descriptor_functions:
            try:
                value = func(mol)
                if value is None:
                    value = np.nan
            except Exception:
                value = np.nan
                descriptor_failure_count += 1

            values.append(value)

        rows.append(values)

    X_raw = np.array(rows, dtype=np.float64)

    # Element-wise replacement only. This is not fitted preprocessing.
    X_raw[~np.isfinite(X_raw)] = np.nan

    float32_safe_max = np.finfo(np.float32).max / 10.0
    too_large_mask = np.abs(X_raw) > float32_safe_max
    n_too_large_entries = int(too_large_mask.sum())
    X_raw[too_large_mask] = np.nan

    n_nan_entries = int(np.isnan(X_raw).sum())
    n_total_entries = int(X_raw.size)

    col_all_nan = np.isnan(X_raw).all(axis=0)
    n_all_nan_cols = int(col_all_nan.sum())

    summary = {
        "n_molecules": int(len(smiles_list)),
        "n_invalid_molecules": int(invalid_count),
        "n_descriptors_raw": int(len(descriptor_names)),
        "n_descriptor_value_failures": int(descriptor_failure_count),
        "n_total_matrix_entries": n_total_entries,
        "n_nan_entries": n_nan_entries,
        "fraction_nan_entries": float(n_nan_entries / n_total_entries) if n_total_entries else 0.0,
        "n_all_nan_descriptor_columns_raw": n_all_nan_cols,
        "all_nan_descriptor_columns_raw": [
            descriptor_names[i] for i, is_all_nan in enumerate(col_all_nan) if is_all_nan
        ],
        "important_note": (
            "This is a raw descriptor matrix. Imputation and variance filtering "
            "are fit only on each training split inside sklearn Pipeline."
        ),
    }

    return X_raw, descriptor_names, summary


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | None]:
    """Compute regression metrics."""
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


def load_split(split_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Step 04 train/valid/test split indices."""
    train = np.load(SPLIT_DIR / f"fxa_04_{split_name}_train.npy")
    valid = np.load(SPLIT_DIR / f"fxa_04_{split_name}_valid.npy")
    test = np.load(SPLIT_DIR / f"fxa_04_{split_name}_test.npy")
    return train, valid, test


def make_base_models() -> Dict[str, object]:
    """Create fresh base model objects."""
    models: Dict[str, object] = {
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=700,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    else:
        print("WARNING: XGBoost is not installed. Skipping XGBoost.")

    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            subsample_freq=1,  # required for LightGBM row subsampling
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbosity=-1,
        )
    else:
        print("WARNING: LightGBM is not installed. Skipping LightGBM.")

    return models


def make_leak_free_pipeline(model) -> Pipeline:
    """
    Create preprocessing + model pipeline.

    The imputer and variance filter are fit only on training molecules.
    This prevents validation/test preprocessing leakage.
    """
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(threshold=0.0)),
            ("model", model),
        ]
    )


def get_n_features_after_pipeline_fit(pipeline: Pipeline) -> int | None:
    """Return number of features after imputation + variance threshold."""
    try:
        support = pipeline.named_steps["var"].get_support()
        return int(np.sum(support))
    except Exception:
        return None


def evaluate_pipeline(
    pipeline: Pipeline,
    model_name: str,
    feature_set: str,
    split_type: str,
    split_label: str,
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
) -> Dict[str, object]:
    """Evaluate fitted pipeline on one split."""
    y_true = y[idx]
    y_pred = pipeline.predict(X[idx])

    metrics = regression_metrics(y_true, y_pred)

    return {
        "feature_set": feature_set,
        "split_type": split_type,
        "split": split_label,
        "model": model_name,
        "n": int(len(idx)),
        **metrics,
    }


def make_prediction_rows(
    df: pd.DataFrame,
    pipeline: Pipeline,
    model_name: str,
    feature_set: str,
    split_type: str,
    split_label: str,
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
) -> List[Dict[str, object]]:
    """Create molecule-level predictions for valid/test splits."""
    y_true = y[idx]
    y_pred = pipeline.predict(X[idx])

    rows: List[Dict[str, object]] = []

    for row_idx, true_val, pred_val in zip(idx, y_true, y_pred):
        row = df.iloc[int(row_idx)]

        rows.append(
            {
                "feature_set": feature_set,
                "split_type": split_type,
                "split": split_label,
                "model": model_name,
                "row_index": int(row_idx),
                "molecule_chembl_id": row["molecule_chembl_id"]
                if "molecule_chembl_id" in df.columns
                else None,
                "model_smiles": row[SMILES_COL]
                if SMILES_COL in df.columns
                else None,
                "y_true_pKi": float(true_val),
                "y_pred_pKi": float(pred_val),
                "residual_true_minus_pred": float(true_val - pred_val),
            }
        )

    return rows


def assert_required_files_exist(paths: List[Path]):
    """Fail if required files are missing."""
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input files. Run Step 04 first.\n" + "\n".join(missing)
        )


def assert_y_matches_dataframe(df: pd.DataFrame, y: np.ndarray):
    """Check y vector matches target_pKi in modeling CSV."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Modeling CSV missing target column: {TARGET_COL}")

    y_from_csv = pd.to_numeric(df[TARGET_COL], errors="coerce").values.astype(np.float32)

    if len(y_from_csv) != len(y):
        raise ValueError("CSV target length does not match y vector length.")

    if not np.allclose(y_from_csv, y, equal_nan=False):
        max_abs_diff = float(np.max(np.abs(y_from_csv - y)))
        raise ValueError(
            f"Saved y vector does not match CSV {TARGET_COL}. "
            f"Max absolute difference: {max_abs_diff}"
        )


def plot_scaffold_valid_rmse(metrics_df: pd.DataFrame, out_path: Path):
    """Plot scaffold validation RMSE by feature set/model."""
    plot_df = metrics_df[
        (metrics_df["split_type"] == "scaffold")
        & (metrics_df["split"] == "valid")
    ].copy()

    plot_df["label"] = plot_df["feature_set"] + "\n" + plot_df["model"]
    plot_df = plot_df.sort_values("rmse")

    plt.figure(figsize=(11, 6))
    plt.bar(plot_df["label"], plot_df["rmse"])
    plt.ylabel("Scaffold validation RMSE")
    plt.xlabel("Feature set + model")
    plt.title("Factor Xa baselines: scaffold validation RMSE")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_scaffold_test_rmse(metrics_df: pd.DataFrame, out_path: Path):
    """Plot scaffold test RMSE by feature set/model."""
    plot_df = metrics_df[
        (metrics_df["split_type"] == "scaffold")
        & (metrics_df["split"] == "test")
    ].copy()

    plot_df["label"] = plot_df["feature_set"] + "\n" + plot_df["model"]
    plot_df = plot_df.sort_values("rmse")

    plt.figure(figsize=(11, 6))
    plt.bar(plot_df["label"], plot_df["rmse"])
    plt.ylabel("Scaffold test RMSE")
    plt.xlabel("Feature set + model")
    plt.title("Factor Xa baselines: scaffold test RMSE")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
):
    """Plot predicted vs actual pKi."""
    plt.figure(figsize=(6, 6))
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
    print("STEP 05b: MORGAN / RDKIT 2D / COMBINED FEATURE BENCHMARK")
    print("=" * 90)

    print(f"Project root: {ROOT}")

    required_files = [
        MODELING_CSV,
        X_MORGAN_PATH,
        Y_PATH,
        SPLIT_DIR / "fxa_04_scaffold_train.npy",
        SPLIT_DIR / "fxa_04_scaffold_valid.npy",
        SPLIT_DIR / "fxa_04_scaffold_test.npy",
        SPLIT_DIR / "fxa_04_random_train.npy",
        SPLIT_DIR / "fxa_04_random_valid.npy",
        SPLIT_DIR / "fxa_04_random_test.npy",
    ]

    assert_required_files_exist(required_files)

    df = pd.read_csv(MODELING_CSV)
    X_morgan = np.load(X_MORGAN_PATH).astype(np.float64)
    y = np.load(Y_PATH)

    print(f"Loaded modeling dataframe: {df.shape}")
    print(f"Loaded Morgan X: {X_morgan.shape}, dtype={X_morgan.dtype}")
    print(f"Loaded y: {y.shape}, dtype={y.dtype}")

    if SMILES_COL not in df.columns:
        raise ValueError(f"Modeling CSV missing SMILES column: {SMILES_COL}")

    if len(df) != X_morgan.shape[0] or len(df) != len(y):
        raise ValueError("Mismatch among dataframe rows, Morgan X rows, and y length.")

    assert_y_matches_dataframe(df, y)
    print("Target consistency check passed.")

    # -------------------------------------------------------------------------
    # Raw RDKit 2D descriptors
    # -------------------------------------------------------------------------
    X_rdkit2d_raw, descriptor_names, descriptor_raw_summary = compute_rdkit2d_raw_descriptors(
        df[SMILES_COL].tolist()
    )

    if X_rdkit2d_raw.shape[0] != len(df):
        raise ValueError("RDKit 2D descriptor row count does not match dataframe.")

    X_combined_raw = np.concatenate(
        [X_morgan, X_rdkit2d_raw],
        axis=1,
    )

    rdkit2d_path = FEATURE_DIR / "fxa_05b_X_rdkit2d_raw.npy"
    combined_path = FEATURE_DIR / "fxa_05b_X_morgan_plus_rdkit2d_raw.npy"
    desc_names_path = FEATURE_DIR / "fxa_05b_rdkit2d_descriptor_names.txt"
    desc_summary_path = FEATURE_DIR / "fxa_05b_rdkit2d_raw_summary.json"

    np.save(rdkit2d_path, X_rdkit2d_raw)
    np.save(combined_path, X_combined_raw)

    with open(desc_names_path, "w", encoding="utf-8") as f:
        for name in descriptor_names:
            f.write(f"{name}\n")

    with open(desc_summary_path, "w", encoding="utf-8") as f:
        json.dump(descriptor_raw_summary, f, indent=2)

    print("\nSaved raw RDKit 2D feature files:")
    print(rdkit2d_path)
    print(combined_path)
    print(desc_names_path)
    print(desc_summary_path)

    print(f"Raw RDKit 2D X shape: {X_rdkit2d_raw.shape}")
    print(f"Raw combined X shape: {X_combined_raw.shape}")

    feature_sets = {
        "Morgan_2048": X_morgan,
        "RDKit_2D_raw": X_rdkit2d_raw,
        "Morgan_plus_RDKit_2D_raw": X_combined_raw,
    }

    all_metric_rows: List[Dict[str, object]] = []
    all_prediction_rows: List[Dict[str, object]] = []
    model_paths: Dict[Tuple[str, str, str], Path] = {}
    preprocessing_rows: List[Dict[str, object]] = []

    # -------------------------------------------------------------------------
    # Train and evaluate
    # -------------------------------------------------------------------------
    for split_type in ["scaffold", "random"]:
        train_idx, valid_idx, test_idx = load_split(split_type)

        print("\n" + "-" * 90)
        print(f"Split type: {split_type}")
        print(
            f"Split sizes: train={len(train_idx)}, "
            f"valid={len(valid_idx)}, test={len(test_idx)}"
        )
        print("-" * 90)

        for feature_set_name, X in feature_sets.items():
            print(f"\nFeature set: {feature_set_name}")
            print(f"Raw X shape: {X.shape}")

            base_models = make_base_models()

            for model_name, base_model in base_models.items():
                print(f"Training {model_name} on {split_type} / {feature_set_name}...")

                pipeline = make_leak_free_pipeline(base_model)

                # Critical leak-free line:
                # imputer + variance threshold + model are fit on train_idx only.
                pipeline.fit(X[train_idx], y[train_idx])

                n_features_after = get_n_features_after_pipeline_fit(pipeline)

                model_token = safe_token(model_name)
                feature_token = safe_token(feature_set_name)

                model_path = (
                    MODELS_DIR
                    / f"fxa_05b_{split_type}_{feature_token}_{model_token}.joblib"
                )

                joblib.dump(pipeline, model_path)
                model_paths[(split_type, feature_set_name, model_name)] = model_path

                preprocessing_rows.append(
                    {
                        "split_type": split_type,
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "raw_n_features": int(X.shape[1]),
                        "n_features_after_train_fit_preprocessing": n_features_after,
                        "preprocessing": "SimpleImputer(median) + VarianceThreshold(0.0), fit on train only",
                        "model_path": str(model_path),
                    }
                )

                for split_label, idx in [
                    ("train", train_idx),
                    ("valid", valid_idx),
                    ("test", test_idx),
                ]:
                    metric_row = evaluate_pipeline(
                        pipeline=pipeline,
                        model_name=model_name,
                        feature_set=feature_set_name,
                        split_type=split_type,
                        split_label=split_label,
                        X=X,
                        y=y,
                        idx=idx,
                    )

                    all_metric_rows.append(metric_row)

                    spearman_display = metric_row["spearman_r"]
                    if spearman_display is not None:
                        spearman_display = f"{spearman_display:.3f}"

                    print(
                        f"{feature_set_name:28s} | {model_name:12s} | "
                        f"{split_type:8s} | {split_label:5s} | "
                        f"RMSE={metric_row['rmse']:.3f} | "
                        f"MAE={metric_row['mae']:.3f} | "
                        f"R2={metric_row['r2']:.3f} | "
                        f"Spearman={spearman_display}"
                    )

                    if split_label in ["valid", "test"]:
                        pred_rows = make_prediction_rows(
                            df=df,
                            pipeline=pipeline,
                            model_name=model_name,
                            feature_set=feature_set_name,
                            split_type=split_type,
                            split_label=split_label,
                            X=X,
                            y=y,
                            idx=idx,
                        )
                        all_prediction_rows.extend(pred_rows)

    # -------------------------------------------------------------------------
    # Save metrics and predictions
    # -------------------------------------------------------------------------
    metrics_df = pd.DataFrame(all_metric_rows)
    predictions_df = pd.DataFrame(all_prediction_rows)
    preprocessing_df = pd.DataFrame(preprocessing_rows)

    metrics_csv = METRICS_DIR / "fxa_05b_feature_set_metrics.csv"
    metrics_json = METRICS_DIR / "fxa_05b_feature_set_metrics.json"
    predictions_csv = TABLES_DIR / "fxa_05b_feature_set_predictions.csv"
    preprocessing_csv = METRICS_DIR / "fxa_05b_preprocessing_fit_summary.csv"

    metrics_df.to_csv(metrics_csv, index=False)
    predictions_df.to_csv(predictions_csv, index=False)
    preprocessing_df.to_csv(preprocessing_csv, index=False)

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(all_metric_rows, f, indent=2)

    print("\nSaved metrics, predictions, and preprocessing summary:")
    print(metrics_csv)
    print(metrics_json)
    print(predictions_csv)
    print(preprocessing_csv)

    # -------------------------------------------------------------------------
    # Best model selected by scaffold validation RMSE
    # -------------------------------------------------------------------------
    scaffold_valid = metrics_df[
        (metrics_df["split_type"] == "scaffold")
        & (metrics_df["split"] == "valid")
    ].copy()

    scaffold_valid = scaffold_valid.sort_values("rmse")
    best_valid_row = scaffold_valid.iloc[0]

    best_feature_set = str(best_valid_row["feature_set"])
    best_model_name = str(best_valid_row["model"])

    best_model_path = model_paths[("scaffold", best_feature_set, best_model_name)]
    best_copy_path = MODELS_DIR / "fxa_05b_best_scaffold_feature_model.joblib"
    shutil.copyfile(best_model_path, best_copy_path)

    print("\nBest feature/model selected by scaffold validation RMSE:")
    print(best_valid_row.to_string())
    print(f"Best model copied to: {best_copy_path}")

    scaffold_test_row = metrics_df[
        (metrics_df["split_type"] == "scaffold")
        & (metrics_df["split"] == "test")
        & (metrics_df["feature_set"] == best_feature_set)
        & (metrics_df["model"] == best_model_name)
    ].iloc[0]

    print("\nBest feature/model scaffold test metrics:")
    print(scaffold_test_row.to_string())

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    valid_rmse_fig = FIGURES_DIR / "fxa_05b_scaffold_valid_rmse_by_feature_set.png"
    test_rmse_fig = FIGURES_DIR / "fxa_05b_scaffold_test_rmse_by_feature_set.png"
    best_pred_fig = FIGURES_DIR / "fxa_05b_best_scaffold_test_pred_vs_actual.png"

    plot_scaffold_valid_rmse(metrics_df, valid_rmse_fig)
    plot_scaffold_test_rmse(metrics_df, test_rmse_fig)

    best_X = feature_sets[best_feature_set]
    best_pipeline = joblib.load(best_model_path)
    scaffold_test_idx = np.load(SPLIT_DIR / "fxa_04_scaffold_test.npy")

    y_true_test = y[scaffold_test_idx]
    y_pred_test = best_pipeline.predict(best_X[scaffold_test_idx])

    plot_pred_vs_actual(
        y_true=y_true_test,
        y_pred=y_pred_test,
        title=f"Best feature baseline: {best_feature_set} + {best_model_name}",
        out_path=best_pred_fig,
    )

    print("\nSaved figures:")
    print(valid_rmse_fig)
    print(test_rmse_fig)
    print(best_pred_fig)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    summary = {
        "script": "05b_benchmark_morgan_rdkit2d_combined_features.py",
        "input_modeling_csv": str(MODELING_CSV),
        "input_morgan_X": str(X_MORGAN_PATH),
        "input_y": str(Y_PATH),

        "feature_sets": {
            "Morgan_2048": {
                "shape": list(feature_sets["Morgan_2048"].shape),
                "source": str(X_MORGAN_PATH),
            },
            "RDKit_2D_raw": {
                "shape": list(feature_sets["RDKit_2D_raw"].shape),
                "source": str(rdkit2d_path),
                "descriptor_names": str(desc_names_path),
                "raw_descriptor_summary": str(desc_summary_path),
            },
            "Morgan_plus_RDKit_2D_raw": {
                "shape": list(feature_sets["Morgan_plus_RDKit_2D_raw"].shape),
                "source": str(combined_path),
            },
        },

        "leakage_control": {
            "raw_descriptor_computation": "Computed independently per molecule.",
            "imputation": "SimpleImputer(strategy='median') fit on train split only.",
            "variance_filtering": "VarianceThreshold(0.0) fit on train split only.",
            "validation_test_use": "Validation/test splits are transformed only by preprocessing learned from train.",
        },

        "models_trained": sorted(list(set(metrics_df["model"].tolist()))),
        "headline_split": "scaffold",
        "comparison_split": "random",
        "selection_rule": "Best feature/model selected by scaffold validation RMSE, not test RMSE.",

        "best_scaffold_valid_feature_set": best_feature_set,
        "best_scaffold_valid_model": best_model_name,
        "best_scaffold_valid_metrics": best_valid_row.to_dict(),
        "best_scaffold_test_metrics": scaffold_test_row.to_dict(),
        "best_model_path": str(best_model_path),
        "best_model_copy": str(best_copy_path),

        "metrics_csv": str(metrics_csv),
        "metrics_json": str(metrics_json),
        "predictions_csv": str(predictions_csv),
        "preprocessing_csv": str(preprocessing_csv),

        "figures": [
            str(valid_rmse_fig),
            str(test_rmse_fig),
            str(best_pred_fig),
        ],

        "important_note": (
            "Use scaffold test performance as the headline comparison. "
            "Random split performance is comparison only and usually more optimistic. "
            "This benchmark compares Morgan fingerprints, raw RDKit 2D descriptors, "
            "and raw Morgan + RDKit 2D descriptors using the exact same saved Step 04 splits. "
            "All preprocessing statistics are fit only on the training split."
        ),
    }

    summary_path = METRICS_DIR / "fxa_05b_feature_set_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nSaved summary:")
    print(summary_path)

    print("\n" + "=" * 90)
    print("STEP 05b COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()


