#!/usr/bin/env python
"""
How to run:
-----------
python -m py_compile .\scripts\07_uncertainty_error_analysis.py
python .\scripts\07_uncertainty_error_analysis.py *> .\scripts\07.log
Get-Content .\scripts\07.log -Tail 100

07_uncertainty_error_analysis.py

Step 07 of the Factor Xa portfolio workflow.

Purpose
-------
Perform uncertainty and error analysis on the best classical baseline model
from Step 05b.

Current best model:
    Morgan_2048 + RandomForest

Uncertainty approach
--------------------
For RandomForestRegressor, uncertainty is estimated as the standard deviation
of predictions across individual trees.

This is not a fully calibrated Bayesian uncertainty estimate, but it is a useful
practical ensemble-disagreement signal.

Inputs
------
data/processed/fxa_04_modeling_dataset.csv
data/features/fxa_04_X_morgan_2048.npy
data/features/fxa_05b_X_rdkit2d_raw.npy
data/features/fxa_05b_X_morgan_plus_rdkit2d_raw.npy

data/splits/fxa_04_scaffold_valid.npy
data/splits/fxa_04_scaffold_test.npy

results/metrics/fxa_05b_feature_set_summary.json
models/fxa_05b_best_scaffold_feature_model.joblib

Outputs
-------
results/metrics/fxa_07_uncertainty_error_summary.json
results/metrics/fxa_07_uncertainty_error_metrics.csv
results/tables/fxa_07_scaffold_valid_test_predictions_with_uncertainty.csv
results/tables/fxa_07_high_error_molecules.csv
results/tables/fxa_07_high_uncertainty_molecules.csv
results/tables/fxa_07_uncertainty_decile_analysis.csv
results/tables/fxa_07_error_by_pki_bin.csv

results/figures/fxa_07_scaffold_test_pred_vs_actual_uncertainty.png
results/figures/fxa_07_scaffold_test_abs_error_vs_uncertainty.png
results/figures/fxa_07_uncertainty_deciles_abs_error.png
results/figures/fxa_07_error_by_pki_bin.png

How to run
----------
python -m py_compile .\\scripts\\07_uncertainty_error_analysis.py
python .\\scripts\\07_uncertainty_error_analysis.py *> .\\scripts\\07.log
"""

from __future__ import annotations

import json
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
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.pipeline import Pipeline

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

MODELING_CSV = ROOT / "data" / "processed" / "fxa_04_modeling_dataset.csv"

X_MORGAN_PATH = ROOT / "data" / "features" / "fxa_04_X_morgan_2048.npy"
X_RDKIT2D_PATH = ROOT / "data" / "features" / "fxa_05b_X_rdkit2d_raw.npy"
X_COMBINED_PATH = ROOT / "data" / "features" / "fxa_05b_X_morgan_plus_rdkit2d_raw.npy"

SPLIT_DIR = ROOT / "data" / "splits"

SUMMARY_05B = ROOT / "results" / "metrics" / "fxa_05b_feature_set_summary.json"
BEST_MODEL_PATH = ROOT / "models" / "fxa_05b_best_scaffold_feature_model.joblib"

METRICS_DIR = ROOT / "results" / "metrics"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"

for directory in [METRICS_DIR, TABLES_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Settings
# =============================================================================

SMILES_COL = "model_smiles"
TARGET_COL = "target_pKi"


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


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_matrix(feature_set: str) -> np.ndarray:
    if feature_set == "Morgan_2048":
        return np.load(X_MORGAN_PATH)

    if feature_set == "RDKit_2D_raw":
        return np.load(X_RDKIT2D_PATH)

    if feature_set == "Morgan_plus_RDKit_2D_raw":
        return np.load(X_COMBINED_PATH)

    raise ValueError(f"Unsupported feature set: {feature_set}")


def load_scaffold_valid_test_indices() -> Tuple[np.ndarray, np.ndarray]:
    valid_idx = np.load(SPLIT_DIR / "fxa_04_scaffold_valid.npy")
    test_idx = np.load(SPLIT_DIR / "fxa_04_scaffold_test.npy")
    return valid_idx, test_idx


def get_pipeline_parts(model_object):
    """
    Return preprocessor and final estimator.

    Step 05b saved sklearn Pipeline objects:
        SimpleImputer -> VarianceThreshold -> model
    """

    if isinstance(model_object, Pipeline):
        preprocessor = Pipeline(model_object.steps[:-1])
        final_model = model_object.steps[-1][1]
        return preprocessor, final_model

    return None, model_object


def predict_with_rf_uncertainty(model_object, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict with RandomForest and estimate uncertainty by tree prediction std.

    Returns:
        mean prediction from model/pipeline
        std across individual tree predictions
    """

    preprocessor, final_model = get_pipeline_parts(model_object)

    y_pred = model_object.predict(X)

    if not hasattr(final_model, "estimators_"):
        raise TypeError(
            "Best model does not expose estimators_. "
            "This Step 07 script expects RandomForestRegressor."
        )

    if preprocessor is not None:
        X_model = preprocessor.transform(X)
    else:
        X_model = X

    tree_preds = np.vstack([tree.predict(X_model) for tree in final_model.estimators_])
    tree_std = np.std(tree_preds, axis=0, ddof=1)

    return y_pred.astype(float), tree_std.astype(float)


def smiles_to_scaffold(smiles: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return None

        return Chem.MolToSmiles(scaffold, isomericSmiles=False)
    except Exception:
        return None


def compute_basic_properties(smiles: str) -> Dict[str, float | None]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid mol")

        return {
            "MW": float(Descriptors.MolWt(mol)),
            "LogP": float(Descriptors.MolLogP(mol)),
            "TPSA": float(Descriptors.TPSA(mol)),
            "HBD": float(Descriptors.NumHDonors(mol)),
            "HBA": float(Descriptors.NumHAcceptors(mol)),
            "RotBonds": float(Descriptors.NumRotatableBonds(mol)),
            "HeavyAtoms": float(Descriptors.HeavyAtomCount(mol)),
            "RingCount": float(Descriptors.RingCount(mol)),
        }
    except Exception:
        return {
            "MW": None,
            "LogP": None,
            "TPSA": None,
            "HBD": None,
            "HBA": None,
            "RotBonds": None,
            "HeavyAtoms": None,
            "RingCount": None,
        }


def make_prediction_dataframe(
    df: pd.DataFrame,
    X: np.ndarray,
    model_object,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for split_label, idx_array in [("valid", valid_idx), ("test", test_idx)]:
        X_split = X[idx_array]
        y_pred, uncertainty = predict_with_rf_uncertainty(model_object, X_split)

        for local_pos, row_idx in enumerate(idx_array):
            row = df.iloc[int(row_idx)]
            smiles = row[SMILES_COL]
            y_true = float(row[TARGET_COL])
            pred = float(y_pred[local_pos])
            unc = float(uncertainty[local_pos])
            residual = y_true - pred
            abs_error = abs(residual)

            prop_dict = compute_basic_properties(smiles)

            rows.append(
                {
                    "split_type": "scaffold",
                    "split": split_label,
                    "row_index": int(row_idx),
                    "molecule_chembl_id": row["molecule_chembl_id"]
                    if "molecule_chembl_id" in df.columns
                    else None,
                    "model_smiles": smiles,
                    "scaffold_smiles": row["scaffold_smiles"]
                    if "scaffold_smiles" in df.columns
                    else smiles_to_scaffold(smiles),
                    "y_true_pKi": y_true,
                    "y_pred_pKi": pred,
                    "residual_true_minus_pred": residual,
                    "abs_error": abs_error,
                    "rf_tree_std": unc,
                    **prop_dict,
                }
            )

    pred_df = pd.DataFrame(rows)

    pred_df["uncertainty_rank_desc"] = pred_df["rf_tree_std"].rank(
        ascending=False, method="first"
    )
    pred_df["abs_error_rank_desc"] = pred_df["abs_error"].rank(
        ascending=False, method="first"
    )

    return pred_df


def summarize_split(pred_df: pd.DataFrame, split_label: str) -> Dict[str, object]:
    subset = pred_df[pred_df["split"] == split_label].copy()

    y_true = subset["y_true_pKi"].values
    y_pred = subset["y_pred_pKi"].values
    abs_error = subset["abs_error"].values
    uncertainty = subset["rf_tree_std"].values

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(abs_error))
    r2_num = float(np.sum((y_true - y_pred) ** 2))
    r2_den = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - r2_num / r2_den) if r2_den > 0 else None

    if spearmanr is not None:
        rho_unc_error, p_unc_error = spearmanr(uncertainty, abs_error)
        rho_pred_true, p_pred_true = spearmanr(y_true, y_pred)
    else:
        rho_unc_error, p_unc_error = None, None
        rho_pred_true, p_pred_true = None, None

    coverage_rows = []

    for k in [0.5, 1.0, 1.5, 2.0]:
        covered = abs_error <= (k * uncertainty)
        coverage_rows.append(
            {
                "k": k,
                "coverage": float(np.mean(covered)),
                "mean_interval_half_width": float(np.mean(k * uncertainty)),
            }
        )

    high_error_threshold = float(np.quantile(abs_error, 0.75))
    is_high_error = abs_error >= high_error_threshold
    baseline_high_error_rate = float(np.mean(is_high_error))

    enrichment_rows = []

    for frac in [0.05, 0.10, 0.20]:
        n_top = max(1, int(round(frac * len(subset))))
        top_unc = subset.sort_values("rf_tree_std", ascending=False).head(n_top)
        precision = float(np.mean(top_unc["abs_error"] >= high_error_threshold))
        enrichment = (
            precision / baseline_high_error_rate
            if baseline_high_error_rate > 0
            else None
        )

        enrichment_rows.append(
            {
                "top_uncertainty_fraction": frac,
                "n_top": int(n_top),
                "precision_for_top25pct_error": precision,
                "enrichment_over_random": enrichment,
            }
        )

    return {
        "split": split_label,
        "n": int(len(subset)),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "spearman_true_pred": float(rho_pred_true)
        if rho_pred_true is not None and not np.isnan(rho_pred_true)
        else None,
        "spearman_true_pred_p": float(p_pred_true)
        if p_pred_true is not None and not np.isnan(p_pred_true)
        else None,
        "mean_uncertainty_rf_tree_std": float(np.mean(uncertainty)),
        "median_uncertainty_rf_tree_std": float(np.median(uncertainty)),
        "spearman_uncertainty_abs_error": float(rho_unc_error)
        if rho_unc_error is not None and not np.isnan(rho_unc_error)
        else None,
        "spearman_uncertainty_abs_error_p": float(p_unc_error)
        if p_unc_error is not None and not np.isnan(p_unc_error)
        else None,
        "high_error_threshold_abs_error_75pct": high_error_threshold,
        "baseline_high_error_rate": baseline_high_error_rate,
        "coverage_by_k_times_uncertainty": coverage_rows,
        "top_uncertainty_enrichment": enrichment_rows,
    }


def make_uncertainty_deciles(pred_df: pd.DataFrame, split_label: str) -> pd.DataFrame:
    subset = pred_df[pred_df["split"] == split_label].copy()

    subset["uncertainty_decile"] = pd.qcut(
        subset["rf_tree_std"],
        q=10,
        labels=False,
        duplicates="drop",
    )

    rows = []

    for decile, group in subset.groupby("uncertainty_decile", observed=True):
        rows.append(
            {
                "split": split_label,
                "uncertainty_decile": int(decile),
                "n": int(len(group)),
                "mean_uncertainty": float(group["rf_tree_std"].mean()),
                "median_uncertainty": float(group["rf_tree_std"].median()),
                "mean_abs_error": float(group["abs_error"].mean()),
                "median_abs_error": float(group["abs_error"].median()),
                "rmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                group["y_true_pKi"].values
                                - group["y_pred_pKi"].values
                            )
                            ** 2
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def make_error_by_pki_bin(pred_df: pd.DataFrame, split_label: str) -> pd.DataFrame:
    subset = pred_df[pred_df["split"] == split_label].copy()

    bins = [-np.inf, 6.0, 7.0, 8.0, np.inf]
    labels = ["pKi < 6", "6 <= pKi < 7", "7 <= pKi < 8", "pKi >= 8"]

    subset["pki_bin"] = pd.cut(
        subset["y_true_pKi"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    rows = []

    for pki_bin, group in subset.groupby("pki_bin", observed=True):
        rows.append(
            {
                "split": split_label,
                "pki_bin": str(pki_bin),
                "n": int(len(group)),
                "mean_true_pKi": float(group["y_true_pKi"].mean()),
                "mean_pred_pKi": float(group["y_pred_pKi"].mean()),
                "mean_abs_error": float(group["abs_error"].mean()),
                "median_abs_error": float(group["abs_error"].median()),
                "rmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                group["y_true_pKi"].values
                                - group["y_pred_pKi"].values
                            )
                            ** 2
                        )
                    )
                ),
                "mean_uncertainty": float(group["rf_tree_std"].mean()),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================

def plot_pred_vs_actual_uncertainty(pred_df: pd.DataFrame, out_path: Path):
    subset = pred_df[pred_df["split"] == "test"].copy()

    y_true = subset["y_true_pKi"].values
    y_pred = subset["y_pred_pKi"].values
    uncertainty = subset["rf_tree_std"].values

    plt.figure(figsize=(6.5, 6))
    scatter = plt.scatter(y_true, y_pred, c=uncertainty, alpha=0.75)
    plt.colorbar(scatter, label="RF tree prediction std")

    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Actual pKi")
    plt.ylabel("Predicted pKi")
    plt.title("Scaffold test: predicted vs actual colored by uncertainty")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_abs_error_vs_uncertainty(pred_df: pd.DataFrame, out_path: Path):
    subset = pred_df[pred_df["split"] == "test"].copy()

    plt.figure(figsize=(6.5, 5))
    plt.scatter(subset["rf_tree_std"], subset["abs_error"], alpha=0.75)
    plt.xlabel("RF tree prediction std")
    plt.ylabel("Absolute error |true - predicted|")
    plt.title("Scaffold test: uncertainty vs absolute error")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_uncertainty_deciles(decile_df: pd.DataFrame, out_path: Path):
    subset = decile_df[decile_df["split"] == "test"].copy()

    plt.figure(figsize=(7, 5))
    plt.bar(subset["uncertainty_decile"].astype(str), subset["mean_abs_error"])
    plt.xlabel("Uncertainty decile")
    plt.ylabel("Mean absolute error")
    plt.title("Scaffold test: error by uncertainty decile")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_error_by_pki_bin(error_bin_df: pd.DataFrame, out_path: Path):
    subset = error_bin_df[error_bin_df["split"] == "test"].copy()

    plt.figure(figsize=(7, 5))
    plt.bar(subset["pki_bin"], subset["mean_abs_error"])
    plt.xlabel("Actual pKi bin")
    plt.ylabel("Mean absolute error")
    plt.title("Scaffold test: error by potency range")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 90)
    print("STEP 07: UNCERTAINTY AND ERROR ANALYSIS")
    print("=" * 90)

    print(f"Project root: {ROOT}")

    required = [
        MODELING_CSV,
        SUMMARY_05B,
        BEST_MODEL_PATH,
        SPLIT_DIR / "fxa_04_scaffold_valid.npy",
        SPLIT_DIR / "fxa_04_scaffold_test.npy",
    ]

    assert_required_files_exist(required)

    df = pd.read_csv(MODELING_CSV)

    if SMILES_COL not in df.columns:
        raise ValueError(f"Missing required SMILES column: {SMILES_COL}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET_COL}")

    summary_05b = load_json(SUMMARY_05B)

    best_feature_set = summary_05b["best_scaffold_valid_feature_set"]
    best_model_name = summary_05b["best_scaffold_valid_model"]

    print(f"Best Step 05b feature set: {best_feature_set}")
    print(f"Best Step 05b model: {best_model_name}")

    if best_model_name != "RandomForest":
        raise ValueError(
            "This uncertainty script currently expects the best model to be RandomForest. "
            f"Found: {best_model_name}"
        )

    X = load_feature_matrix(best_feature_set)
    model_object = joblib.load(BEST_MODEL_PATH)

    print(f"Loaded X for {best_feature_set}: {X.shape}")
    print(f"Loaded model: {BEST_MODEL_PATH}")

    valid_idx, test_idx = load_scaffold_valid_test_indices()

    pred_df = make_prediction_dataframe(
        df=df,
        X=X,
        model_object=model_object,
        valid_idx=valid_idx,
        test_idx=test_idx,
    )

    valid_summary = summarize_split(pred_df, "valid")
    test_summary = summarize_split(pred_df, "test")

    metrics_rows = [
        {
            "split": "valid",
            "n": valid_summary["n"],
            "rmse": valid_summary["rmse"],
            "mae": valid_summary["mae"],
            "r2": valid_summary["r2"],
            "spearman_true_pred": valid_summary["spearman_true_pred"],
            "mean_uncertainty_rf_tree_std": valid_summary[
                "mean_uncertainty_rf_tree_std"
            ],
            "spearman_uncertainty_abs_error": valid_summary[
                "spearman_uncertainty_abs_error"
            ],
        },
        {
            "split": "test",
            "n": test_summary["n"],
            "rmse": test_summary["rmse"],
            "mae": test_summary["mae"],
            "r2": test_summary["r2"],
            "spearman_true_pred": test_summary["spearman_true_pred"],
            "mean_uncertainty_rf_tree_std": test_summary[
                "mean_uncertainty_rf_tree_std"
            ],
            "spearman_uncertainty_abs_error": test_summary[
                "spearman_uncertainty_abs_error"
            ],
        },
    ]

    metrics_df = pd.DataFrame(metrics_rows)

    decile_valid = make_uncertainty_deciles(pred_df, "valid")
    decile_test = make_uncertainty_deciles(pred_df, "test")
    decile_df = pd.concat([decile_valid, decile_test], ignore_index=True)

    error_bin_valid = make_error_by_pki_bin(pred_df, "valid")
    error_bin_test = make_error_by_pki_bin(pred_df, "test")
    error_bin_df = pd.concat([error_bin_valid, error_bin_test], ignore_index=True)

    high_error_df = (
        pred_df[pred_df["split"] == "test"]
        .sort_values("abs_error", ascending=False)
        .head(50)
        .copy()
    )

    high_uncertainty_df = (
        pred_df[pred_df["split"] == "test"]
        .sort_values("rf_tree_std", ascending=False)
        .head(50)
        .copy()
    )

    # Save tables
    pred_csv = TABLES_DIR / "fxa_07_scaffold_valid_test_predictions_with_uncertainty.csv"
    high_error_csv = TABLES_DIR / "fxa_07_high_error_molecules.csv"
    high_unc_csv = TABLES_DIR / "fxa_07_high_uncertainty_molecules.csv"
    decile_csv = TABLES_DIR / "fxa_07_uncertainty_decile_analysis.csv"
    error_bin_csv = TABLES_DIR / "fxa_07_error_by_pki_bin.csv"
    metrics_csv = METRICS_DIR / "fxa_07_uncertainty_error_metrics.csv"

    pred_df.to_csv(pred_csv, index=False)
    high_error_df.to_csv(high_error_csv, index=False)
    high_uncertainty_df.to_csv(high_unc_csv, index=False)
    decile_df.to_csv(decile_csv, index=False)
    error_bin_df.to_csv(error_bin_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)

    print("\nSaved Step 07 tables:")
    print(pred_csv)
    print(high_error_csv)
    print(high_unc_csv)
    print(decile_csv)
    print(error_bin_csv)
    print(metrics_csv)

    # Save figures
    fig_pred_unc = FIGURES_DIR / "fxa_07_scaffold_test_pred_vs_actual_uncertainty.png"
    fig_err_unc = FIGURES_DIR / "fxa_07_scaffold_test_abs_error_vs_uncertainty.png"
    fig_deciles = FIGURES_DIR / "fxa_07_uncertainty_deciles_abs_error.png"
    fig_pki_bins = FIGURES_DIR / "fxa_07_error_by_pki_bin.png"

    plot_pred_vs_actual_uncertainty(pred_df, fig_pred_unc)
    plot_abs_error_vs_uncertainty(pred_df, fig_err_unc)
    plot_uncertainty_deciles(decile_df, fig_deciles)
    plot_error_by_pki_bin(error_bin_df, fig_pki_bins)

    print("\nSaved Step 07 figures:")
    print(fig_pred_unc)
    print(fig_err_unc)
    print(fig_deciles)
    print(fig_pki_bins)

    summary = {
        "script": "07_uncertainty_error_analysis.py",
        "model_source": str(BEST_MODEL_PATH),
        "best_feature_set": best_feature_set,
        "best_model_name": best_model_name,
        "uncertainty_method": (
            "RandomForest tree-ensemble prediction standard deviation."
        ),
        "important_method_note": (
            "RF tree std is an ensemble-disagreement signal, not a calibrated "
            "Bayesian uncertainty estimate."
        ),
        "valid_summary": valid_summary,
        "test_summary": test_summary,
        "outputs": {
            "prediction_table": str(pred_csv),
            "high_error_table": str(high_error_csv),
            "high_uncertainty_table": str(high_unc_csv),
            "uncertainty_decile_table": str(decile_csv),
            "error_by_pki_bin_table": str(error_bin_csv),
            "metrics_csv": str(metrics_csv),
            "figures": [
                str(fig_pred_unc),
                str(fig_err_unc),
                str(fig_deciles),
                str(fig_pki_bins),
            ],
        },
        "portfolio_interpretation": (
            "This step identifies where the best classical model is uncertain, "
            "where it makes large errors, and whether RF ensemble disagreement "
            "is useful for prioritizing molecules for review or future active learning."
        ),
    }

    summary_json = METRICS_DIR / "fxa_07_uncertainty_error_summary.json"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)

    print("\nValidation uncertainty/error summary:")
    print(json.dumps(valid_summary, indent=2, default=json_default))

    print("\nTest uncertainty/error summary:")
    print(json.dumps(test_summary, indent=2, default=json_default))

    print("\nSaved Step 07 summary:")
    print(summary_json)

    print("\n" + "=" * 90)
    print("STEP 07 COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()