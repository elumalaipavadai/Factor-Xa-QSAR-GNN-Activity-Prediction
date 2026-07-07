#!/usr/bin/env python
"""
08_active_learning_uncertainty_vs_random.py

Step 08 of the Factor Xa portfolio workflow.

Purpose
-------
Run a fixed-test active-learning simulation using the best classical model family:

    Morgan_2048 + RandomForest

Comparison arms
---------------
1. Random acquisition
2. Uncertainty acquisition using RandomForest tree prediction standard deviation

Design
------
- Fixed test set: scaffold test split only.
- Acquisition pool: scaffold train split only.
- Scaffold test set is never used for acquisition or training.
- Same initial labeled set is used for both arms within each seed.
- Batched acquisition, not one molecule at a time.
- Multiple seeds are averaged.

Inputs
------
data/processed/fxa_04_modeling_dataset.csv
data/features/fxa_04_X_morgan_2048.npy
data/features/fxa_04_y_pKi.npy
data/splits/fxa_04_scaffold_train.npy
data/splits/fxa_04_scaffold_test.npy
results/metrics/fxa_05b_feature_set_summary.json

Outputs
-------
results/metrics/fxa_08_active_learning_round_metrics.csv
results/metrics/fxa_08_active_learning_curve_summary.csv
results/metrics/fxa_08_active_learning_final_comparison.csv
results/metrics/fxa_08_active_learning_label_efficiency.csv
results/metrics/fxa_08_active_learning_summary.json

results/tables/fxa_08_active_learning_acquisitions.csv

results/figures/fxa_08_al_test_rmse_curve.png
results/figures/fxa_08_al_test_spearman_curve.png
results/figures/fxa_08_al_test_mae_curve.png

How to run
----------
python -m py_compile .\\scripts\\08_active_learning_uncertainty_vs_random.py
python .\\scripts\\08_active_learning_uncertainty_vs_random.py *> .\\scripts\\08.log
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# NumPy compatibility:
# np.trapz was deprecated/removed in newer NumPy 2.x.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

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


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

MODELING_CSV = ROOT / "data" / "processed" / "fxa_04_modeling_dataset.csv"
X_MORGAN_PATH = ROOT / "data" / "features" / "fxa_04_X_morgan_2048.npy"
Y_PATH = ROOT / "data" / "features" / "fxa_04_y_pKi.npy"

SPLIT_DIR = ROOT / "data" / "splits"
TRAIN_SPLIT_PATH = SPLIT_DIR / "fxa_04_scaffold_train.npy"
TEST_SPLIT_PATH = SPLIT_DIR / "fxa_04_scaffold_test.npy"

SUMMARY_05B = ROOT / "results" / "metrics" / "fxa_05b_feature_set_summary.json"

METRICS_DIR = ROOT / "results" / "metrics"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"

for directory in [METRICS_DIR, TABLES_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Settings
# =============================================================================

RANDOM_SEED = 42
SMILES_COL = "model_smiles"
TARGET_COL = "target_pKi"

SEEDS = [11, 22, 33, 44, 55]

INITIAL_LABELED_FRACTION = 0.10
BATCH_SIZE = 50
MAX_ROUNDS = 20

RF_N_ESTIMATORS = 300
RF_MAX_FEATURES = "sqrt"
RF_MIN_SAMPLES_LEAF = 1
RF_N_JOBS = -1

ARMS = ["random", "uncertainty"]


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def assert_required_files_exist(paths: List[Path]):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_y_matches_dataframe(df: pd.DataFrame, y: np.ndarray):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column in dataframe: {TARGET_COL}")

    y_from_csv = pd.to_numeric(df[TARGET_COL], errors="coerce").values.astype(np.float32)

    if len(y_from_csv) != len(y):
        raise ValueError("CSV target length does not match saved y vector length.")

    if not np.allclose(y_from_csv, y, equal_nan=False):
        max_abs_diff = float(np.max(np.abs(y_from_csv - y)))
        raise ValueError(
            f"Saved y vector does not match CSV {TARGET_COL}. "
            f"Max absolute difference: {max_abs_diff}"
        )


def make_rf(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_features=RF_MAX_FEATURES,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=RF_N_JOBS,
        random_state=seed,
    )


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


def rf_tree_uncertainty(model: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    tree_preds = np.vstack([tree.predict(X) for tree in model.estimators_])
    return np.std(tree_preds, axis=0, ddof=1).astype(float)


def make_stratified_initial_set(
    train_idx: np.ndarray,
    y: np.ndarray,
    n_initial: int,
    seed: int,
    n_bins: int = 10,
) -> np.ndarray:
    """
    Create a seed-specific initial labeled set.

    Uses target-stratified sampling across pKi quantiles so the first training
    set is not accidentally too narrow in potency range.
    """

    rng = np.random.default_rng(seed)

    if n_initial >= len(train_idx):
        return np.array(train_idx, dtype=int)

    train_idx = np.array(train_idx, dtype=int)

    try:
        labels = pd.qcut(
            y[train_idx],
            q=min(n_bins, len(np.unique(y[train_idx]))),
            labels=False,
            duplicates="drop",
        )
    except Exception:
        return np.array(rng.choice(train_idx, size=n_initial, replace=False), dtype=int)

    temp = pd.DataFrame({"idx": train_idx, "bin": labels})
    selected: List[int] = []

    grouped = list(temp.groupby("bin", dropna=False))

    for _, group in grouped:
        group_indices = group["idx"].values.astype(int)
        k = int(round(n_initial * len(group_indices) / len(train_idx)))
        k = max(1, k)
        k = min(k, len(group_indices))

        chosen = rng.choice(group_indices, size=k, replace=False)
        selected.extend([int(x) for x in chosen])

    selected = list(dict.fromkeys(selected))

    if len(selected) > n_initial:
        selected = rng.choice(np.array(selected, dtype=int), size=n_initial, replace=False).tolist()

    if len(selected) < n_initial:
        remaining = np.setdiff1d(train_idx, np.array(selected, dtype=int), assume_unique=False)
        n_needed = n_initial - len(selected)
        extra = rng.choice(remaining, size=n_needed, replace=False)
        selected.extend([int(x) for x in extra])

    return np.array(sorted(selected), dtype=int)


def evaluate_on_fixed_test(
    model: RandomForestRegressor,
    X: np.ndarray,
    y: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, float | None]:
    y_true = y[test_idx]
    y_pred = model.predict(X[test_idx])
    return regression_metrics(y_true, y_pred)


def acquire_random(
    pool_idx: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n_acquire = min(batch_size, len(pool_idx))
    selected = rng.choice(pool_idx, size=n_acquire, replace=False)
    selected = np.array(selected, dtype=int)
    remaining = np.setdiff1d(pool_idx, selected, assume_unique=False)
    return selected, remaining


def acquire_uncertainty(
    model: RandomForestRegressor,
    X: np.ndarray,
    pool_idx: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_acquire = min(batch_size, len(pool_idx))

    uncertainty = rf_tree_uncertainty(model, X[pool_idx])

    order = np.argsort(-uncertainty)
    selected_positions = order[:n_acquire]

    selected = pool_idx[selected_positions].astype(int)
    selected_uncertainty = uncertainty[selected_positions].astype(float)

    remaining = np.setdiff1d(pool_idx, selected, assume_unique=False)

    return selected, remaining, selected_uncertainty


def assert_no_overlap(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray):
    overlap = np.intersect1d(a, b)
    if len(overlap) > 0:
        raise ValueError(
            f"Leakage detected: {name_a} overlaps with {name_b}. "
            f"n_overlap={len(overlap)}"
        )


# =============================================================================
# Active learning simulation
# =============================================================================

def run_one_arm(
    arm: str,
    seed: int,
    initial_labeled_idx: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rng = np.random.default_rng(seed + 100000)

    labeled_idx = np.array(initial_labeled_idx, dtype=int)
    pool_idx = np.setdiff1d(train_idx, labeled_idx, assume_unique=False).astype(int)

    assert_no_overlap("initial_labeled_idx", labeled_idx, "test_idx", test_idx)
    assert_no_overlap("initial_pool_idx", pool_idx, "test_idx", test_idx)

    metric_rows: List[Dict[str, object]] = []
    acquisition_rows: List[Dict[str, object]] = []

    print(f"\nSeed={seed} | arm={arm}")
    print(f"Initial labeled={len(labeled_idx)} | initial pool={len(pool_idx)}")

    for round_id in range(MAX_ROUNDS + 1):
        model_seed = seed * 1000 + round_id
        model = make_rf(model_seed)

        model.fit(X[labeled_idx], y[labeled_idx])

        metrics = evaluate_on_fixed_test(
            model=model,
            X=X,
            y=y,
            test_idx=test_idx,
        )

        metric_row = {
            "seed": int(seed),
            "arm": arm,
            "round": int(round_id),
            "n_labeled": int(len(labeled_idx)),
            "n_pool_remaining": int(len(pool_idx)),
            "test_n": int(len(test_idx)),
            **metrics,
        }

        metric_rows.append(metric_row)

        spearman_display = (
            f"{metrics['spearman_r']:.3f}"
            if metrics["spearman_r"] is not None
            else "NA"
        )

        print(
            f"seed={seed:2d} | {arm:11s} | round={round_id:02d} | "
            f"n_labeled={len(labeled_idx):4d} | "
            f"test RMSE={metrics['rmse']:.3f} | "
            f"MAE={metrics['mae']:.3f} | "
            f"Spearman={spearman_display}"
        )

        if round_id >= MAX_ROUNDS or len(pool_idx) == 0:
            break

        if arm == "random":
            selected, pool_idx_new = acquire_random(
                pool_idx=pool_idx,
                batch_size=BATCH_SIZE,
                rng=rng,
            )
            selected_scores = np.full(len(selected), np.nan, dtype=float)

        elif arm == "uncertainty":
            selected, pool_idx_new, selected_scores = acquire_uncertainty(
                model=model,
                X=X,
                pool_idx=pool_idx,
                batch_size=BATCH_SIZE,
            )

        else:
            raise ValueError(f"Unknown acquisition arm: {arm}")

        assert_no_overlap("selected_acquisition_batch", selected, "test_idx", test_idx)

        acquisition_round = round_id + 1

        for rank, row_idx in enumerate(selected, start=1):
            df_row = df.iloc[int(row_idx)]

            score = selected_scores[rank - 1]
            score_value = float(score) if np.isfinite(score) else None

            acquisition_rows.append(
                {
                    "seed": int(seed),
                    "arm": arm,
                    "acquisition_round": int(acquisition_round),
                    "rank_within_batch": int(rank),
                    "row_index": int(row_idx),
                    "molecule_chembl_id": df_row["molecule_chembl_id"]
                    if "molecule_chembl_id" in df.columns
                    else None,
                    "model_smiles": df_row[SMILES_COL],
                    "target_pKi": float(y[int(row_idx)]),
                    "acquisition_score_rf_tree_std": score_value,
                }
            )

        labeled_idx = np.concatenate([labeled_idx, selected]).astype(int)
        labeled_idx = np.array(sorted(set(labeled_idx.tolist())), dtype=int)
        pool_idx = pool_idx_new.astype(int)

    return metric_rows, acquisition_rows


def run_active_learning_simulation(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_metric_rows: List[Dict[str, object]] = []
    all_acquisition_rows: List[Dict[str, object]] = []

    n_initial = int(round(INITIAL_LABELED_FRACTION * len(train_idx)))
    n_initial = max(BATCH_SIZE, n_initial)

    print("Active-learning settings:")
    print(f"Seeds: {SEEDS}")
    print(f"Train pool size: {len(train_idx)}")
    print(f"Fixed scaffold test size: {len(test_idx)}")
    print(f"Initial labeled fraction: {INITIAL_LABELED_FRACTION}")
    print(f"Initial labeled n: {n_initial}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max rounds: {MAX_ROUNDS}")
    print(f"RF estimators per model: {RF_N_ESTIMATORS}")

    for seed in SEEDS:
        set_seed(seed)

        initial_labeled_idx = make_stratified_initial_set(
            train_idx=train_idx,
            y=y,
            n_initial=n_initial,
            seed=seed,
        )

        assert_no_overlap("initial_labeled_idx", initial_labeled_idx, "test_idx", test_idx)

        for arm in ARMS:
            metric_rows, acquisition_rows = run_one_arm(
                arm=arm,
                seed=seed,
                initial_labeled_idx=initial_labeled_idx,
                train_idx=train_idx,
                test_idx=test_idx,
                X=X,
                y=y,
                df=df,
            )

            all_metric_rows.extend(metric_rows)
            all_acquisition_rows.extend(acquisition_rows)

    metrics_df = pd.DataFrame(all_metric_rows)
    acquisitions_df = pd.DataFrame(all_acquisition_rows)

    return metrics_df, acquisitions_df


# =============================================================================
# Summaries
# =============================================================================

def make_curve_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouped = metrics_df.groupby(["arm", "round", "n_labeled"], observed=True)

    for (arm, round_id, n_labeled), group in grouped:
        rows.append(
            {
                "arm": arm,
                "round": int(round_id),
                "n_labeled": int(n_labeled),
                "n_seeds": int(group["seed"].nunique()),
                "rmse_mean": float(group["rmse"].mean()),
                "rmse_std": float(group["rmse"].std(ddof=1)),
                "mae_mean": float(group["mae"].mean()),
                "mae_std": float(group["mae"].std(ddof=1)),
                "r2_mean": float(group["r2"].mean()),
                "r2_std": float(group["r2"].std(ddof=1)),
                "spearman_mean": float(group["spearman_r"].mean()),
                "spearman_std": float(group["spearman_r"].std(ddof=1)),
            }
        )

    return pd.DataFrame(rows).sort_values(["arm", "round"])


def compute_auc_by_seed(metrics_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []

    for (arm, seed), group in metrics_df.groupby(["arm", "seed"], observed=True):
        group = group.sort_values("n_labeled")
        x = group["n_labeled"].values.astype(float)
        y_vals = group[metric].values.astype(float)

        if len(group) >= 2 and x.max() > x.min():
            auc = float(_trapz(y_vals, x))
            normalized_auc = float(auc / (x.max() - x.min()))
        else:
            auc = None
            normalized_auc = None

        rows.append(
            {
                "arm": arm,
                "seed": int(seed),
                f"{metric}_auc": auc,
                f"{metric}_normalized_auc": normalized_auc,
            }
        )

    return pd.DataFrame(rows)


def make_final_comparison(metrics_df: pd.DataFrame) -> pd.DataFrame:
    final_round = int(metrics_df["round"].max())
    final_df = metrics_df[metrics_df["round"] == final_round].copy()

    auc_rmse_df = compute_auc_by_seed(metrics_df, "rmse")
    auc_mae_df = compute_auc_by_seed(metrics_df, "mae")

    merged = (
        final_df.merge(auc_rmse_df, on=["arm", "seed"], how="left")
        .merge(auc_mae_df, on=["arm", "seed"], how="left")
    )

    rows = []

    for arm, group in merged.groupby("arm", observed=True):
        rows.append(
            {
                "arm": arm,
                "final_round": final_round,
                "final_n_labeled_mean": float(group["n_labeled"].mean()),
                "final_rmse_mean": float(group["rmse"].mean()),
                "final_rmse_std": float(group["rmse"].std(ddof=1)),
                "final_mae_mean": float(group["mae"].mean()),
                "final_mae_std": float(group["mae"].std(ddof=1)),
                "final_spearman_mean": float(group["spearman_r"].mean()),
                "final_spearman_std": float(group["spearman_r"].std(ddof=1)),
                "rmse_normalized_auc_mean": float(group["rmse_normalized_auc"].mean()),
                "rmse_normalized_auc_std": float(group["rmse_normalized_auc"].std(ddof=1)),
                "mae_normalized_auc_mean": float(group["mae_normalized_auc"].mean()),
                "mae_normalized_auc_std": float(group["mae_normalized_auc"].std(ddof=1)),
            }
        )

    return pd.DataFrame(rows).sort_values("arm")


def make_label_efficiency_summary(
    curve_summary_df: pd.DataFrame,
    thresholds: List[float],
) -> pd.DataFrame:
    rows = []

    for arm, group in curve_summary_df.groupby("arm", observed=True):
        group = group.sort_values("n_labeled")

        for threshold in thresholds:
            hit = group[group["rmse_mean"] <= threshold]

            if len(hit) == 0:
                rows.append(
                    {
                        "arm": arm,
                        "rmse_threshold": threshold,
                        "first_round_reaching_threshold": None,
                        "n_labeled_reaching_threshold": None,
                    }
                )
            else:
                first = hit.iloc[0]
                rows.append(
                    {
                        "arm": arm,
                        "rmse_threshold": threshold,
                        "first_round_reaching_threshold": int(first["round"]),
                        "n_labeled_reaching_threshold": int(first["n_labeled"]),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================

def plot_metric_curve(
    curve_summary_df: pd.DataFrame,
    metric_mean_col: str,
    metric_std_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    reference_value: float | None = None,
    reference_label: str | None = None,
):
    plt.figure(figsize=(7.5, 5.5))

    for arm, group in curve_summary_df.groupby("arm", observed=True):
        group = group.sort_values("n_labeled")

        x = group["n_labeled"].values.astype(float)
        mean = group[metric_mean_col].values.astype(float)
        std = group[metric_std_col].fillna(0.0).values.astype(float)

        plt.plot(x, mean, marker="o", label=arm)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    if reference_value is not None:
        plt.axhline(reference_value, linestyle="--", label=reference_label)

    plt.xlabel("Number of labeled scaffold-train molecules")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 90)
    print("STEP 08: ACTIVE LEARNING SIMULATION")
    print("=" * 90)

    print(f"Project root: {ROOT}")

    required = [
        MODELING_CSV,
        X_MORGAN_PATH,
        Y_PATH,
        TRAIN_SPLIT_PATH,
        TEST_SPLIT_PATH,
        SUMMARY_05B,
    ]

    assert_required_files_exist(required)

    df = pd.read_csv(MODELING_CSV)
    X = np.load(X_MORGAN_PATH)
    y = np.load(Y_PATH)

    train_idx = np.load(TRAIN_SPLIT_PATH).astype(int)
    test_idx = np.load(TEST_SPLIT_PATH).astype(int)

    print(f"Loaded dataframe: {df.shape}")
    print(f"Loaded Morgan X: {X.shape}, dtype={X.dtype}")
    print(f"Loaded y: {y.shape}, dtype={y.dtype}")
    print(f"Scaffold train pool size: {len(train_idx)}")
    print(f"Fixed scaffold test size: {len(test_idx)}")

    if SMILES_COL not in df.columns:
        raise ValueError(f"Missing SMILES column: {SMILES_COL}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    if len(df) != X.shape[0] or len(df) != len(y):
        raise ValueError("Mismatch among dataframe rows, X rows, and y length.")

    assert_y_matches_dataframe(df, y)
    print("Target consistency check passed.")

    assert_no_overlap("scaffold_train_pool", train_idx, "fixed_scaffold_test", test_idx)
    print("Leakage check passed: scaffold train pool and scaffold test are disjoint.")

    summary_05b = load_json(SUMMARY_05B)

    best_feature_set = summary_05b.get("best_scaffold_valid_feature_set")
    best_model = summary_05b.get("best_scaffold_valid_model")
    baseline_test = summary_05b.get("best_scaffold_test_metrics", {})
    baseline_rmse = baseline_test.get("rmse")
    baseline_mae = baseline_test.get("mae")
    baseline_spearman = baseline_test.get("spearman_r")

    print(f"Step 05b best feature set: {best_feature_set}")
    print(f"Step 05b best model: {best_model}")
    print(f"Step 05b scaffold test RMSE reference: {baseline_rmse}")

    if best_feature_set != "Morgan_2048" or best_model != "RandomForest":
        print(
            "WARNING: Step 08 is designed around Morgan_2048 + RandomForest. "
            "Continuing with Morgan_2048 RF active learning for methodological consistency."
        )

    metrics_df, acquisitions_df = run_active_learning_simulation(
        df=df,
        X=X,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
    )

    if len(acquisitions_df) > 0:
        acquired_idx = acquisitions_df["row_index"].values.astype(int)
        assert_no_overlap("acquired_molecules", acquired_idx, "fixed_scaffold_test", test_idx)
        print("Leakage check passed: acquired molecules have zero overlap with scaffold test.")

    curve_summary_df = make_curve_summary(metrics_df)
    final_comparison_df = make_final_comparison(metrics_df)

    thresholds = []

    if baseline_rmse is not None:
        thresholds.extend(
            [
                float(baseline_rmse) + 0.20,
                float(baseline_rmse) + 0.10,
                float(baseline_rmse) + 0.05,
            ]
        )

    thresholds.extend([1.20, 1.10, 1.00])

    thresholds = sorted(set([round(x, 4) for x in thresholds]), reverse=True)
    label_efficiency_df = make_label_efficiency_summary(curve_summary_df, thresholds)

    # Save outputs
    round_metrics_csv = METRICS_DIR / "fxa_08_active_learning_round_metrics.csv"
    curve_summary_csv = METRICS_DIR / "fxa_08_active_learning_curve_summary.csv"
    final_comparison_csv = METRICS_DIR / "fxa_08_active_learning_final_comparison.csv"
    label_efficiency_csv = METRICS_DIR / "fxa_08_active_learning_label_efficiency.csv"
    acquisitions_csv = TABLES_DIR / "fxa_08_active_learning_acquisitions.csv"

    metrics_df.to_csv(round_metrics_csv, index=False)
    curve_summary_df.to_csv(curve_summary_csv, index=False)
    final_comparison_df.to_csv(final_comparison_csv, index=False)
    label_efficiency_df.to_csv(label_efficiency_csv, index=False)
    acquisitions_df.to_csv(acquisitions_csv, index=False)

    print("\nSaved Step 08 tables:")
    print(round_metrics_csv)
    print(curve_summary_csv)
    print(final_comparison_csv)
    print(label_efficiency_csv)
    print(acquisitions_csv)

    # Figures
    rmse_fig = FIGURES_DIR / "fxa_08_al_test_rmse_curve.png"
    spearman_fig = FIGURES_DIR / "fxa_08_al_test_spearman_curve.png"
    mae_fig = FIGURES_DIR / "fxa_08_al_test_mae_curve.png"

    plot_metric_curve(
        curve_summary_df=curve_summary_df,
        metric_mean_col="rmse_mean",
        metric_std_col="rmse_std",
        ylabel="Fixed scaffold-test RMSE",
        title="Active learning: uncertainty vs random acquisition",
        out_path=rmse_fig,
        reference_value=float(baseline_rmse) if baseline_rmse is not None else None,
        reference_label="Full Step 05b RF baseline",
    )

    plot_metric_curve(
        curve_summary_df=curve_summary_df,
        metric_mean_col="spearman_mean",
        metric_std_col="spearman_std",
        ylabel="Fixed scaffold-test Spearman",
        title="Active learning: uncertainty vs random acquisition",
        out_path=spearman_fig,
        reference_value=float(baseline_spearman) if baseline_spearman is not None else None,
        reference_label="Full Step 05b RF baseline",
    )

    plot_metric_curve(
        curve_summary_df=curve_summary_df,
        metric_mean_col="mae_mean",
        metric_std_col="mae_std",
        ylabel="Fixed scaffold-test MAE",
        title="Active learning: uncertainty vs random acquisition",
        out_path=mae_fig,
        reference_value=float(baseline_mae) if baseline_mae is not None else None,
        reference_label="Full Step 05b RF baseline",
    )

    print("\nSaved Step 08 figures:")
    print(rmse_fig)
    print(spearman_fig)
    print(mae_fig)

    # Final summary
    random_final = final_comparison_df[final_comparison_df["arm"] == "random"]
    uncertainty_final = final_comparison_df[final_comparison_df["arm"] == "uncertainty"]

    delta_summary = None

    if len(random_final) == 1 and len(uncertainty_final) == 1:
        r = random_final.iloc[0]
        u = uncertainty_final.iloc[0]

        delta_summary = {
            "final_rmse_uncertainty_minus_random": float(
                u["final_rmse_mean"] - r["final_rmse_mean"]
            ),
            "final_mae_uncertainty_minus_random": float(
                u["final_mae_mean"] - r["final_mae_mean"]
            ),
            "final_spearman_uncertainty_minus_random": float(
                u["final_spearman_mean"] - r["final_spearman_mean"]
            ),
            "rmse_auc_uncertainty_minus_random": float(
                u["rmse_normalized_auc_mean"] - r["rmse_normalized_auc_mean"]
            ),
            "interpretation_note": (
                "For RMSE/MAE and RMSE AUC, negative values favor uncertainty acquisition. "
                "For Spearman, positive values favor uncertainty acquisition."
            ),
        }

    summary = {
        "script": "08_active_learning_uncertainty_vs_random.py",
        "design": {
            "fixed_test_set": "fxa_04_scaffold_test.npy",
            "acquisition_pool": "fxa_04_scaffold_train.npy",
            "validation_set_used_for_acquisition": False,
            "test_set_used_for_acquisition": False,
            "arms": ARMS,
            "same_initial_labeled_set_per_seed_for_both_arms": True,
            "initial_labeled_fraction": INITIAL_LABELED_FRACTION,
            "batch_size": BATCH_SIZE,
            "max_rounds": MAX_ROUNDS,
            "seeds": SEEDS,
            "anti_leakage_checks": [
                "scaffold train pool disjoint from scaffold test",
                "initial labeled set disjoint from scaffold test",
                "each acquisition batch disjoint from scaffold test",
                "all acquired molecules disjoint from scaffold test",
            ],
        },
        "model": {
            "feature_set": "Morgan_2048",
            "model_family": "RandomForestRegressor",
            "n_estimators": RF_N_ESTIMATORS,
            "max_features": RF_MAX_FEATURES,
            "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
        },
        "step05b_reference": {
            "best_feature_set": best_feature_set,
            "best_model": best_model,
            "full_baseline_scaffold_test_rmse": baseline_rmse,
            "full_baseline_scaffold_test_mae": baseline_mae,
            "full_baseline_scaffold_test_spearman": baseline_spearman,
        },
        "outputs": {
            "round_metrics_csv": str(round_metrics_csv),
            "curve_summary_csv": str(curve_summary_csv),
            "final_comparison_csv": str(final_comparison_csv),
            "label_efficiency_csv": str(label_efficiency_csv),
            "acquisitions_csv": str(acquisitions_csv),
            "figures": [
                str(rmse_fig),
                str(spearman_fig),
                str(mae_fig),
            ],
        },
        "final_comparison": final_comparison_df.to_dict(orient="records"),
        "delta_summary": delta_summary,
        "important_interpretation_note": (
            "This is a retrospective active-learning simulation. "
            "The fixed scaffold test set is never acquired or trained on. "
            "Uncertainty sampling is evaluated only by whether it improves label efficiency "
            "relative to random acquisition."
        ),
    }

    summary_json = METRICS_DIR / "fxa_08_active_learning_summary.json"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)

    print("\nFinal Step 08 comparison:")
    print(final_comparison_df.to_string(index=False))

    if delta_summary is not None:
        print("\nUncertainty minus random:")
        print(json.dumps(delta_summary, indent=2, default=json_default))

    print("\nSaved Step 08 summary:")
    print(summary_json)

    print("\n" + "=" * 90)
    print("STEP 08 COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()