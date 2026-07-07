#!/usr/bin/env python
"""
06_train_gnn_fxa.py

Step 06 of the Factor Xa portfolio workflow.

Purpose
-------
Train a graph neural network baseline for Factor Xa pKi prediction using
the same curated dataset and same Step 04 scaffold/random splits.

Model
-----
PyTorch Geometric GINE-style graph neural network:

SMILES -> PyG graph -> atom feature embeddings -> bond feature embeddings
-> GINEConv layers -> graph pooling -> pKi regression

Important
---------
This script uses the same split files as the classical baselines:

data/splits/fxa_04_scaffold_train.npy
data/splits/fxa_04_scaffold_valid.npy
data/splits/fxa_04_scaffold_test.npy

data/splits/fxa_04_random_train.npy
data/splits/fxa_04_random_valid.npy
data/splits/fxa_04_random_test.npy

The headline result should be scaffold test performance.

Inputs
------
data/processed/fxa_04_modeling_dataset.csv
data/features/fxa_04_y_pKi.npy
data/splits/fxa_04_*.npy

Outputs
-------
results/metrics/fxa_06_gnn_metrics.csv
results/metrics/fxa_06_gnn_metrics.json
results/metrics/fxa_06_gnn_summary.json

results/tables/fxa_06_gnn_predictions.csv

results/figures/fxa_06_scaffold_training_curve.png
results/figures/fxa_06_random_training_curve.png
results/figures/fxa_06_scaffold_test_pred_vs_actual.png

models/fxa_06_scaffold_gine.pt
models/fxa_06_random_gine.pt
models/fxa_06_best_scaffold_gnn.pt

How to run
----------
python -m py_compile .\\scripts\\06_train_gnn_fxa.py
python .\\scripts\\06_train_gnn_fxa.py *> .\\scripts\\06.log
"""

from __future__ import annotations

import json
import math
import random
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool
from torch_geometric.utils import from_smiles


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

MODELING_CSV = ROOT / "data" / "processed" / "fxa_04_modeling_dataset.csv"
Y_PATH = ROOT / "data" / "features" / "fxa_04_y_pKi.npy"
SPLIT_DIR = ROOT / "data" / "splits"

METRICS_DIR = ROOT / "results" / "metrics"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"
MODELS_DIR = ROOT / "models"

for directory in [METRICS_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Settings
# =============================================================================

RANDOM_SEED = 42
SMILES_COL = "model_smiles"
TARGET_COL = "target_pKi"

MODEL_NAME = "GINE"

BATCH_SIZE = 64
MAX_EPOCHS = 150
PATIENCE = 25
MIN_DELTA = 1e-4

HIDDEN_DIM = 128
NUM_GNN_LAYERS = 3
DROPOUT = 0.15

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")


# =============================================================================
# Utility functions
# =============================================================================

def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def assert_required_files_exist(paths: List[Path]):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input files. Run Step 04 first.\n" + "\n".join(missing)
        )


def assert_y_matches_dataframe(df: pd.DataFrame, y: np.ndarray):
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


def load_split(split_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.load(SPLIT_DIR / f"fxa_04_{split_name}_train.npy")
    valid = np.load(SPLIT_DIR / f"fxa_04_{split_name}_valid.npy")
    test = np.load(SPLIT_DIR / f"fxa_04_{split_name}_test.npy")
    return train, valid, test


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


# =============================================================================
# Graph construction
# =============================================================================

def build_graphs_from_dataframe(df: pd.DataFrame, y: np.ndarray):
    """
    Convert SMILES into PyG graph objects.

    from_smiles returns:
        data.x: atom categorical features, shape [num_atoms, 9]
        data.edge_index: directed bond graph
        data.edge_attr: bond categorical features, shape [num_edges, 3]
    """

    if SMILES_COL not in df.columns:
        raise ValueError(f"Missing SMILES column: {SMILES_COL}")

    graphs = []
    invalid_rows = []

    print(f"Converting {len(df)} SMILES to PyG graphs...")

    for i, smi in enumerate(df[SMILES_COL].tolist()):
        try:
            data = from_smiles(smi)
        except Exception as exc:
            invalid_rows.append((i, smi, str(exc)))
            continue

        if data is None or data.x is None or data.x.numel() == 0:
            invalid_rows.append((i, smi, "Empty graph"))
            continue

        data.x = data.x.long()

        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr.long()

        data.y = torch.tensor([float(y[i])], dtype=torch.float32)
        data.row_id = torch.tensor([i], dtype=torch.long)

        graphs.append(data)

    if invalid_rows:
        preview = invalid_rows[:10]
        raise ValueError(
            f"Invalid graph conversion for {len(invalid_rows)} molecules. "
            f"First examples: {preview}"
        )

    if len(graphs) != len(df):
        raise ValueError("Number of graphs does not match dataframe rows.")

    print(f"Graph conversion complete: {len(graphs)} graphs")

    return graphs


def infer_atom_feature_dims(graphs) -> List[int]:
    num_atom_features = int(graphs[0].x.shape[1])
    max_values = np.zeros(num_atom_features, dtype=np.int64)

    for graph in graphs:
        x = graph.x.detach().cpu().numpy()

        if x.shape[1] != num_atom_features:
            raise ValueError("Inconsistent atom feature dimensions among graphs.")

        max_values = np.maximum(max_values, x.max(axis=0))

    dims = (max_values + 2).astype(int).tolist()
    return dims


def infer_bond_feature_dims(graphs) -> List[int]:
    edge_feature_dim = None

    for graph in graphs:
        if graph.edge_attr is not None and graph.edge_attr.dim() == 2:
            edge_feature_dim = int(graph.edge_attr.shape[1])
            break

    if edge_feature_dim is None:
        return []

    max_values = np.zeros(edge_feature_dim, dtype=np.int64)

    for graph in graphs:
        if graph.edge_attr is None or graph.edge_attr.numel() == 0:
            continue

        edge_attr = graph.edge_attr.detach().cpu().numpy()

        if edge_attr.shape[1] != edge_feature_dim:
            raise ValueError("Inconsistent bond feature dimensions among graphs.")

        max_values = np.maximum(max_values, edge_attr.max(axis=0))

    dims = (max_values + 2).astype(int).tolist()
    return dims


# =============================================================================
# Model
# =============================================================================

class CategoricalFeatureEncoder(nn.Module):
    """
    Embeds each categorical feature column and sums the embeddings.

    This is better than directly casting PyG from_smiles categorical values to float.
    """

    def __init__(self, feature_dims: List[int], hidden_dim: int):
        super().__init__()

        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim

        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings=max(dim, 2), embedding_dim=hidden_dim) for dim in feature_dims]
        )

        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.long()

        out = torch.zeros(
            (features.size(0), self.hidden_dim),
            dtype=torch.float32,
            device=features.device,
        )

        for col_idx, emb in enumerate(self.embeddings):
            values = features[:, col_idx]
            values = torch.clamp(values, min=0, max=emb.num_embeddings - 1)
            out = out + emb(values)

        return out


class GINERegressor(nn.Module):
    def __init__(
        self,
        atom_feature_dims: List[int],
        bond_feature_dims: List[int],
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_GNN_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.atom_encoder = CategoricalFeatureEncoder(atom_feature_dims, hidden_dim)
        self.bond_encoder = CategoricalFeatureEncoder(bond_feature_dims, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )

            conv = GINEConv(mlp, train_eps=True)

            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_bonds(self, edge_attr: torch.Tensor, num_edges: int, device) -> torch.Tensor:
        if len(self.bond_encoder.embeddings) == 0:
            return torch.zeros((num_edges, self.hidden_dim), dtype=torch.float32, device=device)

        if edge_attr is None or edge_attr.numel() == 0:
            return torch.zeros((num_edges, self.hidden_dim), dtype=torch.float32, device=device)

        return self.bond_encoder(edge_attr)

    def forward(self, batch):
        x = self.atom_encoder(batch.x)
        edge_index = batch.edge_index

        edge_attr = self.encode_bonds(
            edge_attr=batch.edge_attr if hasattr(batch, "edge_attr") else None,
            num_edges=edge_index.size(1),
            device=x.device,
        )

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_attr)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # Residual connection
            x = x + h

        pooled_mean = global_mean_pool(x, batch.batch)
        pooled_max = global_max_pool(x, batch.batch)
        graph_embedding = torch.cat([pooled_mean, pooled_max], dim=1)

        out = self.head(graph_embedding).view(-1)

        return out


# =============================================================================
# Training and evaluation
# =============================================================================

def make_loader(graphs, indices: np.ndarray, shuffle: bool) -> DataLoader:
    subset = [graphs[int(i)] for i in indices]

    return DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,  # important for Windows stability
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    y_mean: float,
    y_std: float,
) -> float:
    model.train()

    total_loss = 0.0
    total_n = 0

    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.tensor(y_std, dtype=torch.float32, device=device)

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        pred_norm = model(batch)
        target_norm = (batch.y.view(-1).to(device) - y_mean_t) / y_std_t

        loss = F.mse_loss(pred_norm, target_norm)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        n = int(batch.y.numel())
        total_loss += float(loss.item()) * n
        total_n += n

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    y_mean: float,
    y_std: float,
) -> Tuple[Dict[str, float | None], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    all_true = []
    all_pred = []
    all_row_index = []

    for batch in loader:
        batch = batch.to(device)

        pred_norm = model(batch)
        pred_raw = pred_norm.detach().cpu().numpy() * y_std + y_mean

        true_raw = batch.y.view(-1).detach().cpu().numpy()
        row_index = batch.row_id.view(-1).detach().cpu().numpy()

        all_pred.append(pred_raw)
        all_true.append(true_raw)
        all_row_index.append(row_index)

    y_pred = np.concatenate(all_pred).astype(float)
    y_true = np.concatenate(all_true).astype(float)
    row_indices = np.concatenate(all_row_index).astype(int)

    metrics = regression_metrics(y_true, y_pred)

    return metrics, y_true, y_pred, row_indices


def save_checkpoint(path: Path, model, config: Dict[str, object]):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train_for_split(
    split_type: str,
    graphs,
    df: pd.DataFrame,
    y: np.ndarray,
    atom_feature_dims: List[int],
    bond_feature_dims: List[int],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], Path]:
    print("\n" + "-" * 90)
    print(f"Training GNN for split type: {split_type}")
    print("-" * 90)

    train_idx, valid_idx, test_idx = load_split(split_type)

    print(
        f"{split_type} split sizes: "
        f"train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}"
    )

    y_train = y[train_idx].astype(float)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))

    if y_std <= 0:
        raise ValueError("Training target standard deviation is zero.")

    print(f"Train-target normalization: mean={y_mean:.4f}, std={y_std:.4f}")

    train_loader = make_loader(graphs, train_idx, shuffle=True)
    valid_loader = make_loader(graphs, valid_idx, shuffle=False)
    test_loader = make_loader(graphs, test_idx, shuffle=False)

    model = GINERegressor(
        atom_feature_dims=atom_feature_dims,
        bond_feature_dims=bond_feature_dims,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GNN_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
    )

    model_path = MODELS_DIR / f"fxa_06_{split_type}_gine.pt"

    config = {
        "model": MODEL_NAME,
        "split_type": split_type,
        "atom_feature_dims": atom_feature_dims,
        "bond_feature_dims": bond_feature_dims,
        "hidden_dim": HIDDEN_DIM,
        "num_gnn_layers": NUM_GNN_LAYERS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "target_normalization": {
            "y_mean_train_only": y_mean,
            "y_std_train_only": y_std,
        },
    }

    best_valid_rmse = math.inf
    best_epoch = -1
    epochs_without_improvement = 0

    history_rows: List[Dict[str, object]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=DEVICE,
            y_mean=y_mean,
            y_std=y_std,
        )

        train_metrics, _, _, _ = evaluate_model(model, train_loader, DEVICE, y_mean, y_std)
        valid_metrics, _, _, _ = evaluate_model(model, valid_loader, DEVICE, y_mean, y_std)

        scheduler.step(valid_metrics["rmse"])

        current_lr = float(optimizer.param_groups[0]["lr"])

        history_rows.append(
            {
                "split_type": split_type,
                "epoch": epoch,
                "train_loss_norm_mse": train_loss,
                "train_rmse": train_metrics["rmse"],
                "valid_rmse": valid_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "valid_mae": valid_metrics["mae"],
                "learning_rate": current_lr,
            }
        )

        improved = valid_metrics["rmse"] < (best_valid_rmse - MIN_DELTA)

        if improved:
            best_valid_rmse = float(valid_metrics["rmse"])
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(model_path, model, config)
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                f"{split_type:8s} | epoch {epoch:03d} | "
                f"train RMSE={train_metrics['rmse']:.3f} | "
                f"valid RMSE={valid_metrics['rmse']:.3f} | "
                f"best valid RMSE={best_valid_rmse:.3f} | "
                f"lr={current_lr:.2e}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch={best_epoch}, best valid RMSE={best_valid_rmse:.3f}"
            )
            break

    checkpoint = load_checkpoint(model_path, DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_metric_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []

    loaders = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
    }

    for split_label, loader in loaders.items():
        metrics, y_true, y_pred, row_indices = evaluate_model(
            model=model,
            loader=loader,
            device=DEVICE,
            y_mean=y_mean,
            y_std=y_std,
        )

        metric_row = {
            "model": MODEL_NAME,
            "split_type": split_type,
            "split": split_label,
            "n": int(len(y_true)),
            "best_epoch": int(best_epoch),
            **metrics,
        }

        final_metric_rows.append(metric_row)

        print(
            f"{MODEL_NAME:12s} | {split_type:8s} | {split_label:5s} | "
            f"RMSE={metrics['rmse']:.3f} | "
            f"MAE={metrics['mae']:.3f} | "
            f"R2={metrics['r2']:.3f} | "
            f"Spearman={metrics['spearman_r']:.3f}"
            if metrics["spearman_r"] is not None
            else ""
        )

        if split_label in ["valid", "test"]:
            for row_idx, true_val, pred_val in zip(row_indices, y_true, y_pred):
                row = df.iloc[int(row_idx)]

                prediction_rows.append(
                    {
                        "model": MODEL_NAME,
                        "split_type": split_type,
                        "split": split_label,
                        "row_index": int(row_idx),
                        "molecule_chembl_id": row["molecule_chembl_id"]
                        if "molecule_chembl_id" in df.columns
                        else None,
                        "model_smiles": row[SMILES_COL],
                        "y_true_pKi": float(true_val),
                        "y_pred_pKi": float(pred_val),
                        "residual_true_minus_pred": float(true_val - pred_val),
                    }
                )

    return final_metric_rows, prediction_rows, history_rows, model_path


# =============================================================================
# Plotting
# =============================================================================

def plot_training_curve(history_df: pd.DataFrame, split_type: str, out_path: Path):
    plot_df = history_df[history_df["split_type"] == split_type].copy()

    plt.figure(figsize=(7, 5))
    plt.plot(plot_df["epoch"], plot_df["train_rmse"], label="train RMSE")
    plt.plot(plot_df["epoch"], plot_df["valid_rmse"], label="valid RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title(f"GNN training curve: {split_type} split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pred_vs_actual(predictions_df: pd.DataFrame, split_type: str, split_label: str, out_path: Path):
    plot_df = predictions_df[
        (predictions_df["split_type"] == split_type)
        & (predictions_df["split"] == split_label)
    ].copy()

    y_true = plot_df["y_true_pKi"].values
    y_pred = plot_df["y_pred_pKi"].values

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)

    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual pKi")
    plt.ylabel("Predicted pKi")
    plt.title(f"GNN predicted vs actual: {split_type} {split_label}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =============================================================================
# Baseline comparison
# =============================================================================

def load_best_classical_baseline_summary():
    summary_path = METRICS_DIR / "fxa_05b_feature_set_summary.json"

    if not summary_path.exists():
        return None

    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Main
# =============================================================================

def main():
    set_seed(RANDOM_SEED)

    print("=" * 90)
    print("STEP 06: TRAIN GNN BASELINE FOR FACTOR XA pKi")
    print("=" * 90)

    print(f"Project root: {ROOT}")
    print(f"Device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")

    required_files = [
        MODELING_CSV,
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
    y = np.load(Y_PATH)

    print(f"Loaded modeling dataframe: {df.shape}")
    print(f"Loaded y: {y.shape}, dtype={y.dtype}")
    print(f"Target range: min={float(np.min(y)):.3f}, max={float(np.max(y)):.3f}")

    assert_y_matches_dataframe(df, y)
    print("Target consistency check passed.")

    graphs = build_graphs_from_dataframe(df, y)

    atom_feature_dims = infer_atom_feature_dims(graphs)
    bond_feature_dims = infer_bond_feature_dims(graphs)

    print(f"Atom feature dimensions: {atom_feature_dims}")
    print(f"Bond feature dimensions: {bond_feature_dims}")

    all_metric_rows: List[Dict[str, object]] = []
    all_prediction_rows: List[Dict[str, object]] = []
    all_history_rows: List[Dict[str, object]] = []
    model_paths: Dict[str, str] = {}

    for split_type in ["scaffold", "random"]:
        metric_rows, prediction_rows, history_rows, model_path = train_for_split(
            split_type=split_type,
            graphs=graphs,
            df=df,
            y=y,
            atom_feature_dims=atom_feature_dims,
            bond_feature_dims=bond_feature_dims,
        )

        all_metric_rows.extend(metric_rows)
        all_prediction_rows.extend(prediction_rows)
        all_history_rows.extend(history_rows)
        model_paths[split_type] = str(model_path)

    metrics_df = pd.DataFrame(all_metric_rows)
    predictions_df = pd.DataFrame(all_prediction_rows)
    history_df = pd.DataFrame(all_history_rows)

    metrics_csv = METRICS_DIR / "fxa_06_gnn_metrics.csv"
    metrics_json = METRICS_DIR / "fxa_06_gnn_metrics.json"
    summary_json = METRICS_DIR / "fxa_06_gnn_summary.json"
    predictions_csv = TABLES_DIR / "fxa_06_gnn_predictions.csv"
    history_csv = METRICS_DIR / "fxa_06_gnn_training_history.csv"

    metrics_df.to_csv(metrics_csv, index=False)
    predictions_df.to_csv(predictions_csv, index=False)
    history_df.to_csv(history_csv, index=False)

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(all_metric_rows, f, indent=2, default=json_default)

    print("\nSaved GNN metrics and predictions:")
    print(metrics_csv)
    print(metrics_json)
    print(predictions_csv)
    print(history_csv)

    scaffold_model_path = Path(model_paths["scaffold"])
    best_scaffold_copy = MODELS_DIR / "fxa_06_best_scaffold_gnn.pt"
    shutil.copyfile(scaffold_model_path, best_scaffold_copy)

    scaffold_test_row = metrics_df[
        (metrics_df["split_type"] == "scaffold")
        & (metrics_df["split"] == "test")
    ].iloc[0]

    scaffold_valid_row = metrics_df[
        (metrics_df["split_type"] == "scaffold")
        & (metrics_df["split"] == "valid")
    ].iloc[0]

    print("\nGNN scaffold validation metrics:")
    print(scaffold_valid_row.to_string())

    print("\nGNN scaffold test metrics:")
    print(scaffold_test_row.to_string())

    # Figures
    scaffold_curve_fig = FIGURES_DIR / "fxa_06_scaffold_training_curve.png"
    random_curve_fig = FIGURES_DIR / "fxa_06_random_training_curve.png"
    scaffold_pred_fig = FIGURES_DIR / "fxa_06_scaffold_test_pred_vs_actual.png"

    plot_training_curve(history_df, "scaffold", scaffold_curve_fig)
    plot_training_curve(history_df, "random", random_curve_fig)
    plot_pred_vs_actual(predictions_df, "scaffold", "test", scaffold_pred_fig)

    print("\nSaved GNN figures:")
    print(scaffold_curve_fig)
    print(random_curve_fig)
    print(scaffold_pred_fig)

    baseline_summary = load_best_classical_baseline_summary()

    baseline_comparison = None

    if baseline_summary is not None:
        baseline_test = baseline_summary.get("best_scaffold_test_metrics", {})

        baseline_comparison = {
            "classical_best_feature_set": baseline_summary.get(
                "best_scaffold_valid_feature_set"
            ),
            "classical_best_model": baseline_summary.get("best_scaffold_valid_model"),
            "classical_scaffold_test_rmse": baseline_test.get("rmse"),
            "classical_scaffold_test_mae": baseline_test.get("mae"),
            "classical_scaffold_test_r2": baseline_test.get("r2"),
            "classical_scaffold_test_spearman": baseline_test.get("spearman_r"),
            "gnn_scaffold_test_rmse": float(scaffold_test_row["rmse"]),
            "gnn_scaffold_test_mae": float(scaffold_test_row["mae"]),
            "gnn_scaffold_test_r2": float(scaffold_test_row["r2"]),
            "gnn_scaffold_test_spearman": float(scaffold_test_row["spearman_r"])
            if pd.notna(scaffold_test_row["spearman_r"])
            else None,
        }

        if baseline_test.get("rmse") is not None:
            baseline_comparison["gnn_minus_classical_rmse"] = float(
                scaffold_test_row["rmse"] - baseline_test["rmse"]
            )

    summary = {
        "script": "06_train_gnn_fxa.py",
        "model": MODEL_NAME,
        "device": DEVICE,
        "input_modeling_csv": str(MODELING_CSV),
        "input_y": str(Y_PATH),
        "smiles_column": SMILES_COL,
        "target_column": TARGET_COL,
        "n_molecules": int(len(df)),
        "graph_feature_info": {
            "atom_feature_dims": atom_feature_dims,
            "bond_feature_dims": bond_feature_dims,
            "atom_feature_count": int(len(atom_feature_dims)),
            "bond_feature_count": int(len(bond_feature_dims)),
            "note": (
                "PyG from_smiles categorical atom/bond features were embedded "
                "rather than directly treated as continuous float features."
            ),
        },
        "training_config": {
            "random_seed": RANDOM_SEED,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "hidden_dim": HIDDEN_DIM,
            "num_gnn_layers": NUM_GNN_LAYERS,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "target_normalization": "pKi z-score using train split mean/std only",
        },
        "headline_split": "scaffold",
        "comparison_split": "random",
        "selection_rule": "Best model checkpoint selected by scaffold validation RMSE.",
        "scaffold_valid_metrics": scaffold_valid_row.to_dict(),
        "scaffold_test_metrics": scaffold_test_row.to_dict(),
        "model_paths": model_paths,
        "best_scaffold_copy": str(best_scaffold_copy),
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(metrics_json),
        "predictions_csv": str(predictions_csv),
        "history_csv": str(history_csv),
        "figures": [
            str(scaffold_curve_fig),
            str(random_curve_fig),
            str(scaffold_pred_fig),
        ],
        "baseline_comparison": baseline_comparison,
        "important_note": (
            "Use scaffold test performance as the headline GNN result. "
            "Random split performance is comparison only and usually more optimistic."
        ),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)

    print("\nSaved GNN summary:")
    print(summary_json)

    print("\n" + "=" * 90)
    print("STEP 06 COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
