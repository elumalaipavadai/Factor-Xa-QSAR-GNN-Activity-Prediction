#!/usr/bin/env python
"""
04_make_features_and_splits.py

How to run
----------
python -m py_compile .\scripts\04_make_features_and_splits.py
python .\scripts\04_make_features_and_splits.py *> .\scripts\04.log
Get-Content .\scripts\04.log

Step 04 of the Factor Xa portfolio workflow.

Purpose
-------
Create the fixed modeling dataset, Morgan fingerprint matrix, target vector,
and train/validation/test splits for Factor Xa pKi prediction.

Input
-----
data/processed/fxa_02_curated_structure_level.csv

Required columns
----------------
smiles
target_pKi

Outputs
-------
data/processed/fxa_04_modeling_dataset.csv

data/features/fxa_04_X_morgan_2048.npy
data/features/fxa_04_y_pKi.npy

data/splits/fxa_04_scaffold_train.npy
data/splits/fxa_04_scaffold_valid.npy
data/splits/fxa_04_scaffold_test.npy

data/splits/fxa_04_random_train.npy
data/splits/fxa_04_random_valid.npy
data/splits/fxa_04_random_test.npy

results/metrics/fxa_04_feature_split_summary.json

Why this step matters
---------------------
All later models must use these exact saved splits.
This prevents silent train/test leakage and makes model comparison fair.

Important implementation notes
------------------------------
1. Morgan fingerprints use the modern RDKit fingerprint generator API:
       rdFingerprintGenerator.GetMorganGenerator(...)
   This avoids deprecated AllChem.GetMorganFingerprintAsBitVect warnings.

2. Scaffold split uses non-isomeric Murcko scaffold SMILES:
       isomericSmiles=False
   This groups stereoisomers with the same 2D framework into the same split.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split


# =============================================================================
# Project paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = ROOT / "data" / "processed" / "fxa_02_curated_structure_level.csv"

PROCESSED_DIR = ROOT / "data" / "processed"
FEATURE_DIR = ROOT / "data" / "features"
SPLIT_DIR = ROOT / "data" / "splits"
METRICS_DIR = ROOT / "results" / "metrics"

for directory in [PROCESSED_DIR, FEATURE_DIR, SPLIT_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Settings
# =============================================================================

SMILES_COL = "smiles"
TARGET_COL = "target_pKi"

RANDOM_SEED = 42

FP_RADIUS = 2
FP_BITS = 2048

TRAIN_FRAC = 0.70
VALID_FRAC = 0.15
TEST_FRAC = 0.15

if abs((TRAIN_FRAC + VALID_FRAC + TEST_FRAC) - 1.0) > 1e-8:
    raise ValueError("TRAIN_FRAC + VALID_FRAC + TEST_FRAC must equal 1.0")


# =============================================================================
# Modern RDKit Morgan fingerprint generator
# =============================================================================

_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=FP_RADIUS,
    fpSize=FP_BITS,
)


# =============================================================================
# Helper functions
# =============================================================================

def smiles_to_mol(smiles: str):
    """Convert SMILES to RDKit Mol. Return None if invalid."""
    if not isinstance(smiles, str):
        return None

    smiles = smiles.strip()
    if not smiles:
        return None

    return Chem.MolFromSmiles(smiles)


def canonicalize_smiles(smiles: str):
    """
    Return canonical RDKit SMILES or None if invalid.

    Keep isomericSmiles=True here because the model should preserve
    stereochemical information when it is present.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, isomericSmiles=True)


def mol_to_morgan_fp(mol) -> np.ndarray:
    """
    Convert RDKit Mol to Morgan fingerprint numpy array.

    Uses the modern RDKit fingerprint generator API instead of the deprecated:
        AllChem.GetMorganFingerprintAsBitVect(...)
    """
    fp = _MORGAN_GENERATOR.GetFingerprint(mol)

    arr = np.zeros((FP_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


def get_murcko_scaffold(smiles: str) -> str:
    """
    Get Bemis-Murcko scaffold from SMILES.

    Important:
        isomericSmiles=False is used for the scaffold key only.

    Reason:
        Stereoisomers with the same 2D core should stay in the same split.
        This makes the scaffold split more conservative and reduces leakage.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return "INVALID_MOL"

    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)

    if scaffold_mol is None:
        return "NO_SCAFFOLD"

    scaffold_smiles = Chem.MolToSmiles(scaffold_mol, isomericSmiles=False)

    if scaffold_smiles == "":
        return "NO_SCAFFOLD"

    return scaffold_smiles


def make_scaffold_split(
    smiles_list: List[str],
    train_frac: float = TRAIN_FRAC,
    valid_frac: float = VALID_FRAC,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, List[int]]]:
    """
    Create scaffold train/valid/test split.

    The algorithm:
        1. Groups molecules by Murcko scaffold.
        2. Keeps each scaffold group entirely in one split.
        3. Assigns largest scaffold groups first.

    This avoids putting close analogs with the same core scaffold in both
    train and test.
    """

    scaffold_to_indices: Dict[str, List[int]] = {}

    for idx, smi in enumerate(smiles_list):
        scaffold = get_murcko_scaffold(smi)
        scaffold_to_indices.setdefault(scaffold, []).append(idx)

    scaffold_groups = list(scaffold_to_indices.values())

    # Deterministic tie-breaking.
    rng = random.Random(seed)
    rng.shuffle(scaffold_groups)

    # Assign largest scaffold groups first.
    scaffold_groups = sorted(scaffold_groups, key=len, reverse=True)

    n_total = len(smiles_list)
    n_train_target = int(train_frac * n_total)
    n_valid_target = int(valid_frac * n_total)

    train_idx: List[int] = []
    valid_idx: List[int] = []
    test_idx: List[int] = []

    for group in scaffold_groups:
        if len(train_idx) + len(group) <= n_train_target:
            train_idx.extend(group)
        elif len(valid_idx) + len(group) <= n_valid_target:
            valid_idx.extend(group)
        else:
            test_idx.extend(group)

    train_idx = np.array(sorted(train_idx), dtype=int)
    valid_idx = np.array(sorted(valid_idx), dtype=int)
    test_idx = np.array(sorted(test_idx), dtype=int)

    return train_idx, valid_idx, test_idx, scaffold_to_indices


def assert_no_overlap(name_a: str, idx_a: np.ndarray, name_b: str, idx_b: np.ndarray):
    """Fail loudly if any split indices overlap."""
    overlap = set(idx_a.tolist()).intersection(set(idx_b.tolist()))

    if overlap:
        raise ValueError(
            f"Split leakage detected between {name_a} and {name_b}: "
            f"{len(overlap)} overlapping rows."
        )


def assert_complete_split(name: str, idx_a: np.ndarray, idx_b: np.ndarray, idx_c: np.ndarray, n_total: int):
    """Check that train/valid/test cover every row exactly once."""
    combined = np.concatenate([idx_a, idx_b, idx_c])

    if len(combined) != n_total:
        raise ValueError(
            f"{name} split does not cover all rows. "
            f"Combined rows={len(combined)}, expected={n_total}"
        )

    if len(np.unique(combined)) != n_total:
        raise ValueError(f"{name} split has duplicate indices.")

    if combined.min() < 0 or combined.max() >= n_total:
        raise ValueError(f"{name} split contains out-of-range indices.")


def describe_split(name: str, idx: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Return target distribution summary for one split."""
    y_split = y[idx]

    return {
        "name": name,
        "n": int(len(idx)),
        "target_min": float(np.min(y_split)),
        "target_max": float(np.max(y_split)),
        "target_mean": float(np.mean(y_split)),
        "target_std": float(np.std(y_split)),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 90)
    print("STEP 04: MAKE FEATURES AND SPLITS")
    print("=" * 90)

    print(f"Project root: {ROOT}")
    print(f"Input CSV:    {INPUT_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file does not exist: {INPUT_CSV}")

    df_raw = pd.read_csv(INPUT_CSV)
    n_input_rows = len(df_raw)

    print(f"Loaded dataset shape: {df_raw.shape}")
    print(f"Available columns: {df_raw.columns.tolist()}")

    required_cols = [SMILES_COL, TARGET_COL]
    missing_cols = [col for col in required_cols if col not in df_raw.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Keep useful metadata for downstream interpretation.
    keep_cols = [
        "molecule_chembl_id",
        SMILES_COL,
        "canonical_smiles_rdkit",
        "endpoint",
        TARGET_COL,
        "target_value",
        "n_activity_records",
        "p_endpoint_range",
        "high_replicate_spread",
        "n_molecule_chembl_ids_collapsed",
        "collapsed_molecule_chembl_ids",
        "heavy_atom_count",
        "mol_wt_rdkit",
        "assay_type_set",
        "n_unique_assays",
        "n_unique_documents",
        "structure_dedup_note",
    ]
    keep_cols = [col for col in keep_cols if col in df_raw.columns]

    df = df_raw[keep_cols].copy()

    # Clean target.
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[SMILES_COL, TARGET_COL]).reset_index(drop=True)

    n_after_target_smiles_drop = len(df)

    # Validate and canonicalize SMILES.
    model_rows = []
    model_smiles = []

    for i, smi in enumerate(df[SMILES_COL].tolist()):
        can = canonicalize_smiles(smi)

        if can is None:
            continue

        model_rows.append(i)
        model_smiles.append(can)

    df = df.iloc[model_rows].copy().reset_index(drop=True)
    df["model_smiles"] = model_smiles

    n_after_smiles_validation = len(df)

    # Remove accidental duplicate model_smiles if any remain.
    n_before_dup = len(df)
    df = df.drop_duplicates(subset=["model_smiles"], keep="first").reset_index(drop=True)
    n_after_dup = len(df)
    n_removed_dup = n_before_dup - n_after_dup

    print("\nCleaning summary:")
    print(f"Input rows:                         {n_input_rows}")
    print(f"Rows after SMILES/target drop:       {n_after_target_smiles_drop}")
    print(f"Rows after RDKit SMILES validation:  {n_after_smiles_validation}")
    print(f"Duplicate model_smiles removed:      {n_removed_dup}")
    print(f"Final modeling rows:                 {len(df)}")

    if len(df) < 1500:
        print("WARNING: Modeling set has fewer than 1500 rows. Continue only if expected.")

    print("\nTarget distribution:")
    print(df[TARGET_COL].describe())

    # Save modeling dataset.
    modeling_csv = PROCESSED_DIR / "fxa_04_modeling_dataset.csv"
    df.to_csv(modeling_csv, index=False)

    print("\nSaved modeling dataset:")
    print(modeling_csv)

    # Generate Morgan fingerprints.
    print("\nGenerating Morgan fingerprints with modern RDKit MorganGenerator...")

    mols = [smiles_to_mol(smi) for smi in df["model_smiles"].tolist()]

    if any(mol is None for mol in mols):
        raise ValueError("Unexpected invalid molecules after canonicalization.")

    X = np.array(
        [mol_to_morgan_fp(mol) for mol in mols],
        dtype=np.uint8,
    )

    y = df[TARGET_COL].values.astype(np.float32)

    x_path = FEATURE_DIR / "fxa_04_X_morgan_2048.npy"
    y_path = FEATURE_DIR / "fxa_04_y_pKi.npy"

    np.save(x_path, X)
    np.save(y_path, y)

    print(f"Saved feature matrix: {x_path}")
    print(f"X shape: {X.shape}")
    print(f"Saved target vector:  {y_path}")
    print(f"y shape: {y.shape}")

    # Scaffold split.
    print("\nCreating scaffold train/valid/test split...")
    scaffold_train, scaffold_valid, scaffold_test, scaffold_to_indices = make_scaffold_split(
        df["model_smiles"].tolist(),
        train_frac=TRAIN_FRAC,
        valid_frac=VALID_FRAC,
        seed=RANDOM_SEED,
    )

    assert_no_overlap("scaffold_train", scaffold_train, "scaffold_valid", scaffold_valid)
    assert_no_overlap("scaffold_train", scaffold_train, "scaffold_test", scaffold_test)
    assert_no_overlap("scaffold_valid", scaffold_valid, "scaffold_test", scaffold_test)
    assert_complete_split("scaffold", scaffold_train, scaffold_valid, scaffold_test, len(df))

    np.save(SPLIT_DIR / "fxa_04_scaffold_train.npy", scaffold_train)
    np.save(SPLIT_DIR / "fxa_04_scaffold_valid.npy", scaffold_valid)
    np.save(SPLIT_DIR / "fxa_04_scaffold_test.npy", scaffold_test)

    print(
        "Scaffold split sizes: "
        f"train={len(scaffold_train)}, "
        f"valid={len(scaffold_valid)}, "
        f"test={len(scaffold_test)}"
    )
    print(f"Unique Murcko scaffolds: {len(scaffold_to_indices)}")

    # Random split.
    print("\nCreating random train/valid/test split...")

    all_idx = np.arange(len(df))

    random_train, temp_idx = train_test_split(
        all_idx,
        train_size=TRAIN_FRAC,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    valid_fraction_of_temp = VALID_FRAC / (VALID_FRAC + TEST_FRAC)

    random_valid, random_test = train_test_split(
        temp_idx,
        train_size=valid_fraction_of_temp,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    random_train = np.array(sorted(random_train), dtype=int)
    random_valid = np.array(sorted(random_valid), dtype=int)
    random_test = np.array(sorted(random_test), dtype=int)

    assert_no_overlap("random_train", random_train, "random_valid", random_valid)
    assert_no_overlap("random_train", random_train, "random_test", random_test)
    assert_no_overlap("random_valid", random_valid, "random_test", random_test)
    assert_complete_split("random", random_train, random_valid, random_test, len(df))

    np.save(SPLIT_DIR / "fxa_04_random_train.npy", random_train)
    np.save(SPLIT_DIR / "fxa_04_random_valid.npy", random_valid)
    np.save(SPLIT_DIR / "fxa_04_random_test.npy", random_test)

    print(
        "Random split sizes: "
        f"train={len(random_train)}, "
        f"valid={len(random_valid)}, "
        f"test={len(random_test)}"
    )

    # Summary JSON.
    summary = {
        "script": "04_make_features_and_splits.py",
        "input_csv": str(INPUT_CSV),
        "modeling_csv": str(modeling_csv),

        "n_input_rows": int(n_input_rows),
        "n_after_target_smiles_drop": int(n_after_target_smiles_drop),
        "n_after_smiles_validation": int(n_after_smiles_validation),
        "n_duplicate_model_smiles_removed": int(n_removed_dup),
        "n_modeling_rows": int(len(df)),

        "smiles_column": SMILES_COL,
        "model_smiles_column": "model_smiles",
        "target_column": TARGET_COL,

        "target_summary": {
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
        },

        "fingerprint": {
            "type": "Morgan",
            "api": "rdFingerprintGenerator.GetMorganGenerator",
            "radius": int(FP_RADIUS),
            "n_bits": int(FP_BITS),
            "dtype": str(X.dtype),
            "shape": list(X.shape),
        },

        "fractions_requested": {
            "train": float(TRAIN_FRAC),
            "valid": float(VALID_FRAC),
            "test": float(TEST_FRAC),
        },

        "scaffold_split": {
            "method": "Bemis-Murcko scaffold split",
            "scaffold_smiles_isomeric": False,
            "train": int(len(scaffold_train)),
            "valid": int(len(scaffold_valid)),
            "test": int(len(scaffold_test)),
            "n_unique_murcko_scaffolds": int(len(scaffold_to_indices)),
            "train_target_summary": describe_split("scaffold_train", scaffold_train, y),
            "valid_target_summary": describe_split("scaffold_valid", scaffold_valid, y),
            "test_target_summary": describe_split("scaffold_test", scaffold_test, y),
        },

        "random_split": {
            "method": "random split",
            "train": int(len(random_train)),
            "valid": int(len(random_valid)),
            "test": int(len(random_test)),
            "train_target_summary": describe_split("random_train", random_train, y),
            "valid_target_summary": describe_split("random_valid", random_valid, y),
            "test_target_summary": describe_split("random_test", random_test, y),
        },

        "random_seed": int(RANDOM_SEED),

        "important_note": (
            "Use scaffold split as the headline evaluation. "
            "Use random split only as a comparison. "
            "All later models must reuse these exact saved split index files."
        ),
    }

    summary_path = METRICS_DIR / "fxa_04_feature_split_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved summary JSON:")
    print(summary_path)

    print("\nOutput files created:")
    print(f"  {modeling_csv}")
    print(f"  {x_path}")
    print(f"  {y_path}")
    print(f"  {SPLIT_DIR / 'fxa_04_scaffold_train.npy'}")
    print(f"  {SPLIT_DIR / 'fxa_04_scaffold_valid.npy'}")
    print(f"  {SPLIT_DIR / 'fxa_04_scaffold_test.npy'}")
    print(f"  {SPLIT_DIR / 'fxa_04_random_train.npy'}")
    print(f"  {SPLIT_DIR / 'fxa_04_random_valid.npy'}")
    print(f"  {SPLIT_DIR / 'fxa_04_random_test.npy'}")
    print(f"  {summary_path}")

    print("\n" + "=" * 90)
    print("STEP 04 COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()