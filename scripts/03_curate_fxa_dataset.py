#!/usr/bin/env python
"""
03_curate_fxa_dataset.py

Step 02 for the FXa GNN portfolio.

Purpose:
Convert the P0 selected ChEMBL activity-record table into a clean compound/structure-level
modeling dataset.

Default input:
    data/processed/fxa_p0_selected_ki_binding_only.csv

Default output:
    data/processed/fxa_02_curated_structure_level.csv

What this script does:
1. Reads the P0 selected endpoint dataset.
2. Re-checks basic filters:
   - standard_units == nM
   - standard_relation == "="
   - positive numeric standard_value
   - non-missing SMILES
3. Infers endpoint:
   - Ki   -> pKi
   - IC50 -> pIC50
4. Recomputes pEndpoint from nM:
   pEndpoint = 9 - log10(standard_value_nM)
5. Cross-checks recomputed pEndpoint against ChEMBL pchembl_value where available.
6. Validates and canonicalizes SMILES with RDKit.
7. Keeps largest fragment only.
8. Aggregates repeated ChEMBL activity records by molecule_chembl_id.
9. Flags high replicate disagreement:
   - replicate spread = max(pEndpoint) - min(pEndpoint)
   - default threshold = 1.0 log unit
10. Checks for residual structure-level duplicates:
   - different molecule_chembl_id values can collapse to the same RDKit canonical SMILES
11. Collapses duplicate RDKit canonical SMILES groups by default to prevent train/test leakage.
12. Writes audit files and JSON metadata.

Important standardization note:
This script performs light standardization:
    largest-fragment selection + RDKit sanitization + canonical isomeric SMILES

It does NOT perform:
    charge neutralization / uncharging
    tautomer canonicalization

This is intentional for speed and transparency. More aggressive standardization can be added
later if needed, but should be documented because it can change chemical identity assumptions.

Why this matters:
ChEMBL rows are activity records, not unique compounds. Also, ChEMBL molecule_chembl_id values
are not always guaranteed to be unique at the structure level after salt removal / fragment
selection / canonicalization. This script therefore reports both molecule-level and
canonical-SMILES-level counts.

Run from project root:
    conda activate fxa_portfolio_clean
    python scripts/03_curate_fxa_dataset.py

Optional:
    python scripts/03_curate_fxa_dataset.py --replicate-spread-threshold 1.0
    python scripts/03_curate_fxa_dataset.py --keep-high-spread
    python scripts/03_curate_fxa_dataset.py --no-collapse-duplicate-smiles
    python scripts/03_curate_fxa_dataset.py --strict-pchembl-check
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
    LARGEST_FRAGMENT_CHOOSER = rdMolStandardize.LargestFragmentChooser()
except Exception:
    LARGEST_FRAGMENT_CHOOSER = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curate P0-selected Factor Xa ChEMBL activity records to structure-level dataset."
    )

    parser.add_argument(
        "--input",
        default="data/processed/fxa_p0_selected_ki_binding_only.csv",
        help="Input P0 selected dataset CSV. Default: data/processed/fxa_p0_selected_ki_binding_only.csv",
    )

    parser.add_argument(
        "--outdir",
        default="data/processed",
        help="Output directory. Default: data/processed",
    )

    parser.add_argument(
        "--replicate-spread-threshold",
        type=float,
        default=1.0,
        help=(
            "Maximum allowed replicate spread in pEndpoint log units. "
            "Groups with spread above this threshold are dropped unless --keep-high-spread is set. "
            "Default: 1.0"
        ),
    )

    parser.add_argument(
        "--keep-high-spread",
        action="store_true",
        help=(
            "Keep molecule-level groups with replicate spread above threshold. "
            "Default behavior is to drop them from the final modeling dataset."
        ),
    )

    parser.add_argument(
        "--group-by",
        choices=["molecule_chembl_id", "canonical_smiles_rdkit"],
        default="molecule_chembl_id",
        help=(
            "First aggregation key. Default: molecule_chembl_id. "
            "Residual canonical-SMILES duplicates are checked after this step."
        ),
    )

    parser.add_argument(
        "--no-collapse-duplicate-smiles",
        action="store_true",
        help=(
            "Do not collapse residual duplicate canonical_smiles_rdkit groups after molecule_chembl_id aggregation. "
            "Default is to collapse them to prevent train/test leakage."
        ),
    )

    parser.add_argument(
        "--duplicate-smiles-spread-threshold",
        type=float,
        default=1.0,
        help=(
            "Maximum allowed spread among rows sharing the same canonical_smiles_rdkit after molecule-level aggregation. "
            "Default: 1.0 log unit."
        ),
    )

    parser.add_argument(
        "--min-pendpoint",
        type=float,
        default=None,
        help="Optional minimum pEndpoint filter after calculation. Default: no minimum.",
    )

    parser.add_argument(
        "--max-pendpoint",
        type=float,
        default=None,
        help="Optional maximum pEndpoint filter after calculation. Default: no maximum.",
    )

    parser.add_argument(
        "--pchembl-tolerance",
        type=float,
        default=0.05,
        help=(
            "Tolerance for comparing recomputed pEndpoint to ChEMBL pchembl_value on rows where pchembl_value exists. "
            "Default: 0.05 log units."
        ),
    )

    parser.add_argument(
        "--strict-pchembl-check",
        action="store_true",
        help=(
            "Fail the script if recomputed pEndpoint differs from pchembl_value by more than --pchembl-tolerance. "
            "Default: warn only."
        ),
    )

    return parser.parse_args()


def ensure_output_dir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def require_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def normalize_relation(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace("'", "").replace('"', "")


def infer_endpoint(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Infer selected endpoint from input table.

    Returns:
        standard_type: Ki or IC50
        endpoint_col: pKi or pIC50
        target_col: target_pKi or target_pIC50
    """
    standard_type = None

    if "activity_type_cleaned" in df.columns:
        vals = [str(x).strip() for x in df["activity_type_cleaned"].dropna().unique()]
        vals = [x for x in vals if x and x.lower() != "nan"]
        if len(vals) == 1:
            standard_type = vals[0]

    if standard_type is None and "standard_type" in df.columns:
        vals = [str(x).strip() for x in df["standard_type"].dropna().unique()]
        vals = [x for x in vals if x and x.lower() != "nan"]
        if len(vals) == 1:
            standard_type = vals[0]

    if standard_type is None:
        raise ValueError(
            "Could not infer a single endpoint type. "
            "The input should be a single-endpoint P0 selected dataset."
        )

    standard_type_upper = standard_type.upper()

    if standard_type_upper == "KI":
        return "Ki", "pKi", "target_pKi"

    if standard_type_upper == "IC50":
        return "IC50", "pIC50", "target_pIC50"

    raise ValueError(f"Unsupported endpoint type: {standard_type}. Expected Ki or IC50.")


def compute_p_endpoint_from_nm(value_nm: float) -> float:
    return 9.0 - math.log10(float(value_nm))


def choose_largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    if LARGEST_FRAGMENT_CHOOSER is not None:
        return LARGEST_FRAGMENT_CHOOSER.choose(mol)

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not fragments:
        return mol

    return max(fragments, key=lambda m: m.GetNumHeavyAtoms())


def canonicalize_smiles(smiles: Any) -> Tuple[Optional[str], Optional[int], Optional[float], str]:
    """
    Light standardization:
    - parse SMILES
    - keep largest fragment
    - sanitize
    - return RDKit canonical isomeric SMILES

    No uncharging or tautomer canonicalization is performed.
    """
    if pd.isna(smiles):
        return None, None, None, "missing_smiles"

    smiles_str = str(smiles).strip()
    if smiles_str == "" or smiles_str.lower() == "nan":
        return None, None, None, "missing_smiles"

    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return None, None, None, "rdkit_parse_failed"

        mol = choose_largest_fragment(mol)
        Chem.SanitizeMol(mol)

        heavy_atoms = int(mol.GetNumHeavyAtoms())
        if heavy_atoms <= 0:
            return None, None, None, "zero_heavy_atoms"

        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        mw = float(Descriptors.MolWt(mol))

        return canonical, heavy_atoms, mw, ""

    except Exception as exc:
        return None, None, None, f"rdkit_exception: {type(exc).__name__}"


def most_common_nonmissing(values: List[Any]) -> Any:
    cleaned = []
    for value in values:
        if pd.isna(value):
            continue
        value_str = str(value).strip()
        if value_str == "" or value_str.lower() == "nan":
            continue
        cleaned.append(value_str)

    if not cleaned:
        return pd.NA

    return Counter(cleaned).most_common(1)[0][0]


def joined_unique(values: List[Any], max_items: int = 20) -> str:
    cleaned = []
    for value in values:
        if pd.isna(value):
            continue
        value_str = str(value).strip()
        if value_str == "" or value_str.lower() == "nan":
            continue
        cleaned.append(value_str)

    unique_values = sorted(set(cleaned))

    if len(unique_values) > max_items:
        shown = unique_values[:max_items]
        return ";".join(shown) + f";...(+{len(unique_values) - max_items} more)"

    return ";".join(unique_values)


def add_basic_filter_columns(df: pd.DataFrame, endpoint_col: str) -> pd.DataFrame:
    df = df.copy()

    df["standard_value_num"] = pd.to_numeric(
        df.get("standard_value_num", df.get("standard_value")),
        errors="coerce",
    )

    df["standard_units_norm"] = df["standard_units"].astype(str).str.strip().str.lower()
    df["standard_relation_norm"] = df["standard_relation"].map(normalize_relation)

    df[endpoint_col + "_calc"] = df["standard_value_num"].apply(
        lambda x: compute_p_endpoint_from_nm(x) if pd.notna(x) and float(x) > 0 else pd.NA
    )

    return df


def pchembl_cross_check(
    df: pd.DataFrame,
    endpoint_col: str,
    tolerance: float,
    strict: bool,
) -> Dict[str, Any]:
    """
    Compare recomputed pEndpoint against ChEMBL pchembl_value.

    This catches unit mistakes or unexpected contamination.
    """
    result: Dict[str, Any] = {
        "pchembl_rows_available": 0,
        "pchembl_rows_checked": 0,
        "pchembl_max_abs_diff": None,
        "pchembl_mean_abs_diff": None,
        "pchembl_rows_above_tolerance": 0,
        "pchembl_tolerance": tolerance,
        "pchembl_check_passed": True,
        "pchembl_check_note": "",
    }

    if "pchembl_value" not in df.columns or endpoint_col not in df.columns:
        result["pchembl_check_note"] = "pchembl_value or endpoint column missing; check skipped."
        return result

    tmp = df.copy()
    tmp["pchembl_value_num"] = pd.to_numeric(tmp["pchembl_value"], errors="coerce")
    tmp = tmp[tmp["pchembl_value_num"].notna()].copy()

    result["pchembl_rows_available"] = int(len(tmp))

    if tmp.empty:
        result["pchembl_check_note"] = "No numeric pchembl_value rows available; check skipped."
        return result

    tmp["abs_diff"] = (
        pd.to_numeric(tmp[endpoint_col], errors="coerce") - tmp["pchembl_value_num"]
    ).abs()

    max_diff = float(tmp["abs_diff"].max())
    mean_diff = float(tmp["abs_diff"].mean())
    n_bad = int((tmp["abs_diff"] > tolerance).sum())

    result.update(
        {
            "pchembl_rows_checked": int(len(tmp)),
            "pchembl_max_abs_diff": max_diff,
            "pchembl_mean_abs_diff": mean_diff,
            "pchembl_rows_above_tolerance": n_bad,
            "pchembl_check_passed": n_bad == 0,
        }
    )

    if n_bad == 0:
        result["pchembl_check_note"] = (
            f"PASS: recomputed {endpoint_col} agrees with pchembl_value within "
            f"{tolerance} log units where pchembl_value exists."
        )
    else:
        result["pchembl_check_note"] = (
            f"WARNING: {n_bad} rows differ from pchembl_value by more than "
            f"{tolerance} log units. Review unit handling and ChEMBL pchembl rules."
        )

    if strict and n_bad > 0:
        raise AssertionError(result["pchembl_check_note"])

    return result


def curate_records(
    df: pd.DataFrame,
    endpoint_col: str,
    min_pendpoint: Optional[float],
    max_pendpoint: Optional[float],
    pchembl_tolerance: float,
    strict_pchembl_check: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Record-level curation before aggregation.
    """
    required = [
        "activity_id",
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_value_num",
        "standard_units",
        "pchembl_value",
        "assay_type",
        "assay_chembl_id",
        "assay_description",
        "target_chembl_id",
        "target_pref_name",
        "document_chembl_id",
    ]

    df = require_columns(df, required)
    df = add_basic_filter_columns(df, endpoint_col=endpoint_col)

    counts: Dict[str, Any] = {}
    counts["input_activity_records"] = int(len(df))
    counts["input_unique_molecule_chembl_ids"] = int(df["molecule_chembl_id"].nunique())
    counts["input_unique_smiles_original"] = int(df["canonical_smiles"].nunique())

    valid_mask = pd.Series(True, index=df.index)

    valid_mask &= df["standard_units_norm"].eq("nm")
    counts["after_units_nM_records"] = int(valid_mask.sum())

    valid_mask &= df["standard_relation_norm"].eq("=")
    counts["after_relation_equal_records"] = int(valid_mask.sum())

    valid_mask &= df["standard_value_num"].notna() & (df["standard_value_num"] > 0)
    counts["after_positive_numeric_value_records"] = int(valid_mask.sum())

    valid_mask &= df["canonical_smiles"].notna()
    valid_mask &= df["canonical_smiles"].astype(str).str.strip().ne("")
    valid_mask &= df["canonical_smiles"].astype(str).str.lower().ne("nan")
    counts["after_nonmissing_smiles_records"] = int(valid_mask.sum())

    valid_records = df[valid_mask].copy()
    invalid_basic = df[~valid_mask].copy()
    invalid_basic["invalid_reason"] = "failed_basic_filter"

    # Canonicalization.
    canonical_results = valid_records["canonical_smiles"].apply(canonicalize_smiles)
    valid_records["canonical_smiles_rdkit"] = [x[0] for x in canonical_results]
    valid_records["heavy_atom_count"] = [x[1] for x in canonical_results]
    valid_records["mol_wt_rdkit"] = [x[2] for x in canonical_results]
    valid_records["rdkit_invalid_reason"] = [x[3] for x in canonical_results]

    invalid_rdkit = valid_records[valid_records["canonical_smiles_rdkit"].isna()].copy()
    invalid_rdkit["invalid_reason"] = invalid_rdkit["rdkit_invalid_reason"]

    valid_records = valid_records[valid_records["canonical_smiles_rdkit"].notna()].copy()
    counts["after_rdkit_valid_records"] = int(len(valid_records))
    counts["after_rdkit_valid_unique_molecule_chembl_ids"] = int(valid_records["molecule_chembl_id"].nunique())
    counts["after_rdkit_valid_unique_canonical_smiles"] = int(valid_records["canonical_smiles_rdkit"].nunique())

    # Use recomputed endpoint as the canonical label.
    valid_records[endpoint_col] = pd.to_numeric(valid_records[endpoint_col + "_calc"], errors="coerce")
    valid_records = valid_records[valid_records[endpoint_col].notna()].copy()
    counts["after_endpoint_calculated_records"] = int(len(valid_records))

    pchembl_check = pchembl_cross_check(
        valid_records,
        endpoint_col=endpoint_col,
        tolerance=pchembl_tolerance,
        strict=strict_pchembl_check,
    )

    if min_pendpoint is not None:
        valid_records = valid_records[valid_records[endpoint_col] >= min_pendpoint].copy()
        counts[f"after_min_{endpoint_col}_{min_pendpoint}_records"] = int(len(valid_records))

    if max_pendpoint is not None:
        valid_records = valid_records[valid_records[endpoint_col] <= max_pendpoint].copy()
        counts[f"after_max_{endpoint_col}_{max_pendpoint}_records"] = int(len(valid_records))

    invalid_records = pd.concat([invalid_basic, invalid_rdkit], ignore_index=True, sort=False)
    counts["invalid_or_dropped_record_level_rows"] = int(len(invalid_records))

    return valid_records, invalid_records, counts, pchembl_check


def make_replicate_summary(
    valid_records: pd.DataFrame,
    group_by: str,
    endpoint_col: str,
    target_col: str,
    spread_threshold: float,
) -> pd.DataFrame:
    """
    Aggregate records to one row per group.

    Default group is molecule_chembl_id. Median pEndpoint is used as target.
    """
    rows: List[Dict[str, Any]] = []

    for group_value, group in valid_records.groupby(group_by, dropna=False):
        endpoint_values = pd.to_numeric(group[endpoint_col], errors="coerce").dropna()

        if endpoint_values.empty:
            continue

        standard_values_nm = pd.to_numeric(group["standard_value_num"], errors="coerce").dropna()

        p_min = float(endpoint_values.min())
        p_max = float(endpoint_values.max())
        p_range = float(p_max - p_min)

        row = {
            group_by: group_value,
            "molecule_chembl_id": most_common_nonmissing(group["molecule_chembl_id"].tolist()),
            "canonical_smiles": most_common_nonmissing(group["canonical_smiles"].tolist()),
            "canonical_smiles_rdkit": most_common_nonmissing(group["canonical_smiles_rdkit"].tolist()),
            "smiles": most_common_nonmissing(group["canonical_smiles_rdkit"].tolist()),
            "endpoint": endpoint_col,
            target_col: float(endpoint_values.median()),
            "target_value": float(endpoint_values.median()),
            "p_endpoint_mean": float(endpoint_values.mean()),
            "p_endpoint_std": float(endpoint_values.std(ddof=1)) if len(endpoint_values) > 1 else 0.0,
            "p_endpoint_min": p_min,
            "p_endpoint_max": p_max,
            "p_endpoint_range": p_range,
            "high_replicate_spread": bool(p_range > spread_threshold),
            "n_activity_records": int(len(group)),
            "n_unique_assays": int(group["assay_chembl_id"].nunique()),
            "n_unique_documents": int(group["document_chembl_id"].nunique()),
            "n_unique_original_smiles": int(group["canonical_smiles"].nunique()),
            "n_unique_rdkit_smiles": int(group["canonical_smiles_rdkit"].nunique()),
            "median_standard_value_nM": float(standard_values_nm.median()) if not standard_values_nm.empty else pd.NA,
            "min_standard_value_nM": float(standard_values_nm.min()) if not standard_values_nm.empty else pd.NA,
            "max_standard_value_nM": float(standard_values_nm.max()) if not standard_values_nm.empty else pd.NA,
            "heavy_atom_count": int(pd.to_numeric(group["heavy_atom_count"], errors="coerce").median()),
            "mol_wt_rdkit": float(pd.to_numeric(group["mol_wt_rdkit"], errors="coerce").median()),
            "assay_type_set": joined_unique(group["assay_type"].tolist()),
            "assay_chembl_ids": joined_unique(group["assay_chembl_id"].tolist()),
            "document_chembl_ids": joined_unique(group["document_chembl_id"].tolist()),
            "activity_ids": joined_unique(group["activity_id"].tolist(), max_items=50),
        }

        rows.append(row)

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary = summary.sort_values(
            by=["target_value", "n_activity_records"],
            ascending=[False, False],
        ).reset_index(drop=True)

    return summary


def collapse_duplicate_smiles(
    molecule_level_df: pd.DataFrame,
    endpoint_col: str,
    target_col: str,
    duplicate_spread_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Collapse residual duplicate canonical_smiles_rdkit rows after molecule-level aggregation.

    Why:
    Different ChEMBL IDs can collapse to the same structure after salt removal / largest-fragment
    selection / canonicalization. Keeping duplicate structures risks train/test leakage.

    Rule:
    - group by canonical_smiles_rdkit
    - median target_value becomes final label
    - flag groups where duplicate-SMILES target spread > threshold
    - high-spread duplicate-SMILES groups are dropped from the final structure-level dataset
    """
    if molecule_level_df.empty:
        return molecule_level_df.copy(), pd.DataFrame(), {}

    duplicate_mask = molecule_level_df.duplicated("canonical_smiles_rdkit", keep=False)
    duplicate_smiles_rows = molecule_level_df[duplicate_mask].copy()

    rows: List[Dict[str, Any]] = []

    for smiles, group in molecule_level_df.groupby("canonical_smiles_rdkit", dropna=False):
        target_values = pd.to_numeric(group["target_value"], errors="coerce").dropna()

        if target_values.empty:
            continue

        target_min = float(target_values.min())
        target_max = float(target_values.max())
        target_range = float(target_max - target_min)

        # Pick representative row from the group with the most underlying activity records.
        rep = group.sort_values(
            by=["n_activity_records", "target_value"],
            ascending=[False, False],
        ).iloc[0].to_dict()

        rep[target_col] = float(target_values.median())
        rep["target_value"] = float(target_values.median())
        rep["structure_level_target_mean"] = float(target_values.mean())
        rep["structure_level_target_std"] = float(target_values.std(ddof=1)) if len(target_values) > 1 else 0.0
        rep["structure_level_target_min"] = target_min
        rep["structure_level_target_max"] = target_max
        rep["structure_level_target_range"] = target_range
        rep["high_duplicate_smiles_spread"] = bool(target_range > duplicate_spread_threshold)
        rep["n_molecule_chembl_ids_collapsed"] = int(group["molecule_chembl_id"].nunique())
        rep["collapsed_molecule_chembl_ids"] = joined_unique(group["molecule_chembl_id"].tolist(), max_items=50)
        rep["collapsed_activity_record_count"] = int(pd.to_numeric(group["n_activity_records"], errors="coerce").sum())
        rep["structure_dedup_note"] = (
            "single_structure" if len(group) == 1 else "collapsed_duplicate_canonical_smiles"
        )

        rows.append(rep)

    collapsed_all = pd.DataFrame(rows)

    if collapsed_all.empty:
        final = collapsed_all
        high_spread_duplicate_groups = pd.DataFrame()
    else:
        high_spread_duplicate_groups = collapsed_all[collapsed_all["high_duplicate_smiles_spread"]].copy()
        final = collapsed_all[~collapsed_all["high_duplicate_smiles_spread"]].copy()
        final = final.sort_values(by=["target_value", "collapsed_activity_record_count"], ascending=[False, False]).reset_index(drop=True)

    stats = {
        "molecule_level_rows_before_structure_dedup": int(len(molecule_level_df)),
        "unique_canonical_smiles_before_structure_dedup": int(molecule_level_df["canonical_smiles_rdkit"].nunique()),
        "rows_with_duplicate_canonical_smiles_before_structure_dedup": int(len(duplicate_smiles_rows)),
        "duplicate_canonical_smiles_groups_before_structure_dedup": int(
            molecule_level_df["canonical_smiles_rdkit"].value_counts().gt(1).sum()
        ),
        "structure_level_rows_after_collapse_before_high_spread_drop": int(len(collapsed_all)),
        "high_duplicate_smiles_spread_groups": int(len(high_spread_duplicate_groups)),
        "final_structure_level_rows_after_structure_dedup": int(len(final)),
        "duplicate_smiles_spread_threshold_log_units": duplicate_spread_threshold,
    }

    return final, duplicate_smiles_rows, {
        "stats": stats,
        "high_spread_duplicate_groups": high_spread_duplicate_groups,
        "collapsed_all": collapsed_all,
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    ensure_output_dir(outdir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run scripts/01_pull_fxa_chembl_raw.py first, or pass --input explicitly."
        )

    print("=" * 100)
    print("Step 02: Curate selected FXa endpoint dataset")
    print("=" * 100)
    print(f"Input: {input_path}")
    print(f"Output directory: {outdir}")
    print(f"Replicate spread threshold: {args.replicate_spread_threshold}")
    print(f"Keep high-spread molecule groups: {args.keep_high_spread}")
    print(f"Aggregation key: {args.group_by}")
    print(f"Collapse duplicate canonical SMILES: {not args.no_collapse_duplicate_smiles}")
    print(f"Duplicate-SMILES spread threshold: {args.duplicate_smiles_spread_threshold}")

    df = pd.read_csv(input_path)
    standard_type, endpoint_col, target_col = infer_endpoint(df)

    print(f"\nInferred selected endpoint type: {standard_type}")
    print(f"Endpoint column: {endpoint_col}")
    print(f"Final target column: {target_col}")
    print("\nImportant:")
    print("Input rows are ChEMBL activity records, not unique compounds.")
    print("This script aggregates activity records and then checks residual structure-level duplicates.")
    print("Standardization: largest fragment + RDKit sanitize/canonical SMILES only.")
    print("No uncharging or tautomer normalization is performed.")

    valid_records, invalid_records, record_counts, pchembl_check = curate_records(
        df=df,
        endpoint_col=endpoint_col,
        min_pendpoint=args.min_pendpoint,
        max_pendpoint=args.max_pendpoint,
        pchembl_tolerance=args.pchembl_tolerance,
        strict_pchembl_check=args.strict_pchembl_check,
    )

    print("\nRecord-level curation counts:")
    for key, value in record_counts.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    print("\npChEMBL sanity check:")
    print(f"  Rows checked: {pchembl_check.get('pchembl_rows_checked', 0):,}")
    print(f"  Max abs diff: {pchembl_check.get('pchembl_max_abs_diff')}")
    print(f"  Mean abs diff: {pchembl_check.get('pchembl_mean_abs_diff')}")
    print(f"  Rows above tolerance: {pchembl_check.get('pchembl_rows_above_tolerance')}")
    print(f"  Note: {pchembl_check.get('pchembl_check_note')}")

    if valid_records.empty:
        raise RuntimeError("No valid records remain after record-level curation.")

    if args.group_by not in valid_records.columns:
        raise ValueError(f"Requested group-by column not found after curation: {args.group_by}")

    molecule_level_summary = make_replicate_summary(
        valid_records=valid_records,
        group_by=args.group_by,
        endpoint_col=endpoint_col,
        target_col=target_col,
        spread_threshold=args.replicate_spread_threshold,
    )

    if molecule_level_summary.empty:
        raise RuntimeError("No molecule-level rows were created after aggregation.")

    high_spread_molecule_groups = molecule_level_summary[molecule_level_summary["high_replicate_spread"]].copy()

    if args.keep_high_spread:
        molecule_level_for_structure_check = molecule_level_summary.copy()
        high_spread_action = "kept_flagged"
    else:
        molecule_level_for_structure_check = molecule_level_summary[~molecule_level_summary["high_replicate_spread"]].copy()
        high_spread_action = "dropped_from_final"

    molecule_level_for_structure_check = molecule_level_for_structure_check.reset_index(drop=True)

    if args.no_collapse_duplicate_smiles:
        final_df = molecule_level_for_structure_check.copy()
        duplicate_smiles_rows = molecule_level_for_structure_check[
            molecule_level_for_structure_check.duplicated("canonical_smiles_rdkit", keep=False)
        ].copy()
        duplicate_smiles_info = {
            "stats": {
                "molecule_level_rows_before_structure_dedup": int(len(molecule_level_for_structure_check)),
                "unique_canonical_smiles_before_structure_dedup": int(molecule_level_for_structure_check["canonical_smiles_rdkit"].nunique()),
                "rows_with_duplicate_canonical_smiles_before_structure_dedup": int(len(duplicate_smiles_rows)),
                "duplicate_canonical_smiles_groups_before_structure_dedup": int(
                    molecule_level_for_structure_check["canonical_smiles_rdkit"].value_counts().gt(1).sum()
                ),
                "structure_level_rows_after_collapse_before_high_spread_drop": "not_applied",
                "high_duplicate_smiles_spread_groups": "not_applied",
                "final_structure_level_rows_after_structure_dedup": int(len(final_df)),
            },
            "high_spread_duplicate_groups": pd.DataFrame(),
            "collapsed_all": final_df.copy(),
        }
        structure_dedup_action = "not_applied"
    else:
        final_df, duplicate_smiles_rows, duplicate_smiles_info = collapse_duplicate_smiles(
            molecule_level_for_structure_check,
            endpoint_col=endpoint_col,
            target_col=target_col,
            duplicate_spread_threshold=args.duplicate_smiles_spread_threshold,
        )
        structure_dedup_action = "collapsed_by_canonical_smiles_rdkit"

    # Final modeling columns first.
    front_cols = [
        "molecule_chembl_id",
        "smiles",
        "canonical_smiles_rdkit",
        "endpoint",
        target_col,
        "target_value",
        "n_activity_records",
        "p_endpoint_range",
        "high_replicate_spread",
    ]
    if "n_molecule_chembl_ids_collapsed" in final_df.columns:
        front_cols.extend(["n_molecule_chembl_ids_collapsed", "collapsed_molecule_chembl_ids"])

    front_cols = [c for c in front_cols if c in final_df.columns]
    remaining_cols = [c for c in final_df.columns if c not in front_cols]
    final_df = final_df[front_cols + remaining_cols].reset_index(drop=True)

    # Output files.
    prefix = "fxa_02"

    valid_records_path = outdir / f"{prefix}_valid_activity_records_pre_aggregation.csv"
    invalid_records_path = outdir / f"{prefix}_invalid_or_dropped_record_level_rows.csv"
    molecule_summary_path = outdir / f"{prefix}_molecule_level_replicate_summary.csv"
    high_spread_molecule_path = outdir / f"{prefix}_high_replicate_spread_molecule_groups.csv"
    duplicate_smiles_path = outdir / f"{prefix}_duplicate_canonical_smiles_rows.csv"
    duplicate_smiles_collapsed_all_path = outdir / f"{prefix}_structure_level_collapsed_all_before_duplicate_spread_drop.csv"
    high_duplicate_smiles_path = outdir / f"{prefix}_high_duplicate_smiles_spread_groups.csv"
    final_path = outdir / f"{prefix}_curated_structure_level.csv"
    legacy_final_path = outdir / f"{prefix}_curated_compound_level.csv"
    summary_json_path = outdir / f"{prefix}_curation_summary.json"
    counts_csv_path = outdir / f"{prefix}_curation_counts.csv"

    valid_records.to_csv(valid_records_path, index=False)
    invalid_records.to_csv(invalid_records_path, index=False)
    molecule_level_summary.to_csv(molecule_summary_path, index=False)
    high_spread_molecule_groups.to_csv(high_spread_molecule_path, index=False)
    duplicate_smiles_rows.to_csv(duplicate_smiles_path, index=False)

    collapsed_all = duplicate_smiles_info.get("collapsed_all", pd.DataFrame())
    if isinstance(collapsed_all, pd.DataFrame):
        collapsed_all.to_csv(duplicate_smiles_collapsed_all_path, index=False)

    high_dup = duplicate_smiles_info.get("high_spread_duplicate_groups", pd.DataFrame())
    if isinstance(high_dup, pd.DataFrame):
        high_dup.to_csv(high_duplicate_smiles_path, index=False)

    final_df.to_csv(final_path, index=False)
    # Also write legacy name to avoid breaking downstream scripts if they expected this file.
    final_df.to_csv(legacy_final_path, index=False)

    duplicate_stats = duplicate_smiles_info.get("stats", {})

    final_counts = {
        "input_activity_records": int(len(df)),
        "input_unique_molecule_chembl_ids": int(df["molecule_chembl_id"].nunique()) if "molecule_chembl_id" in df.columns else 0,
        "valid_activity_records_pre_aggregation": int(len(valid_records)),
        "valid_unique_molecule_chembl_ids_pre_aggregation": int(valid_records["molecule_chembl_id"].nunique()),
        "valid_unique_canonical_smiles_pre_aggregation": int(valid_records["canonical_smiles_rdkit"].nunique()),
        "molecule_level_rows_after_aggregation": int(len(molecule_level_summary)),
        "molecule_level_unique_canonical_smiles_after_aggregation": int(molecule_level_summary["canonical_smiles_rdkit"].nunique()),
        "high_replicate_spread_molecule_groups": int(len(high_spread_molecule_groups)),
        "molecule_level_rows_after_high_spread_filter": int(len(molecule_level_for_structure_check)),
        "structure_level_rows_final": int(len(final_df)),
        "structure_level_unique_canonical_smiles_final": int(final_df["canonical_smiles_rdkit"].nunique()) if not final_df.empty else 0,
        "final_unique_molecule_chembl_ids": int(final_df["molecule_chembl_id"].nunique()) if not final_df.empty else 0,
    }

    all_counts = {**record_counts, **final_counts, **duplicate_stats}
    counts_df = pd.DataFrame([{"metric": key, "value": value} for key, value in all_counts.items()])
    counts_df.to_csv(counts_csv_path, index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "03_curate_fxa_dataset.py",
        "input_file": str(input_path),
        "selected_standard_type": standard_type,
        "endpoint_column": endpoint_col,
        "target_column": target_col,
        "group_by_first_stage": args.group_by,
        "replicate_spread_threshold_log_units": args.replicate_spread_threshold,
        "high_spread_molecule_action": high_spread_action,
        "structure_dedup_action": structure_dedup_action,
        "duplicate_smiles_spread_threshold_log_units": args.duplicate_smiles_spread_threshold,
        "standardization_policy": {
            "largest_fragment": True,
            "rdkit_sanitize": True,
            "canonical_isomeric_smiles": True,
            "uncharging": False,
            "tautomer_canonicalization": False,
            "note": (
                "Light standardization only. Charge/tautomer normalization is intentionally not applied "
                "to avoid changing chemical identity assumptions without explicit review."
            ),
        },
        "pchembl_cross_check": pchembl_check,
        "record_level_counts": record_counts,
        "duplicate_smiles_stats": duplicate_stats,
        "final_counts": final_counts,
        "output_files": {
            "valid_records_pre_aggregation": str(valid_records_path),
            "invalid_or_dropped_record_level_rows": str(invalid_records_path),
            "molecule_level_replicate_summary": str(molecule_summary_path),
            "high_replicate_spread_molecule_groups": str(high_spread_molecule_path),
            "duplicate_canonical_smiles_rows": str(duplicate_smiles_path),
            "structure_level_collapsed_all_before_duplicate_spread_drop": str(duplicate_smiles_collapsed_all_path),
            "high_duplicate_smiles_spread_groups": str(high_duplicate_smiles_path),
            "final_curated_structure_level": str(final_path),
            "legacy_final_curated_compound_level": str(legacy_final_path),
            "curation_counts": str(counts_csv_path),
        },
        "curation_notes": [
            "Input rows are ChEMBL activity records, not unique compounds.",
            "pEndpoint is recomputed from nM values using pEndpoint = 9 - log10(nM).",
            "pEndpoint is cross-checked against ChEMBL pchembl_value where available.",
            "SMILES are parsed and canonicalized with RDKit after keeping the largest fragment.",
            "No uncharging or tautomer canonicalization is performed.",
            "Replicate records are first aggregated by molecule_chembl_id using median pEndpoint.",
            "High replicate-spread molecule groups are dropped unless --keep-high-spread is used.",
            "Residual duplicate RDKit canonical SMILES groups are collapsed by default to reduce train/test leakage risk.",
            "The final CSV is the structure-level modeling dataset for descriptor baselines and GNN modeling.",
        ],
    }

    write_json(summary_json_path, summary)

    print("\n" + "=" * 100)
    print("Step 02 curation result")
    print("=" * 100)
    print(f"Valid activity records before aggregation: {len(valid_records):,}")
    print(f"Molecule-level rows after aggregation: {len(molecule_level_summary):,}")
    print(f"Molecule-level unique canonical SMILES: {molecule_level_summary['canonical_smiles_rdkit'].nunique():,}")
    print(f"High replicate-spread molecule groups: {len(high_spread_molecule_groups):,}")
    print(f"Molecule-level rows after high-spread filter: {len(molecule_level_for_structure_check):,}")
    print("\nResidual duplicate canonical-SMILES check:")
    for key, value in duplicate_stats.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    print(f"\nFinal structure-level rows: {len(final_df):,}")
    print(f"Final unique canonical SMILES: {final_df['canonical_smiles_rdkit'].nunique():,}" if not final_df.empty else "Final unique canonical SMILES: 0")
    if not final_df.empty and len(final_df) != final_df["canonical_smiles_rdkit"].nunique():
        print("WARNING: Final rows still exceed unique canonical SMILES. Review duplicate_canonical_smiles output.")
    else:
        print("PASS: Final rows match unique canonical SMILES.")

    print("\nSaved outputs:")
    print(f"  Final structure-level dataset: {final_path}")
    print(f"  Legacy final compound-level dataset name: {legacy_final_path}")
    print(f"  Molecule-level replicate summary: {molecule_summary_path}")
    print(f"  Duplicate canonical SMILES rows: {duplicate_smiles_path}")
    print(f"  High duplicate-SMILES spread groups: {high_duplicate_smiles_path}")
    print(f"  Invalid/dropped record-level rows: {invalid_records_path}")
    print(f"  Counts CSV: {counts_csv_path}")
    print(f"  Metadata JSON: {summary_json_path}")

    print("\nNext step:")
    print("Use fxa_02_curated_structure_level.csv for descriptor baselines and graph dataset construction.")


if __name__ == "__main__":
    main()

