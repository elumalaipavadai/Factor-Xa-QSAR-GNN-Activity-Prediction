#!/usr/bin/env python
"""
02_compare_fxa_endpoints.py

Factor X / Factor Xa ChEMBL endpoint-baseline pull.

This script supports TWO honest baseline modes:

1) ki-only
   - Pulls Ki records only.
   - Best mode if you want a closer paper-aligned descriptor-QSAR baseline.
   - Output endpoint: pKi

2) mixed
   - Pulls Ki + IC50 records.
   - Converts both to a shared pActivity column.
   - This is NOT a paper reproduction.
   - Use only as a larger-N mixed-endpoint sensitivity/baseline dataset.

Why this script exists:
- The main GNN portfolio dataset should remain endpoint-controlled:
      Ki OR IC50 only, not combined.
- This script creates baseline datasets for comparison and sensitivity analysis.
- It clearly reports record counts AND unique compound counts to avoid quoting
  activity-record counts as compound counts.

Default target:
- CHEMBL244 = human coagulation factor X / activated Factor Xa target.

Run from project root:
    conda activate fxa_portfolio_clean
    python scripts/02_compare_fxa_endpoints.py --mode ki-only

Or mixed-endpoint sensitivity:
    python scripts/02_compare_fxa_endpoints.py --mode mixed

Optional assay filter:
    python scripts/02_compare_fxa_endpoints.py --mode mixed --assay-type B

CAUTION:
- ChEMBL assay_type == "B" is not automatically better for enzyme inhibition data.
- Many legitimate enzyme inhibition assays may be classified as "F" functional assays.
- Inspect the B/F assay split before filtering by assay_type.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
from chembl_webresource_client.new_client import new_client


CHEMBL_ACTIVITY_FIELDS = [
    "activity_id",
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "assay_type",
    "assay_chembl_id",
    "assay_description",
    "target_chembl_id",
    "target_pref_name",
    "document_chembl_id",
]

MODE_TO_ACTIVITY_TYPES = {
    "ki-only": ["Ki"],
    "mixed": ["Ki", "IC50"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull Factor Xa ChEMBL endpoint baseline datasets: Ki-only or mixed Ki+IC50."
    )
    parser.add_argument(
        "--mode",
        choices=["ki-only", "mixed"],
        default="ki-only",
        help=(
            "Baseline mode. 'ki-only' is closer to the reference-paper endpoint. "
            "'mixed' combines Ki + IC50 as a sensitivity baseline. Default: ki-only."
        ),
    )
    parser.add_argument(
        "--target-chembl-id",
        default="CHEMBL244",
        help="ChEMBL target ID. Default CHEMBL244 is human coagulation factor X / activated Factor Xa.",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Project root directory. Default: current working directory.",
    )
    parser.add_argument(
        "--assay-type",
        default=None,
        choices=["B", "F"],
        help=(
            "Optional ChEMBL assay_type filter. Use with caution. "
            "B = binding; F = functional. Default: no assay_type filter."
        ),
    )
    parser.add_argument(
        "--pchembl-tolerance",
        type=float,
        default=0.05,
        help="Tolerance for comparing computed pEndpoint with ChEMBL pchembl_value. Default: 0.05.",
    )
    parser.add_argument(
        "--strict-pchembl-check",
        action="store_true",
        help="Fail if computed endpoint and pchembl_value differ beyond tolerance.",
    )
    return parser.parse_args()


def ensure_dirs(project_root: Path) -> Dict[str, Path]:
    dirs = {
        "raw": project_root / "data" / "raw",
        "processed": project_root / "data" / "processed",
        "metrics": project_root / "results" / "metrics",
    }
    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    return dirs


def fetch_activity_records(target_chembl_id: str, activity_type: str) -> pd.DataFrame:
    print(f"\nFetching {activity_type} records for target {target_chembl_id} ...")
    base_query = new_client.activity.filter(
        target_chembl_id=target_chembl_id,
        standard_type=activity_type,
    )

    try:
        query = base_query.only(CHEMBL_ACTIVITY_FIELDS)
    except TypeError:
        query = base_query.only(*CHEMBL_ACTIVITY_FIELDS)

    records = list(query)
    df = pd.DataFrame(records)
    print(f"Raw {activity_type} activity records pulled: {len(df):,}")

    if not df.empty:
        df["activity_type_source"] = activity_type
    return df


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


def compute_p_endpoint(df: pd.DataFrame, mode: str) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    endpoint_col = "pKi" if mode == "ki-only" else "pActivity"
    df[endpoint_col] = df["standard_value_num"].apply(lambda x: 9.0 - math.log10(float(x)))
    return df, endpoint_col


def summarize_records_and_compounds(df: pd.DataFrame, prefix: str) -> Dict[str, int]:
    return {
        f"{prefix}_records": int(len(df)),
        f"{prefix}_unique_molecule_chembl_ids": int(df["molecule_chembl_id"].nunique())
        if "molecule_chembl_id" in df.columns
        else 0,
        f"{prefix}_unique_smiles": int(df["canonical_smiles"].nunique())
        if "canonical_smiles" in df.columns
        else 0,
    }


def pchembl_cross_check(
    df: pd.DataFrame,
    endpoint_col: str,
    tolerance: float,
    strict: bool,
) -> Dict[str, Any]:
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

    tmp["abs_diff"] = (tmp[endpoint_col].astype(float) - tmp["pchembl_value_num"].astype(float)).abs()
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
            f"PASS: computed {endpoint_col} agrees with pchembl_value within "
            f"{tolerance} log units where pchembl_value exists."
        )
    else:
        result["pchembl_check_note"] = (
            f"WARNING: {n_bad} rows differ from pchembl_value by more than "
            f"{tolerance} log units. Check unit handling and ChEMBL pchembl rules."
        )

    if strict and n_bad > 0:
        raise AssertionError(result["pchembl_check_note"])

    return result


def filter_records(
    df: pd.DataFrame,
    mode: str,
    assay_type_filter: Optional[str],
    pchembl_tolerance: float,
    strict_pchembl_check: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    expected_types = MODE_TO_ACTIVITY_TYPES[mode]
    needed = CHEMBL_ACTIVITY_FIELDS + ["activity_type_source"]
    df = require_columns(df.copy(), needed)

    counts: Dict[str, Any] = {}
    counts.update(summarize_records_and_compounds(df, "raw"))

    df1 = df[df["standard_type"].astype(str).isin(expected_types)].copy()
    counts.update(summarize_records_and_compounds(df1, "after_endpoint_type_filter"))

    units = df1["standard_units"].astype(str).str.strip().str.lower()
    df2 = df1[units == "nm"].copy()
    counts.update(summarize_records_and_compounds(df2, "after_units_nM"))

    relations = df2["standard_relation"].map(normalize_relation)
    df3 = df2[relations == "="].copy()
    counts.update(summarize_records_and_compounds(df3, "after_relation_equal"))

    df3["standard_value_num"] = pd.to_numeric(df3["standard_value"], errors="coerce")
    df4 = df3[df3["standard_value_num"].notna() & (df3["standard_value_num"] > 0)].copy()
    counts.update(summarize_records_and_compounds(df4, "after_positive_numeric_value"))

    smiles = df4["canonical_smiles"].astype(str).str.strip()
    df5 = df4[smiles.notna() & (smiles != "") & (smiles.str.lower() != "nan")].copy()
    counts.update(summarize_records_and_compounds(df5, "after_nonmissing_smiles"))

    assay_split = (
        df5["assay_type"]
        .fillna("NA")
        .astype(str)
        .str.strip()
        .str.upper()
        .value_counts(dropna=False)
        .to_dict()
    )
    counts["assay_type_split_after_basic_filters"] = {str(k): int(v) for k, v in assay_split.items()}

    if assay_type_filter:
        assay_norm = df5["assay_type"].astype(str).str.strip().str.upper()
        df6 = df5[assay_norm == assay_type_filter.upper()].copy()
        counts["assay_type_filter_applied"] = assay_type_filter.upper()
        counts.update(summarize_records_and_compounds(df6, f"after_assay_type_{assay_type_filter.upper()}"))
    else:
        df6 = df5.copy()
        counts["assay_type_filter_applied"] = "not_applied"

    df7, endpoint_col = compute_p_endpoint(df6, mode=mode)
    counts.update(summarize_records_and_compounds(df7, "final_pre_dedup"))

    endpoint_split = (
        df7["activity_type_source"]
        .fillna("NA")
        .astype(str)
        .value_counts(dropna=False)
        .to_dict()
    )
    counts["endpoint_source_split_final_pre_dedup_records"] = {str(k): int(v) for k, v in endpoint_split.items()}

    if mode == "mixed" and not df7.empty:
        endpoint_sets = {
            endpoint: set(group["molecule_chembl_id"].dropna().astype(str))
            for endpoint, group in df7.groupby("activity_type_source")
        }
        ki_set = endpoint_sets.get("Ki", set())
        ic50_set = endpoint_sets.get("IC50", set())
        counts["compounds_with_both_Ki_and_IC50_final_pre_dedup"] = int(len(ki_set & ic50_set))
    else:
        counts["compounds_with_both_Ki_and_IC50_final_pre_dedup"] = 0

    counts["pchembl_cross_check"] = pchembl_cross_check(
        df7,
        endpoint_col=endpoint_col,
        tolerance=pchembl_tolerance,
        strict=strict_pchembl_check,
    )

    return df7, counts, endpoint_col


def select_output_columns(df: pd.DataFrame, endpoint_col: str) -> pd.DataFrame:
    columns = [
        "activity_id",
        "molecule_chembl_id",
        "canonical_smiles",
        "activity_type_source",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_value_num",
        "standard_units",
        endpoint_col,
        "pchembl_value",
        "assay_type",
        "assay_chembl_id",
        "assay_description",
        "target_chembl_id",
        "target_pref_name",
        "document_chembl_id",
    ]
    df = require_columns(df.copy(), columns)
    return df[columns].copy()


def make_short_mode_label(mode: str, assay_type_filter: Optional[str]) -> str:
    assay_label = f"assay_{assay_type_filter.upper()}" if assay_type_filter else "all_assay_types"
    return f"{mode.replace('-', '_')}_{assay_label}"


def main() -> None:
    args = parse_args()
    project_root = Path(args.outdir).resolve()
    dirs = ensure_dirs(project_root)

    mode = args.mode
    target_chembl_id = args.target_chembl_id
    activity_types = MODE_TO_ACTIVITY_TYPES[mode]
    assay_type_filter = args.assay_type
    label = make_short_mode_label(mode, assay_type_filter)

    print("=" * 100)
    print("01b: Factor Xa ChEMBL endpoint baseline pull")
    print("=" * 100)
    print(f"Project root: {project_root}")
    print(f"Target ChEMBL ID: {target_chembl_id}")
    print(f"Mode: {mode}")
    print(f"Activity types pulled: {', '.join(activity_types)}")
    print(f"Optional assay_type filter: {assay_type_filter if assay_type_filter else 'not applied'}")

    if mode == "ki-only":
        print("\nInterpretation:")
        print("Ki-only mode is the closer paper-aligned baseline endpoint.")
        print("Output endpoint column: pKi")
    else:
        print("\nInterpretation:")
        print("Mixed mode intentionally combines Ki + IC50 into pActivity.")
        print("This is NOT a paper reproduction; use it as a mixed-endpoint sensitivity baseline.")
        print("Output endpoint column: pActivity")

    print("\nImportant counting caveat:")
    print("All final counts are pre-deduplication activity-record counts unless explicitly labeled")
    print("as unique molecule_chembl_id or unique SMILES counts.")
    print("Do not quote activity-record counts as compound counts.")

    print("\nAssay-type caution:")
    print("Do not assume assay_type == B is a quality upgrade for enzyme inhibition data.")
    print("Inspect the B/F split before applying any assay_type filter.")

    raw_frames = []
    raw_counts: Dict[str, int] = {}

    for activity_type in activity_types:
        df = fetch_activity_records(target_chembl_id, activity_type)
        raw_frames.append(df)
        raw_counts[activity_type] = int(len(df))

        raw_path = dirs["raw"] / f"chembl_fxa_{target_chembl_id}_{activity_type}_{label}_raw.csv"
        df.to_csv(raw_path, index=False)
        print(f"Saved raw {activity_type}: {raw_path}")

    combined_raw = pd.concat(raw_frames, ignore_index=True, sort=False) if raw_frames else pd.DataFrame()
    combined_raw_path = dirs["raw"] / f"chembl_fxa_{target_chembl_id}_{label}_raw_combined.csv"
    combined_raw.to_csv(combined_raw_path, index=False)
    print(f"\nSaved combined raw file: {combined_raw_path}")

    filtered_df, filter_counts, endpoint_col = filter_records(
        combined_raw,
        mode=mode,
        assay_type_filter=assay_type_filter,
        pchembl_tolerance=args.pchembl_tolerance,
        strict_pchembl_check=args.strict_pchembl_check,
    )

    output_df = select_output_columns(filtered_df, endpoint_col=endpoint_col)

    output_path = dirs["processed"] / f"fxa_01b_{label}_{endpoint_col}_pre_dedup.csv"
    output_df.to_csv(output_path, index=False)

    endpoint_counts = (
        output_df["activity_type_source"]
        .fillna("NA")
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("activity_type_source")
        .reset_index(name="pre_dedup_activity_record_count")
    )

    assay_counts = (
        output_df["assay_type"]
        .fillna("NA")
        .astype(str)
        .str.strip()
        .str.upper()
        .value_counts(dropna=False)
        .rename_axis("assay_type")
        .reset_index(name="pre_dedup_activity_record_count")
    )

    endpoint_counts_path = dirs["processed"] / f"fxa_01b_{label}_endpoint_counts_pre_dedup.csv"
    assay_counts_path = dirs["processed"] / f"fxa_01b_{label}_assay_type_counts_pre_dedup.csv"
    endpoint_counts.to_csv(endpoint_counts_path, index=False)
    assay_counts.to_csv(assay_counts_path, index=False)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "02_compare_fxa_endpoints.py",
        "target_chembl_id": target_chembl_id,
        "mode": mode,
        "activity_types_pulled": activity_types,
        "endpoint_column": endpoint_col,
        "assay_type_filter": assay_type_filter if assay_type_filter else "not_applied",
        "raw_counts_by_activity_type": raw_counts,
        "filter_counts_pre_dedup": filter_counts,
        "counting_caveat": (
            "ChEMBL rows are activity records, not unique compounds. "
            "Counts are pre-deduplication record counts unless explicitly labeled "
            "unique_molecule_chembl_ids or unique_smiles."
        ),
        "deduplication_deferred_to": (
            "Step 02 curation. Recommended aggregation: group by molecule_chembl_id "
            "and endpoint, take median pEndpoint, and flag/drop compounds with replicate "
            "spread > about 1 log unit."
        ),
        "assay_type_caution": (
            "Do not assume assay_type B is always better for enzyme inhibition targets. "
            "Many valid FXa enzymatic inhibition assays may be classified as F."
        ),
        "formula": f"{endpoint_col} = 9 - log10(standard_value_nM)",
        "combined_raw_output": str(combined_raw_path),
        "final_pre_dedup_output": str(output_path),
        "endpoint_count_output": str(endpoint_counts_path),
        "assay_count_output": str(assay_counts_path),
    }

    if mode == "ki-only":
        metadata["purpose"] = (
            "Ki-only baseline dataset. This is closer to the reference-paper endpoint "
            "than mixed Ki+IC50 data."
        )
    else:
        metadata["purpose"] = (
            "Mixed-endpoint baseline/sensitivity dataset. Ki and IC50 are intentionally "
            "combined into pActivity. This should not be described as direct paper reproduction."
        )

    metadata_path = dirs["processed"] / f"fxa_01b_{label}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 100)
    print("01b output summary")
    print("=" * 100)

    print("\nRaw activity-record counts:")
    for activity_type, n in raw_counts.items():
        print(f"  {activity_type}: {n:,}")

    print("\nFilter summary:")
    for key, value in filter_counts.items():
        if key == "pchembl_cross_check":
            continue
        print(f"  {key}: {value}")

    print("\npChEMBL sanity check:")
    pcheck = filter_counts.get("pchembl_cross_check", {})
    print(f"  Rows checked: {pcheck.get('pchembl_rows_checked', 0):,}")
    print(f"  Max abs diff: {pcheck.get('pchembl_max_abs_diff')}")
    print(f"  Mean abs diff: {pcheck.get('pchembl_mean_abs_diff')}")
    print(f"  Rows above tolerance: {pcheck.get('pchembl_rows_above_tolerance', 0)}")
    print(f"  Note: {pcheck.get('pchembl_check_note', '')}")

    print("\nEndpoint source counts, pre-dedup activity records:")
    print(endpoint_counts.to_string(index=False))

    print("\nAssay type counts, pre-dedup activity records:")
    print(assay_counts.to_string(index=False))

    print("\nSaved outputs:")
    print(f"  Final pre-dedup dataset: {output_path}")
    print(f"  Endpoint counts: {endpoint_counts_path}")
    print(f"  Assay type counts: {assay_counts_path}")
    print(f"  Metadata: {metadata_path}")

    print("\nNext step:")
    print("Do Step 02 curation/deduplication before training models.")
    print("Recommended: group by molecule_chembl_id and endpoint, take median pEndpoint,")
    print("and flag/drop compounds with replicate spread > about 1 log unit.")


if __name__ == "__main__":
    main()

