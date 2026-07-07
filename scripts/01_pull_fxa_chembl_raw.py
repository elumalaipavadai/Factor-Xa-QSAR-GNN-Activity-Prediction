#!/usr/bin/env python
"""
01_pull_fxa_chembl_raw.py

P0 task:
- Pull Factor Xa activities from ChEMBL.
- Count available Ki and IC50 records.
- Choose ONE activity type only:
    * Prefer Ki if it gives enough clean data.
    * Use IC50 if Ki is too small.
    * Do not combine Ki and IC50.
- Filter records:
    * standard_units == nM
    * standard_relation == =
    * positive numeric standard_value
    * non-missing canonical_smiles
    * prefer binding assay records (assay_type == "B") if enough remain

Default target:
- Human coagulation factor X / Factor Xa in ChEMBL: CHEMBL244

Usage from project root:
    conda activate fxa_portfolio_clean
    python scripts/01_pull_fxa_chembl_raw.py

Optional:
    python scripts/01_pull_fxa_chembl_raw.py --min-records 200
    python scripts/01_pull_fxa_chembl_raw.py --target-chembl-id CHEMBL244
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from chembl_webresource_client.new_client import new_client


ACTIVITY_TYPES = ["Ki", "IC50"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull and filter Factor Xa Ki/IC50 activity records from ChEMBL."
    )
    parser.add_argument(
        "--target-chembl-id",
        default="CHEMBL244",
        help="ChEMBL target ID for human coagulation factor X / Factor Xa. Default: CHEMBL244",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=200,
        help=(
            "Minimum number of clean records required to consider an activity type "
            "or binding-assay subset large enough. Default: 200"
        ),
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Project root directory. Default: current directory",
    )
    return parser.parse_args()


def ensure_dirs(project_root: Path) -> Dict[str, Path]:
    dirs = {
        "raw": project_root / "data" / "raw",
        "processed": project_root / "data" / "processed",
        "reports": project_root / "results" / "metrics",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def fetch_activity_records(target_chembl_id: str, standard_type: str) -> pd.DataFrame:
    """
    Fetch ChEMBL activity records for one target and one standard_type.

    Note:
    The ChEMBL client returns strings for many fields. Numeric conversion is handled later.
    """
    print(f"\nFetching {standard_type} records for target {target_chembl_id} ...")

    records = list(
        new_client.activity.filter(
            target_chembl_id=target_chembl_id,
            standard_type=standard_type,
        )
    )

    df = pd.DataFrame(records)
    print(f"Raw {standard_type} records pulled: {len(df):,}")

    if df.empty:
        return df

    df["requested_activity_type"] = standard_type
    return df


def require_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Ensure columns exist so downstream code does not fail when ChEMBL omits a field.
    """
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def normalize_relation(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace("'", "")


def clean_activity_records(df: pd.DataFrame, activity_type: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply P0 filters:
    - standard_units == nM
    - standard_relation == =
    - standard_value positive numeric
    - canonical_smiles present
    """
    needed = [
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
    df = require_columns(df.copy(), needed)

    counts = {}
    counts["raw_total"] = len(df)

    # Standard type sanity filter. ChEMBL should already filter this, but keep it explicit.
    df1 = df[df["standard_type"].astype(str).str.upper() == activity_type.upper()].copy()
    counts["after_standard_type"] = len(df1)

    # Units filter.
    units = df1["standard_units"].astype(str).str.strip().str.lower()
    df2 = df1[units == "nm"].copy()
    counts["after_units_nM"] = len(df2)

    # Relation filter.
    relations = df2["standard_relation"].map(normalize_relation)
    df3 = df2[relations == "="].copy()
    counts["after_relation_equal"] = len(df3)

    # Numeric positive values.
    df3["standard_value_num"] = pd.to_numeric(df3["standard_value"], errors="coerce")
    df4 = df3[df3["standard_value_num"].notna() & (df3["standard_value_num"] > 0)].copy()
    counts["after_positive_numeric_value"] = len(df4)

    # SMILES present.
    smiles = df4["canonical_smiles"].astype(str).str.strip()
    df5 = df4[smiles.notna() & (smiles != "") & (smiles.str.lower() != "nan")].copy()
    counts["after_nonmissing_smiles"] = len(df5)

    # Binding assay subset.
    assay_type = df5["assay_type"].astype(str).str.strip().str.upper()
    df_binding = df5[assay_type == "B"].copy()
    counts["binding_assay_clean"] = len(df_binding)

    # Add standardized potency endpoint.
    # nM to molar: nM * 1e-9; pActivity = -log10(M) = 9 - log10(nM)
    endpoint_name = "pKi" if activity_type.upper() == "KI" else "pIC50"
    df5[endpoint_name] = df5["standard_value_num"].apply(lambda x: 9.0 - math.log10(float(x)))
    df5["selected_endpoint"] = endpoint_name
    df5["activity_type_cleaned"] = activity_type

    if not df_binding.empty:
        df_binding[endpoint_name] = df_binding["standard_value_num"].apply(lambda x: 9.0 - math.log10(float(x)))
        df_binding["selected_endpoint"] = endpoint_name
        df_binding["activity_type_cleaned"] = activity_type

    return df5, counts


def choose_activity_type(
    clean_data: Dict[str, pd.DataFrame],
    counts: Dict[str, Dict[str, int]],
    min_records: int,
) -> Tuple[str, str, pd.DataFrame, Dict[str, str]]:
    """
    Decision rule:
    1. Prefer Ki if Ki has >= min_records clean records.
    2. Otherwise use IC50 if IC50 has >= min_records clean records.
    3. If neither meets threshold, choose the type with more clean records, but label the warning.
    4. For the selected activity type, use binding assays only if binding subset >= min_records.
       Otherwise use all clean records for that activity type.
    """
    ki_n = counts["Ki"]["after_nonmissing_smiles"]
    ic50_n = counts["IC50"]["after_nonmissing_smiles"]

    if ki_n >= min_records:
        selected_type = "Ki"
        reason = f"Ki selected because it has {ki_n:,} clean records, meeting threshold {min_records:,}."
    elif ic50_n >= min_records:
        selected_type = "IC50"
        reason = (
            f"IC50 selected because Ki has only {ki_n:,} clean records, "
            f"below threshold {min_records:,}; IC50 has {ic50_n:,} clean records."
        )
    else:
        selected_type = "Ki" if ki_n >= ic50_n else "IC50"
        reason = (
            f"WARNING: Neither Ki nor IC50 reached threshold {min_records:,}. "
            f"Selected {selected_type} because it has the larger clean count "
            f"(Ki={ki_n:,}, IC50={ic50_n:,}). Consider lowering threshold only if justified."
        )

    selected_clean = clean_data[selected_type]
    binding_n = counts[selected_type]["binding_assay_clean"]

    if binding_n >= min_records:
        final_df = selected_clean[selected_clean["assay_type"].astype(str).str.strip().str.upper() == "B"].copy()
        assay_filter_used = "binding_only"
        assay_reason = (
            f"Binding assays preferred and used because {selected_type} has "
            f"{binding_n:,} clean binding-assay records, meeting threshold {min_records:,}."
        )
    else:
        final_df = selected_clean.copy()
        assay_filter_used = "all_assay_types"
        assay_reason = (
            f"Binding assays not enforced because {selected_type} has only "
            f"{binding_n:,} clean binding-assay records, below threshold {min_records:,}. "
            f"Using all clean {selected_type} assay records."
        )

    metadata = {
        "selected_activity_type": selected_type,
        "selected_assay_filter": assay_filter_used,
        "activity_selection_reason": reason,
        "assay_selection_reason": assay_reason,
    }

    return selected_type, assay_filter_used, final_df, metadata


def select_output_columns(df: pd.DataFrame, selected_type: str) -> pd.DataFrame:
    endpoint_col = "pKi" if selected_type.upper() == "KI" else "pIC50"

    columns = [
        "activity_id",
        "molecule_chembl_id",
        "canonical_smiles",
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
        "selected_endpoint",
        "activity_type_cleaned",
    ]

    df = require_columns(df.copy(), columns)
    return df[columns].copy()


def main() -> None:
    args = parse_args()

    project_root = Path(args.outdir).resolve()
    dirs = ensure_dirs(project_root)

    target_chembl_id = args.target_chembl_id
    min_records = args.min_records

    print("=" * 80)
    print("P0: Factor Xa ChEMBL Ki/IC50 data pull and first-pass filtering")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"Target ChEMBL ID: {target_chembl_id}")
    print(f"Minimum records threshold: {min_records:,}")
    print("Rule: choose Ki if enough clean data; otherwise IC50. Do not combine Ki and IC50.")

    raw_data = {}
    clean_data = {}
    counts = {}

    for activity_type in ACTIVITY_TYPES:
        raw_df = fetch_activity_records(target_chembl_id, activity_type)
        raw_data[activity_type] = raw_df

        raw_path = dirs["raw"] / f"chembl_fxa_{target_chembl_id}_{activity_type}_raw.csv"
        raw_df.to_csv(raw_path, index=False)
        print(f"Saved raw {activity_type}: {raw_path}")

        clean_df, activity_counts = clean_activity_records(raw_df, activity_type)
        clean_data[activity_type] = clean_df
        counts[activity_type] = activity_counts

        clean_preview_path = dirs["processed"] / f"chembl_fxa_{target_chembl_id}_{activity_type}_clean_preview.csv"
        clean_df.to_csv(clean_preview_path, index=False)
        print(f"Saved clean preview {activity_type}: {clean_preview_path}")

    # Make count table.
    count_rows = []
    for activity_type in ACTIVITY_TYPES:
        row = {"activity_type": activity_type}
        row.update(counts[activity_type])
        count_rows.append(row)

    counts_df = pd.DataFrame(count_rows)
    counts_path = dirs["processed"] / "fxa_p0_activity_counts.csv"
    counts_df.to_csv(counts_path, index=False)

    print("\nP0 activity count summary:")
    print(counts_df.to_string(index=False))
    print(f"\nSaved counts: {counts_path}")

    selected_type, assay_filter_used, final_df, metadata = choose_activity_type(
        clean_data=clean_data,
        counts=counts,
        min_records=min_records,
    )

    final_df = select_output_columns(final_df, selected_type)
    final_path = dirs["processed"] / f"fxa_p0_selected_{selected_type.lower()}_{assay_filter_used}.csv"
    final_df.to_csv(final_path, index=False)

    metadata.update(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "target_chembl_id": target_chembl_id,
            "min_records": min_records,
            "ki_counts": counts["Ki"],
            "ic50_counts": counts["IC50"],
            "final_record_count": int(len(final_df)),
            "final_output": str(final_path),
            "raw_outputs": {
                "Ki": str(dirs["raw"] / f"chembl_fxa_{target_chembl_id}_Ki_raw.csv"),
                "IC50": str(dirs["raw"] / f"chembl_fxa_{target_chembl_id}_IC50_raw.csv"),
            },
            "count_output": str(counts_path),
            "important_rule": "Ki and IC50 were counted separately. Only one activity type was selected. They were not combined.",
        }
    )

    metadata_path = dirs["processed"] / "fxa_p0_selection_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("P0 selection result")
    print("=" * 80)
    print(f"Selected activity type: {selected_type}")
    print(f"Selected assay filter: {assay_filter_used}")
    print(metadata["activity_selection_reason"])
    print(metadata["assay_selection_reason"])
    print(f"Final records saved: {len(final_df):,}")
    print(f"Final selected dataset: {final_path}")
    print(f"Selection metadata: {metadata_path}")
    print("\nDone. Next step: inspect counts and selected dataset before deduplication/aggregation.")


if __name__ == "__main__":
    main()

