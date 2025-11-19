"""
Validate the harmonised VQI dataset against the replication checklist schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from build_vqi_dataset import (
    FEATURE_ROLE,
    PREOP_FEATURES,
    POSTOP_FEATURES,
    PROCESSED_DIR,
    PROCESSED_METADATA,
    PROCESSED_PARQUET,
)

REPORT_PATH = PROCESSED_DIR / "merged_vqi_schema_report.json"


def describe_weight(series: pd.Series) -> Dict[str, Any]:
    if series.empty:
        return {"note": "column_missing"}
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return {"note": "all_missing"}
    quantiles = numeric.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    quantiles = {key: float(value) for key, value in quantiles.items()}
    quantiles["unit"] = "kg"
    return quantiles


def main() -> None:
    dataset = pd.read_parquet(PROCESSED_PARQUET)
    metadata = pd.read_csv(PROCESSED_METADATA)

    expected_inputs = set(PREOP_FEATURES)
    expected_outputs = set(POSTOP_FEATURES)
    dataset_columns = set(dataset.columns)

    metadata_inputs = set(
        metadata.loc[metadata["feature_role"] == "input", "column"].tolist()
    )
    metadata_outputs = set(
        metadata.loc[metadata["feature_role"] == "output", "column"].tolist()
    )

    report: Dict[str, Any] = {
        "expected_preoperative_features": len(expected_inputs),
        "actual_preoperative_features": len(metadata_inputs),
        "expected_postoperative_features": len(expected_outputs),
        "actual_postoperative_features": len(metadata_outputs),
        "missing_inputs": sorted(expected_inputs - metadata_inputs),
        "unexpected_inputs": sorted(metadata_inputs - expected_inputs),
        "missing_outputs": sorted(expected_outputs - metadata_outputs),
        "unexpected_outputs": sorted(metadata_outputs - expected_outputs),
        "dataset_only_columns": sorted(
            dataset_columns
            - expected_inputs
            - expected_outputs
            - {"PROCEDURE_GROUP"}
        ),
        "weight_kg_summary": describe_weight(
            dataset.get("WEIGHT_KG", pd.Series(dtype="float64"))
        ),
    }

    metadata_summary = (
        metadata.groupby("feature_role")["column"].count().to_dict()
    )
    role_counts = {role: int(count) for role, count in metadata_summary.items()}
    report["metadata_role_counts"] = role_counts

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote schema report to {REPORT_PATH}")


if __name__ == "__main__":
    main()


