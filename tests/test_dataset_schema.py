from pathlib import Path
import json

import pandas as pd

from src.build_vqi_dataset import PREOP_FEATURES, POSTOP_FEATURES


SCHEMA_REPORT = Path("data/processed/merged_vqi_schema_report.json")
METADATA_PATH = Path("data/processed/merged_vqi_2012_2020_metadata.csv")


def test_feature_counts_constants() -> None:
    assert len(PREOP_FEATURES) == 30
    assert len(POSTOP_FEATURES) == 14


def test_schema_report_alignment() -> None:
    assert SCHEMA_REPORT.exists(), "Run src/validate_dataset_schema.py to create the schema report."
    report = json.loads(SCHEMA_REPORT.read_text())
    assert report["expected_preoperative_features"] == 30
    assert report["expected_postoperative_features"] == 14
    assert report["missing_inputs"] == []
    assert report["missing_outputs"] == []
    assert report["unexpected_inputs"] == []
    assert report["unexpected_outputs"] == []
    weight_summary = report["weight_kg_summary"]
    assert weight_summary["unit"] == "kg"
    assert weight_summary["max"] <= 200  # kg upper bound sanity check


def test_metadata_feature_roles() -> None:
    metadata = pd.read_csv(METADATA_PATH)
    inputs = metadata.loc[metadata["feature_role"] == "input", "column"]
    outputs = metadata.loc[metadata["feature_role"] == "output", "column"]
    assert set(inputs) == set(PREOP_FEATURES)
    assert set(outputs) == set(POSTOP_FEATURES)

