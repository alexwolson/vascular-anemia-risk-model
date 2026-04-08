"""
Produce a 4-row AUC comparison table documenting the effect of data cleaning
and the URGENCY feature on model performance.

Usage:
    uv run python src/analysis_auc_comparison.py

Requires that both the 31-feature and 30-feature models have been trained
via run_h2o_automl.py after data cleaning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_31 = REPO_ROOT / "artifacts" / "h2o_automl" / "DEAD_run_metadata.json"
METADATA_30 = REPO_ROOT / "artifacts" / "h2o_automl_30feat" / "DEAD_run_metadata.json"
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_PATH = OUTPUT_DIR / "auc_comparison.csv"


def load_auc(path: Path) -> float:
    meta = json.loads(path.read_text())
    return float(meta["metric_value"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "model": "Original manuscript",
            "features": "30 (with sentinel values)",
            "auc": 0.7854,
        },
        {
            "model": "New analysis, pre-cleaning",
            "features": "31 (+ URGENCY, with sentinels)",
            "auc": 0.795,
        },
    ]

    if METADATA_31.exists():
        rows.append({
            "model": "New analysis, post-cleaning, 31 features",
            "features": "31 (+ URGENCY, cleaned)",
            "auc": load_auc(METADATA_31),
        })
    else:
        print(f"WARNING: {METADATA_31} not found. Run the 31-feature model first.")

    if METADATA_30.exists():
        rows.append({
            "model": "New analysis, post-cleaning, 30 features",
            "features": "30 (cleaned)",
            "auc": load_auc(METADATA_30),
        })
    else:
        print(f"WARNING: {METADATA_30} not found. Run the 30-feature model first.")

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_PATH, index=False)
    print(f"AUC comparison table saved to {OUTPUT_PATH}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
