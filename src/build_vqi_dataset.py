"""
Construct a harmonized VQI dataset that mirrors the predictor/features set
described in the manuscript appendices.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path.cwd()
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_EXCEL = RAW_DIR / "VQI_Database_MTAEdits.xlsx"
PROCESSED_PARQUET = PROCESSED_DIR / "merged_vqi_2012_2020.parquet"
PROCESSED_METADATA = PROCESSED_DIR / "merged_vqi_2012_2020_metadata.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Sheets corresponding to each procedure cohort and a label to retain.
PROCEDURE_SHEETS = {
    "INFRA": "INFRA_Database",
    "SUPRA": "SUPRA_Database",
    "OPEN_AAA": "OPEN_AAA_Database_",
}

# Final predictor set: 30 pre-operative predictors + 14 postoperative descriptors.
PREOP_FEATURES: List[str] = [
    "DIABETES",
    "SEX",
    "PRIOR_BYPASS",
    "PRIOR_CHF",
    "RACE",
    "TRANSFER",
    "ANEURREP",
    "COPD",
    "DIALYSIS",
    "HTN",
    "PREOP_SMOKING",
    "PREOP_ASA",
    "PREOP_BETABLOCKER",
    "PREOP_P2Y",
    "PREOP_STATIN",
    "LIVINGSTATUS",
    "STRESS",
    "ETHNICITY",
    "PREOP_AMBUL",
    "ASACLASS",
    "PRIOR_CABG",
    "PRIOR_CEACAS",
    "PRIOR_PCI",
    "PREOP_ACE",
    "PREOP_ANTICOAG",
    "AGE",
    "HTCM",
    "WEIGHT_KG",
    "HEMO",
    "PREOP_CREAT",
]

POSTOP_FEATURES: List[str] = [
    "RTOR",
    "POSTOP_DYS",
    "ANTIBIOTICSTART",
    "ANTIBIOTICEND",
    "ANTIBIOTICGEN",
    "LTF_CALC",
    "DC_STATUS",
    "RESPIRATORY",
    "POSTOP_MI",
    "POSTOP_STROKE",
    "POSTOP_LOS",
    "TXFUSION",
    "PROC_SURVIVALDAYS",
    "DEAD"
]

FINAL_COLUMNS: List[str] = PREOP_FEATURES + POSTOP_FEATURES
FEATURE_ROLE: Dict[str, str] = {
    **{column: "input" for column in PREOP_FEATURES},
    **{column: "output" for column in POSTOP_FEATURES},
}

# Column aliases required to harmonise across the three cohorts.
COLUMN_ALIASES: Dict[str, Dict[str, str]] = {
    "SEX": {"INFRA": "GENDER", "SUPRA": "GENDER", "OPEN_AAA": "GENDER"},
    "DIABETES": {"SUPRA": "PREOP_DIABETES"},
    "POSTOP_DYS": {
        "INFRA": "DYSRHYTHMIA",
        "SUPRA": "POSTOP_DYSRHYTHMIA",
        "OPEN_AAA": "POSTOP_DYSRHYTHMIA",
    },
    "POSTOP_MI": {
        "INFRA": "NEWMI",
        "SUPRA": "MYOCARDIAL_INFARCTION",
        "OPEN_AAA": "POSTOP_MI",
    },
    "POSTOP_STROKE": {
        "INFRA": "STROKE",
        "SUPRA": "POSTOP_STROKE",
        "OPEN_AAA": "POSTOP_STROKE",
    },
    "PREOP_CREAT": {
        "INFRA": "CREATININE",
        "SUPRA": "PREOP_CREAT",
        "OPEN_AAA": "PREOP_CREAT",
    },
    "WEIGHT_KG": {
        "INFRA": "WTLB",
        "SUPRA": "WTLB",
        "OPEN_AAA": "WTLB",
    },
    "TXFUSION": {"INFRA": "PRBC", "SUPRA": "TXFUSION", "OPEN_AAA": "TXFUSION"},
    "DIALYSIS": {"SUPRA": "PREOP_DIALYSIS"},
}

# Explicit typing guidance.
CATEGORICAL_FEATURES = {
    "DIABETES",
    "SEX",
    "PRIOR_BYPASS",
    "PRIOR_CHF",
    "RACE",
    "TRANSFER",
    "ANEURREP",
    "COPD",
    "DIALYSIS",
    "HTN",
    "PREOP_SMOKING",
    "PREOP_ASA",
    "PREOP_BETABLOCKER",
    "PREOP_P2Y",
    "PREOP_STATIN",
    "LIVINGSTATUS",
    "STRESS",
    "ETHNICITY",
    "PREOP_AMBUL",
    "ASACLASS",
    "PRIOR_CABG",
    "PRIOR_CEACAS",
    "PRIOR_PCI",
    "PREOP_ACE",
    "PREOP_ANTICOAG",
    "RTOR",
    "POSTOP_DYS",
    "ANTIBIOTICSTART",
    "ANTIBIOTICEND",
    "ANTIBIOTICGEN",
    "DC_STATUS",
    "RESPIRATORY",
    "POSTOP_MI",
    "POSTOP_STROKE",
    "DEAD"
}

NUMERIC_FEATURES = set(FINAL_COLUMNS) - CATEGORICAL_FEATURES


def resolve_column(procedure: str, column: str, frame: pd.DataFrame) -> pd.Series:
    """Return the series corresponding to the harmonised column name."""
    alias = COLUMN_ALIASES.get(column, {}).get(procedure)
    source = alias or column

    if source not in frame.columns:
        raise KeyError(
            f"Expected column '{source}' for '{column}' not found in "
            f"{procedure} dataset."
        )

    series = frame[source]
    non_missing = series.dropna()

    if non_missing.empty:
        raise ValueError(
            f"Column '{source}' (mapped to '{column}') in {procedure} "
            "contains only missing values."
        )

    if non_missing.dtype == object:
        stripped = non_missing.astype(str).str.strip()
        if stripped.eq("").all():
            raise ValueError(
                f"Column '{source}' (mapped to '{column}') in {procedure} "
                "contains only empty strings."
            )

    if column == "WEIGHT_KG":
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            source_name = (alias or source or "").lower()
            if "lb" in source_name:
                converted = numeric * 0.45359237
            else:
                converted = numeric
        else:
            converted = numeric
        return converted

    return series


def load_cohort(
    procedure: str, sheet: str, excel: pd.ExcelFile
) -> tuple[pd.DataFrame, int]:
    """Load and harmonise a single procedure cohort."""
    frame = excel.parse(sheet, dtype="object")
    harmonised = {}
    for column in FINAL_COLUMNS:
        series = resolve_column(procedure, column, frame)
        harmonised[column] = series

    harmonised_df = pd.DataFrame(harmonised)
    harmonised_df.insert(0, "PROCEDURE_GROUP", procedure)
    return harmonised_df, len(frame)


def cast_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply categorical/numeric casting rules."""
    result = frame.copy()
    result["PROCEDURE_GROUP"] = result["PROCEDURE_GROUP"].astype("category")
    for column in CATEGORICAL_FEATURES:
        result[column] = (
            result[column]
            .astype("string")
            .astype("category")
        )
    for column in NUMERIC_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def main() -> None:
    print("Loading Excel file...")
    excel = pd.ExcelFile(RAW_EXCEL)
    cohorts = []
    expected_rows = 0
    for procedure, sheet in PROCEDURE_SHEETS.items():
        print(f"Loading {procedure} cohort...")
        harmonised, n_rows = load_cohort(procedure, sheet, excel)
        cohorts.append(harmonised)
        expected_rows += n_rows

    print("Concatenating cohorts...")
    merged = pd.concat(cohorts, axis=0, ignore_index=True)
    merged = cast_types(merged)

    print("Casting types...")
    if len(merged) != expected_rows:
        raise ValueError(
            f"Row count mismatch: expected {expected_rows}, found {len(merged)}"
        )

    print("Writing merged dataset to parquet...")
    merged.to_parquet(PROCESSED_PARQUET, index=False)

    # Missingness summary.
    print("Generating metadata...")
    metadata = (
        merged.isna()
        .mean()
        .rename("missing_fraction")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "column"})
    )
    metadata["feature_role"] = metadata["column"].map(FEATURE_ROLE).fillna("other")
    metadata.to_csv(
        PROCESSED_METADATA,
        index=False,
    )

    print("Metadata:")
    print(metadata)
    print("Done!")

if __name__ == "__main__":
    main()

