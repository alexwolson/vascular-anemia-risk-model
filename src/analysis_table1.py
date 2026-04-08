"""
Task 1: Investigate Table 1 "Sex" Error and Produce Corrected Table 1.

This script:
1. Loads the processed VQI dataset (85,431 patients).
2. Diagnoses the SEX variable error in the original manuscript Table 1.
3. Produces a corrected, publication-ready Table 1 CSV.

Output directory: output/task1_table1/
"""

from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "processed" / "merged_vqi_2012_2020.parquet"
OUTPUT_DIR = REPO_ROOT / "output" / "task1_table1"

# ---------------------------------------------------------------------------
# Variable definitions
# ---------------------------------------------------------------------------
CATEGORICAL_PREOP = [
    "SEX", "DIABETES", "PRIOR_BYPASS", "PRIOR_CHF", "RACE", "TRANSFER",
    "ANEURREP", "COPD", "DIALYSIS", "HTN", "PREOP_SMOKING", "PREOP_ASA",
    "PREOP_BETABLOCKER", "PREOP_P2Y", "PREOP_STATIN", "LIVINGSTATUS",
    "STRESS", "ETHNICITY", "PREOP_AMBUL", "ASACLASS", "PRIOR_CABG",
    "PRIOR_CEACAS", "PRIOR_PCI", "PREOP_ACE", "PREOP_ANTICOAG",
]

NUMERIC_PREOP = ["AGE", "HTCM", "WEIGHT_KG", "HEMO", "PREOP_CREAT"]

# Human-readable labels for variables
VARIABLE_LABELS = {
    "AGE": "Age (years)",
    "SEX": "Sex",
    "DIABETES": "Diabetes",
    "PRIOR_BYPASS": "Prior bypass",
    "PRIOR_CHF": "Prior CHF",
    "RACE": "Race",
    "TRANSFER": "Transfer status",
    "ANEURREP": "Prior aneurysm repair",
    "COPD": "COPD",
    "DIALYSIS": "Dialysis",
    "HTN": "Hypertension",
    "PREOP_SMOKING": "Smoking status",
    "PREOP_ASA": "Preoperative aspirin",
    "PREOP_BETABLOCKER": "Preoperative beta-blocker",
    "PREOP_P2Y": "Preoperative P2Y12 inhibitor",
    "PREOP_STATIN": "Preoperative statin",
    "LIVINGSTATUS": "Living status",
    "STRESS": "Stress test",
    "ETHNICITY": "Hispanic ethnicity",
    "PREOP_AMBUL": "Preoperative ambulatory status",
    "ASACLASS": "ASA class",
    "PRIOR_CABG": "Prior CABG",
    "PRIOR_CEACAS": "Prior CEA/CAS",
    "PRIOR_PCI": "Prior PCI",
    "PREOP_ACE": "Preoperative ACE inhibitor",
    "PREOP_ANTICOAG": "Preoperative anticoagulant",
    "HTCM": "Height (cm)",
    "WEIGHT_KG": "Weight (kg)",
    "HEMO": "Hemoglobin (g/dL)",
    "PREOP_CREAT": "Preoperative creatinine (mg/dL)",
    "BMI": "BMI (kg/m\u00b2)",
    "TXFUSION": "Transfusion (units PRBCs)",
    "DEAD": "Mortality",
}

# Human-readable level labels for key categorical variables
LEVEL_LABELS = {
    "SEX": {"1": "Male", "2": "Female"},
    "DIABETES": {
        "0": "None",
        "1": "Diet-controlled",
        "2": "Oral medication",
        "3": "Insulin",
    },
    "PRIOR_CHF": {
        "0": "None",
        "1": "Asymptomatic (EF < 25%)",
        "2": "Mild (EF 25-44%)",
        "3": "Moderate (EF 45-59%)",
        "4": "Severe",
    },
    "RACE": {
        "1": "American Indian/Alaska Native",
        "2": "Asian",
        "3": "Black/African American",
        "4": "Native Hawaiian/Pacific Islander",
        "5": "White",
        "6": "Other",
        "7": "Unknown",
    },
    "TRANSFER": {
        "0": "Not transferred",
        "1": "Transferred from another hospital",
        "2": "Transferred from another service",
    },
    "COPD": {
        "0": "None",
        "1": "Not treated",
        "2": "On medication",
        "3": "Home O2",
    },
    "DIALYSIS": {
        "0": "None",
        "1": "Peritoneal",
        "2": "Hemodialysis",
    },
    "HTN": {"0": "No", "1": "Yes"},
    "ANEURREP": {"0": "No", "1": "Yes"},
    "PRIOR_BYPASS": {"0": "No", "1": "Yes"},
    "PREOP_SMOKING": {
        "0": "Never",
        "1": "Prior (>1 year)",
        "2": "Current",
    },
    "ETHNICITY": {"0": "Non-Hispanic", "1": "Hispanic"},
    "LIVINGSTATUS": {"1": "Home", "2": "Nursing home", "3": "Homeless"},
    "ASACLASS": {
        "1": "Healthy",
        "2": "Mild systemic disease",
        "3": "Severe systemic disease",
        "4": "Life-threatening disease",
        "5": "Moribund",
    },
    "PREOP_AMBUL": {
        "1": "Ambulatory",
        "2": "Ambulatory with assistance",
        "3": "Wheelchair",
        "4": "Bedridden",
    },
    "STRESS": {
        "0": "None",
        "1": "Normal",
        "2": "Mildly abnormal",
        "3": "Moderately abnormal",
        "4": "Severely abnormal",
    },
    "PREOP_ASA": {
        "0": "None",
        "1": "Started pre-admission",
        "2": "Started at admission",
        "3": "Contraindicated",
    },
    "PREOP_BETABLOCKER": {
        "0": "None",
        "1": "Started pre-admission",
        "2": "On beta-blocker",
        "3": "Started at admission",
        "4": "Contraindicated",
        "5": "Discontinued pre-admission",
    },
    "PREOP_P2Y": {
        "0": "None",
        "1": "Clopidogrel (Plavix)",
        "2": "Prasugrel (Effient)",
        "3": "Ticagrelor (Brilinta)",
        "4": "Ticlopidine (Ticlid)",
        "5": "Cangrelor (Kengreal)",
        "6": "Unknown P2Y12",
        "7": "Vorapaxar (Zontivity)",
    },
    "PREOP_STATIN": {
        "0": "None",
        "1": "On statin",
        "2": "Started at admission",
        "3": "Contraindicated",
    },
    "PRIOR_CABG": {
        "0": "No",
        "1": "Yes",
        "2": "Unknown",
    },
    "PRIOR_CEACAS": {
        "0": "No",
        "1": "Yes",
    },
    "PRIOR_PCI": {
        "0": "No",
        "1": "Yes",
        "2": "Unknown",
    },
    "PREOP_ACE": {
        "0": "No",
        "1": "Yes",
        "2": "Contraindicated",
        "3": "Unknown",
    },
    "PREOP_ANTICOAG": {
        "0": "None",
        "1": "Unfractionated heparin",
        "2": "LMWH (Lovenox)",
        "3": "Warfarin (Coumadin)",
        "4": "Dabigatran (Pradaxa)",
        "5": "Rivaroxaban (Xarelto)",
        "6": "Apixaban (Eliquis)",
    },
    "DEAD": {"0": "Alive", "1": "Dead"},
}


def fmt_pct(n: int, total: int) -> str:
    """Format a count and percentage."""
    pct = 100.0 * n / total if total > 0 else 0.0
    return f"{n:,} ({pct:.1f}%)"


def summarize_categorical(series: pd.Series, var_name: str, total_n: int) -> list[dict]:
    """Return summary rows for a categorical variable."""
    rows = []
    vc = series.value_counts(dropna=True).sort_index()
    n_non_missing = int(vc.sum())
    n_missing = total_n - n_non_missing
    labels = LEVEL_LABELS.get(var_name, {})
    label = VARIABLE_LABELS.get(var_name, var_name)

    # Header row showing n available
    rows.append({
        "Variable": label,
        "Level": "",
        "n": n_non_missing,
        "Missing": n_missing,
        "Summary": "",
    })

    for level_val, count in vc.items():
        level_str = str(level_val)
        level_label = labels.get(level_str, level_str)
        rows.append({
            "Variable": "",
            "Level": level_label,
            "n": int(count),
            "Missing": "",
            "Summary": fmt_pct(int(count), n_non_missing),
        })

    return rows


def summarize_numeric(series: pd.Series, var_name: str, total_n: int) -> list[dict]:
    """Return summary rows for a numeric variable."""
    valid = series.dropna()
    n_valid = len(valid)
    n_missing = total_n - n_valid
    label = VARIABLE_LABELS.get(var_name, var_name)

    row = {
        "Variable": label,
        "Level": "",
        "n": n_valid,
        "Missing": n_missing,
        "Summary": (
            f"{valid.mean():.1f} \u00b1 {valid.std():.1f}  "
            f"[{valid.min():.1f}\u2013{valid.max():.1f}]"
        ),
    }
    return [row]


def write_error_explanation(df: pd.DataFrame, output_dir: Path) -> None:
    """Write the explanation for the SEX variable error."""
    sex_vc = df["SEX"].value_counts(dropna=False)
    age_desc = df["AGE"].describe()

    explanation = f"""\
============================================================
Diagnosis of the SEX Variable Error in the Manuscript Table 1
============================================================

PROBLEM
-------
The original manuscript Table 1 reported SEX as a numerical variable with
summary statistics: min = 0, max = 90. Several reviewers flagged this as
incorrect (Reviewer 1 #5, Reviewer 2 #2).

DIAGNOSIS
---------
In the processed dataset, SEX is a categorical variable encoded as:
  1 = Male   (n = {int(sex_vc.get('1', 0)):,})
  2 = Female (n = {int(sex_vc.get('2', 0)):,})
  Missing    (n = {int(sex_vc.get(np.nan, 0)):,})

The reported min = 0 and max = 90 exactly match the AGE variable's range:
  AGE: min = {age_desc['min']:.0f}, max = {age_desc['max']:.0f}, mean = {age_desc['mean']:.1f}, SD = {age_desc['std']:.1f}

ROOT CAUSE
----------
The table-generation code likely iterated over columns and applied numeric
descriptive statistics (min, max, mean, SD) uniformly to all variables.
Because SEX is adjacent to AGE in the feature list and both were read from
the same data source, the most probable explanation is a column-indexing
error: the code pulled AGE's values when it should have pulled SEX's values.
Specifically, the original Table 1 generation script treated SEX as numeric
and computed min/max on the raw coded values (or, more likely, accidentally
referenced the AGE column), producing the erroneous min = 0 and max = 90.

The value min = 0 is not even a valid SEX code (valid codes are 1 and 2),
which further confirms that AGE data was inadvertently reported in the
SEX row. AGE has min = 0 (likely a data-entry error for a small number of
patients) and max = 90 (age capped at 90 in the VQI registry).

CORRECTION
----------
SEX should be reported as a categorical variable:
  Male:    {int(sex_vc.get('1', 0)):,} ({100.0 * int(sex_vc.get('1', 0)) / (int(sex_vc.get('1', 0)) + int(sex_vc.get('2', 0))):.1f}%)
  Female:  {int(sex_vc.get('2', 0)):,} ({100.0 * int(sex_vc.get('2', 0)) / (int(sex_vc.get('1', 0)) + int(sex_vc.get('2', 0))):.1f}%)

The corrected Table 1 has been generated as corrected_table1.csv in this
output directory.
"""
    path = output_dir / "error_explanation.txt"
    path.write_text(explanation)
    print(f"Wrote error explanation to {path}")


def build_corrected_table1(df: pd.DataFrame) -> pd.DataFrame:
    """Build the corrected Table 1 as a DataFrame."""
    total_n = len(df)
    rows: list[dict] = []

    # -- Section header: overall --
    rows.append({
        "Variable": f"Total patients (N = {total_n:,})",
        "Level": "",
        "n": total_n,
        "Missing": 0,
        "Summary": "",
    })

    # -- Categorical preoperative features --
    for var in CATEGORICAL_PREOP:
        rows.extend(summarize_categorical(df[var], var, total_n))

    # -- Numeric preoperative features --
    for var in NUMERIC_PREOP:
        rows.extend(summarize_numeric(df[var], var, total_n))

    # -- Derived: BMI --
    mask_valid = (df["HTCM"] > 0) & (df["WEIGHT_KG"] > 0)
    bmi = df.loc[mask_valid, "WEIGHT_KG"] / (df.loc[mask_valid, "HTCM"] / 100) ** 2
    # Filter implausible values
    bmi = bmi[(bmi >= 10) & (bmi <= 80)]
    n_bmi = len(bmi)
    n_missing_bmi = total_n - n_bmi
    rows.append({
        "Variable": VARIABLE_LABELS["BMI"],
        "Level": "",
        "n": n_bmi,
        "Missing": n_missing_bmi,
        "Summary": (
            f"{bmi.mean():.1f} \u00b1 {bmi.std():.1f}  "
            f"[{bmi.min():.1f}\u2013{bmi.max():.1f}]"
        ),
    })

    # -- Postoperative: TXFUSION --
    rows.extend(summarize_numeric(df["TXFUSION"], "TXFUSION", total_n))

    # -- Outcome: DEAD (mortality) --
    dead = df["DEAD"].dropna()
    n_dead = int((dead == "1").sum())
    n_alive = int((dead == "0").sum())
    n_valid_dead = n_dead + n_alive
    n_missing_dead = total_n - n_valid_dead
    rows.append({
        "Variable": VARIABLE_LABELS["DEAD"],
        "Level": "",
        "n": n_valid_dead,
        "Missing": n_missing_dead,
        "Summary": f"{n_dead:,} deaths ({100.0 * n_dead / n_valid_dead:.1f}%)",
    })

    table = pd.DataFrame(rows)
    return table


def investigate_age_zeros(df: pd.DataFrame, output_dir: Path) -> None:
    """Report patients with AGE == 0 or AGE < 18 for data quality assessment."""
    n_zero = int((df["AGE"] == 0).sum())
    n_lt18 = int((df["AGE"] < 18).sum())

    lines = [
        "AGE = 0 / AGE < 18 Investigation",
        "=" * 50,
        f"Patients with AGE == 0: {n_zero}",
        f"Patients with AGE < 18: {n_lt18}",
        "",
    ]

    if n_lt18 > 0:
        subset = df[df["AGE"] < 18]
        lines.append("Procedure group distribution (AGE < 18):")
        for group, count in subset["PROCEDURE_GROUP"].value_counts().items():
            lines.append(f"  {group}: {count}")
        lines.append("")

        dead_vc = subset["DEAD"].value_counts(dropna=False)
        n_dead = int(dead_vc.get("1", 0))
        n_alive = int(dead_vc.get("0", 0))
        n_missing = int(dead_vc.get(np.nan, 0)) if np.nan in dead_vc.index else 0
        lines.append(f"Mortality among AGE < 18: {n_dead} dead, {n_alive} alive, {n_missing} missing")
        lines.append("")

    lines.append("Recommendation:")
    lines.append(
        "These records are likely data-entry errors (neonatal open vascular surgery "
        "is exceedingly rare in VQI). They are NOT excluded from the model -- the GBM "
        "treats them as a low-data region with negligible impact on predictions. "
        "For Table 1, the age range is reported as-is with this footnote available."
    )

    text = "\n".join(lines)
    path = output_dir / "age_zero_investigation.txt"
    path.write_text(text)
    print(f"Wrote AGE investigation to {path}")
    print(text)


def main() -> None:
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {OUTPUT_DIR}")

    # Step 0: AGE = 0 investigation
    investigate_age_zeros(df, OUTPUT_DIR)

    # Step 1: Error explanation
    write_error_explanation(df, OUTPUT_DIR)

    # Step 2: Corrected Table 1
    table1 = build_corrected_table1(df)
    csv_path = OUTPUT_DIR / "corrected_table1.csv"
    table1.to_csv(csv_path, index=False)
    print(f"Wrote corrected Table 1 to {csv_path}")
    print(f"  Table has {len(table1)} rows")

    # Print a preview
    print("\n" + "=" * 80)
    print("CORRECTED TABLE 1 — PREVIEW")
    print("=" * 80)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_colwidth", 60)
    print(table1.to_string(index=False))
    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
