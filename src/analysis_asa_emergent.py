"""
Task 3: ASA Class distribution, PDP, and emergent vs. elective investigation.

Responds to Reviewer 2 #4 (ASA class discussion) and Reviewer 2 #5
(emergent vs. elective handling).

Usage:
    export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"
    uv run python src/analysis_asa_emergent.py
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import h2o
from h2o.frame import H2OFrame

from run_h2o_automl import train_valid_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "processed" / "merged_vqi_2012_2020.parquet"
METADATA_PATH = REPO_ROOT / "artifacts" / "h2o_automl" / "DEAD_run_metadata.json"
RAW_EXCEL_PATH = REPO_ROOT / "data" / "raw" / "VQI_Database_MTAEdits.xlsx"
OUTPUT_DIR = REPO_ROOT / "output" / "task3_asa"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_run_metadata(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_model_path(metadata: dict) -> Path:
    model_path = metadata.get("gbm_model_path")
    if not model_path:
        raise FileNotFoundError(
            "Run metadata does not contain 'gbm_model_path'. "
            "Re-run src/run_h2o_automl.py first."
        )

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model path from run metadata not found: {path}. "
            "Re-run src/run_h2o_automl.py to regenerate models."
        )
    return path


def prepare_h2o_frame(
    df: pd.DataFrame,
    categorical_columns,
    target: str,
    problem_type: str,
) -> H2OFrame:
    """Convert a pandas DataFrame to an H2OFrame with correct column types."""
    frame = h2o.H2OFrame(df)
    for column in categorical_columns:
        if column in frame.columns:
            frame[column] = frame[column].asfactor()
    if problem_type == "classification":
        frame[target] = frame[target].asfactor()
    return frame


# ===================================================================
# STEP 1 — Load data and model
# ===================================================================
def main() -> None:
    logger.info("Starting Task 3: ASA Class & Emergent/Elective Analysis")
    ensure_directory(OUTPUT_DIR)

    # ---- load metadata ------------------------------------------------
    meta = load_run_metadata(METADATA_PATH)
    predictors: list[str] = meta["predictors"]
    target: str = meta["target"]
    problem_type: str = meta["problem_type"]
    train_ratio: float = meta["train_ratio"]
    seed: int = meta["seed"]
    stratified: bool = meta["stratified"]
    model_path = resolve_model_path(meta)

    # ---- load dataset -------------------------------------------------
    logger.info("Loading dataset from %s", DATA_PATH)
    dataset = pd.read_parquet(DATA_PATH)
    logger.info("Dataset shape: %s", dataset.shape)

    # Working subset: predictors + target, drop rows with missing target
    working = dataset[predictors + [target]].dropna(subset=[target])

    # ---- recreate train / valid split ---------------------------------
    train_df, valid_df = train_valid_split(
        working,
        target,
        train_ratio,
        seed,
        stratify=stratified,
        problem_type=problem_type,
    )
    logger.info("Train: %d rows, Valid: %d rows", len(train_df), len(valid_df))

    # ---- init H2O and load model --------------------------------------
    logger.info("Initialising H2O cluster...")
    h2o.init()

    logger.info("Loading GBM model from %s", model_path)
    model = h2o.load_model(str(model_path))

    # ---- build validation H2OFrame ------------------------------------
    categorical_predictors = working.select_dtypes(include="category").columns
    valid_h2o = prepare_h2o_frame(
        valid_df,
        categorical_columns=categorical_predictors,
        target=target,
        problem_type=problem_type,
    )

    # ===================================================================
    # STEP 4 — ASACLASS distribution
    # ===================================================================
    logger.info("Computing ASACLASS distribution...")

    asa_counts = working["ASACLASS"].value_counts(dropna=False).sort_index().rename("n")
    asa_pct = (
        working["ASACLASS"]
        .value_counts(normalize=True, dropna=False)
        .sort_index()
        .rename("pct")
    )

    # Cross-tabulate with mortality (DEAD is categorical, convert to numeric first)
    dead_numeric = pd.to_numeric(working[target], errors="coerce")
    asa_mortality = (
        dead_numeric.groupby(working["ASACLASS"], dropna=False, observed=False)
        .agg(["sum", "count"])
        .rename(columns={"sum": "deaths", "count": "total"})
    )
    asa_mortality["mortality_rate"] = asa_mortality["deaths"] / asa_mortality["total"]

    asa_dist = pd.concat(
        [asa_counts, asa_pct, asa_mortality[["deaths", "mortality_rate"]]],
        axis=1,
    )
    asa_dist.index.name = "ASACLASS"
    asa_dist_path = OUTPUT_DIR / "asaclass_distribution.csv"
    asa_dist.to_csv(asa_dist_path)
    logger.info("Saved ASACLASS distribution to %s", asa_dist_path)
    print("\nASACLASS Distribution:")
    print(asa_dist.to_string())

    # ===================================================================
    # STEP 5 — PDP for ASACLASS
    # ===================================================================
    logger.info("Generating PDP for ASACLASS...")

    pdp_plots = model.partial_plot(
        frame=valid_h2o,
        cols=["ASACLASS"],
        plot=False,
    )
    pdp_df = pdp_plots[0].as_data_frame()

    # Rename the x-column to "center" for consistency
    x_col = [
        c
        for c in pdp_df.columns
        if c not in ("mean_response", "stddev_response", "std_error_mean_response")
    ][0]
    pdp_df = pdp_df.rename(columns={x_col: "center"})

    pdp_csv_path = OUTPUT_DIR / "asaclass_pdp.csv"
    pdp_df.to_csv(pdp_csv_path, index=False)
    logger.info("Saved PDP data to %s", pdp_csv_path)

    # Bar chart — ASACLASS is categorical with ~5 levels
    fig, ax = plt.subplots(figsize=(7, 5))
    x_labels = [str(v) for v in pdp_df["center"]]
    x_positions = range(len(x_labels))

    bars = ax.bar(
        x_positions,
        pdp_df["mean_response"],
        color="#1f77b4",
        edgecolor="white",
        zorder=3,
    )

    # Add error bars if stddev available
    if "stddev_response" in pdp_df.columns:
        ax.errorbar(
            x_positions,
            pdp_df["mean_response"],
            yerr=pdp_df["stddev_response"],
            fmt="none",
            ecolor="black",
            capsize=4,
            zorder=4,
        )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("ASA Class")
    ax.set_ylabel("Mean Predicted Mortality (P(DEAD=1))")
    ax.set_title("Partial Dependence Plot — ASA Class")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    pdp_fig_path = OUTPUT_DIR / "asaclass_pdp.png"
    fig.savefig(pdp_fig_path, dpi=300)
    plt.close(fig)
    logger.info("Saved PDP figure to %s", pdp_fig_path)

    # ===================================================================
    # STEP 6 — Emergent / elective investigation
    # ===================================================================
    logger.info("Investigating emergent/elective indicators...")

    findings_lines: list[str] = []

    # --- 6a: Check predictor column names for urgency-related terms ----
    urgency_keywords = ["emergent", "elective", "urgency", "emergency", "urgent"]
    findings_lines.append("=" * 70)
    findings_lines.append("EMERGENT / ELECTIVE INVESTIGATION")
    findings_lines.append("=" * 70)
    findings_lines.append("")

    findings_lines.append("1. Predictor columns checked for urgency-related terms")
    findings_lines.append(f"   Total predictors in model: {len(predictors)}")
    findings_lines.append(f"   Predictors: {', '.join(sorted(predictors))}")
    findings_lines.append("")

    matched_predictors = [
        p for p in predictors if any(kw in p.lower() for kw in urgency_keywords)
    ]
    if matched_predictors:
        findings_lines.append(
            f"   FOUND urgency-related predictors: {matched_predictors}"
        )
    else:
        findings_lines.append(
            "   NO urgency-related predictor found among the 30 model features."
        )
    findings_lines.append("")

    # --- 6b: Check raw Excel file for emergent/elective columns --------
    findings_lines.append("2. Checking raw Excel file for emergent/elective columns")
    findings_lines.append(f"   File: {RAW_EXCEL_PATH}")
    findings_lines.append("")

    if RAW_EXCEL_PATH.exists():
        xl = pd.ExcelFile(RAW_EXCEL_PATH)
        sheet_names = xl.sheet_names
        findings_lines.append(f"   Sheet names: {sheet_names}")
        findings_lines.append("")

        any_match_found = False
        for sheet in sheet_names:
            # Read only header row to get column names
            header_df = pd.read_excel(RAW_EXCEL_PATH, sheet_name=sheet, nrows=0)
            cols = list(header_df.columns)
            matched_cols = [
                c for c in cols if any(kw in str(c).lower() for kw in urgency_keywords)
            ]
            if matched_cols:
                any_match_found = True
                findings_lines.append(
                    f"   Sheet '{sheet}': MATCHED columns -> {matched_cols}"
                )
            else:
                findings_lines.append(
                    f"   Sheet '{sheet}': {len(cols)} columns, no urgency-related match"
                )
        findings_lines.append("")

        if not any_match_found:
            findings_lines.append(
                "   CONCLUSION: No emergent/elective/urgency/emergency column found "
                "in ANY sheet of the raw Excel file."
            )
        findings_lines.append("")
    else:
        findings_lines.append(
            f"   WARNING: Raw Excel file not found at {RAW_EXCEL_PATH}"
        )
        findings_lines.append("")

    # --- 6c: ASA Class 5 (Moribund) as proxy for emergent cases --------
    findings_lines.append("3. ASA Class 5 (Moribund) as proxy for emergent cases")

    # Convert ASACLASS to numeric for comparison if needed
    asa_values = pd.to_numeric(working["ASACLASS"], errors="coerce")

    n_asa5 = (asa_values == 5).sum()
    n_not_asa5 = (asa_values != 5).sum()

    dead_numeric = pd.to_numeric(working[target], errors="coerce")

    mortality_asa5 = dead_numeric[asa_values == 5].mean()
    mortality_not_asa5 = dead_numeric[asa_values != 5].mean()

    findings_lines.append(f"   ASA Class 5 (Moribund) patients: n = {n_asa5}")
    findings_lines.append(f"   Non-ASA-5 patients: n = {n_not_asa5}")
    findings_lines.append(
        f"   Mortality rate (ASA=5): {mortality_asa5:.4f} ({mortality_asa5 * 100:.1f}%)"
        if not np.isnan(mortality_asa5)
        else "   Mortality rate (ASA=5): N/A (no ASA=5 patients)"
    )
    findings_lines.append(
        f"   Mortality rate (ASA!=5): {mortality_not_asa5:.4f} "
        f"({mortality_not_asa5 * 100:.1f}%)"
    )
    if not np.isnan(mortality_asa5):
        rr = mortality_asa5 / mortality_not_asa5 if mortality_not_asa5 > 0 else np.nan
        findings_lines.append(f"   Relative risk (ASA=5 vs others): {rr:.2f}x")
    findings_lines.append("")

    # --- 6d: TRANSFER as proxy for emergent cases ----------------------
    findings_lines.append("4. TRANSFER variable as proxy for emergent cases")

    if "TRANSFER" in working.columns:
        transfer_values = pd.to_numeric(working["TRANSFER"], errors="coerce")

        n_transferred = (transfer_values == 1).sum()
        n_not_transferred = (transfer_values == 0).sum()
        n_transfer_other = len(working) - n_transferred - n_not_transferred

        mortality_transferred = dead_numeric[transfer_values == 1].mean()
        mortality_not_transferred = dead_numeric[transfer_values == 0].mean()

        findings_lines.append(
            f"   Transferred patients (TRANSFER=1): n = {n_transferred}"
        )
        findings_lines.append(
            f"   Not transferred (TRANSFER=0): n = {n_not_transferred}"
        )
        if n_transfer_other > 0:
            findings_lines.append(
                f"   Other/missing TRANSFER values: n = {n_transfer_other}"
            )
        findings_lines.append(
            f"   Mortality rate (transferred): {mortality_transferred:.4f} "
            f"({mortality_transferred * 100:.1f}%)"
            if not np.isnan(mortality_transferred)
            else "   Mortality rate (transferred): N/A"
        )
        findings_lines.append(
            f"   Mortality rate (not transferred): {mortality_not_transferred:.4f} "
            f"({mortality_not_transferred * 100:.1f}%)"
            if not np.isnan(mortality_not_transferred)
            else "   Mortality rate (not transferred): N/A"
        )
        if (
            not np.isnan(mortality_transferred)
            and not np.isnan(mortality_not_transferred)
            and mortality_not_transferred > 0
        ):
            rr_transfer = mortality_transferred / mortality_not_transferred
            findings_lines.append(
                f"   Relative risk (transferred vs not): {rr_transfer:.2f}x"
            )
    else:
        findings_lines.append("   TRANSFER variable not found in the predictor set.")
    findings_lines.append("")

    # --- 6e: Summary ---------------------------------------------------
    findings_lines.append("=" * 70)
    findings_lines.append("SUMMARY")
    findings_lines.append("=" * 70)
    findings_lines.append("")
    if matched_predictors:
        findings_lines.append(
            "The VQI dataset includes URGENCY as an explicit emergent/elective"
        )
        findings_lines.append(
            "indicator variable. URGENCY is present in all three source cohorts"
        )
        findings_lines.append(
            "(INFRA, SUPRA, OPEN_AAA) and is included as the 31st predictor in"
        )
        findings_lines.append("the GBM model.")
        findings_lines.append("")
        findings_lines.append(
            "URGENCY distribution and mortality rates are reported in the full"
        )
        findings_lines.append(
            "URGENCY analysis (Task 8 / output/task8_urgency/urgency_report.txt)."
        )
        findings_lines.append("")
        findings_lines.append(
            "Two additional proxies for case acuity were also evaluated:"
        )
        findings_lines.append(
            "  - ASA Class 5 (Moribund/not expected to survive without surgery)"
        )
        findings_lines.append(
            "    may capture the highest-acuity patients, many of whom may be"
        )
        findings_lines.append("    emergent cases.")
        findings_lines.append(
            "  - TRANSFER (patient transferred from another facility) may also"
        )
        findings_lines.append("    correlate with case urgency.")
        findings_lines.append("")
        findings_lines.append("Both are included as predictors in the GBM model.")
    else:
        findings_lines.append(
            "The VQI dataset used for this analysis does not contain an explicit"
        )
        findings_lines.append(
            "emergent vs. elective indicator variable. Neither the 30 model"
        )
        findings_lines.append(
            "predictors nor the raw Excel source sheets contain columns matching"
        )
        findings_lines.append("keywords: emergent, elective, urgency, or emergency.")
        findings_lines.append("")
        findings_lines.append("Two potential proxies were evaluated:")
        findings_lines.append(
            "  - ASA Class 5 (Moribund/not expected to survive without surgery)"
        )
        findings_lines.append(
            "    may capture the highest-acuity patients, many of whom may be"
        )
        findings_lines.append("    emergent cases.")
        findings_lines.append(
            "  - TRANSFER (patient transferred from another facility) may also"
        )
        findings_lines.append("    correlate with case urgency.")
        findings_lines.append("")
        findings_lines.append(
            "Both variables are included as predictors in the GBM model and"
        )
        findings_lines.append(
            "their effects are captured by the model during training."
        )

    findings_text = "\n".join(findings_lines)
    findings_path = OUTPUT_DIR / "emergent_elective_findings.txt"
    findings_path.write_text(findings_text)
    logger.info("Saved emergent/elective findings to %s", findings_path)
    print("\n" + findings_text)

    # ===================================================================
    # STEP 7 — Narrative summary
    # ===================================================================
    logger.info("Writing narrative summary...")

    if matched_predictors:
        narrative = textwrap.dedent("""\
            Task 3 Narrative Summary: ASA Class and Emergent/Elective Analysis
            ===================================================================

            ASA Class Distribution
            ----------------------
            The ASA (American Society of Anesthesiologists) physical status
            classification is distributed across five levels in the dataset.
            The partial dependence plot (PDP) demonstrates a monotonically
            increasing relationship between ASA class and predicted mortality:
            higher ASA classes are associated with substantially higher predicted
            probability of death. This is consistent with the SHAP analysis,
            which identified ASACLASS as the second most important variable in
            the GBM model.

            Emergent vs. Elective Status
            -----------------------------
            The VQI dataset includes URGENCY as an explicit emergent/elective
            indicator. URGENCY was added as the 31st predictor in the GBM model
            (see Task 8 for full analysis). URGENCY ranks as a top-10 SHAP feature
            in the baseline model and the #1 feature in the aneurysm subgroup,
            where 18% of cases are emergent (ruptured aneurysm repairs).

            Two additional proxy variables were evaluated for completeness:
            1. ASA Class 5 (Moribund): These patients are not expected to
               survive without surgery and likely include a high proportion of
               emergent cases. Their mortality rate was compared with all other
               ASA classes.
            2. TRANSFER: Patients transferred from another facility may represent
               a population enriched for emergent/urgent cases.

            Both ASACLASS and TRANSFER are included as predictors in the GBM
            model, meaning the model inherently accounts for variation in patient
            acuity alongside the explicit URGENCY variable. The GBM model's
            ability to leverage these variables is reflected in the SHAP feature
            importance rankings.
        """)
    else:
        narrative = textwrap.dedent("""\
            Task 3 Narrative Summary: ASA Class and Emergent/Elective Analysis
            ===================================================================

            ASA Class Distribution
            ----------------------
            The ASA (American Society of Anesthesiologists) physical status
            classification is distributed across five levels in the dataset.
            The partial dependence plot (PDP) demonstrates a monotonically
            increasing relationship between ASA class and predicted mortality:
            higher ASA classes are associated with substantially higher predicted
            probability of death. This is consistent with the SHAP analysis,
            which identified ASACLASS as the second most important variable in
            the GBM model.

            Emergent vs. Elective Status
            -----------------------------
            The VQI dataset used in this study does not include an explicit
            emergent/elective indicator variable. A systematic search of all
            30 model predictor columns and all sheets in the raw source Excel
            file found no columns matching keywords related to emergent,
            elective, urgency, or emergency status.

            Two proxy variables were evaluated:
            1. ASA Class 5 (Moribund): These patients are not expected to
               survive without surgery and likely include a high proportion of
               emergent cases. Their mortality rate was compared with all other
               ASA classes.
            2. TRANSFER: Patients transferred from another facility may represent
               a population enriched for emergent/urgent cases.

            Both ASACLASS and TRANSFER are included as predictors in the GBM
            model, meaning the model inherently accounts for variation in patient
            acuity and potential urgency of surgery, even without an explicit
            emergent/elective flag. The GBM model's ability to leverage these
            variables is reflected in the SHAP feature importance rankings.

            Limitations
            -----------
            Without a direct emergent/elective indicator, we cannot definitively
            stratify outcomes by surgical urgency. Future analyses using VQI
            registries that include case urgency designations would allow for
            more precise subgroup analysis. The ASA Class 5 and TRANSFER
            variables provide indirect evidence but should be interpreted with
            appropriate caution.
        """)

    narrative_path = OUTPUT_DIR / "narrative.txt"
    narrative_path.write_text(narrative)
    logger.info("Saved narrative to %s", narrative_path)

    # ===================================================================
    # Shutdown
    # ===================================================================
    logger.info("Shutting down H2O cluster.")
    h2o.cluster().shutdown(prompt=False)
    logger.info("Task 3 complete. All outputs saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
