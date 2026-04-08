"""
Confounding analysis: assess whether HEMO's predictive contribution to
mortality is independent of comorbidity status.

Produces:
  - Point-biserial correlations between HEMO and key comorbidities
  - SHAP dependence plots for HEMO coloured by each comorbidity
  - Narrative summary

Usage:
    export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"
    uv run python src/analysis_confounding.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

import h2o

from run_h2o_automl import train_valid_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "processed" / "merged_vqi_2012_2020.parquet"
METADATA_PATH = REPO_ROOT / "artifacts" / "h2o_automl" / "DEAD_run_metadata.json"
OUTPUT_DIR = REPO_ROOT / "output" / "task6_confounding"

# Comorbidities used in the model
COMORBIDITY_FEATURES = [
    "DIABETES",
    "PRIOR_CHF",
    "COPD",
    "DIALYSIS",
    "HTN",
    "PRIOR_BYPASS",
    "PRIOR_CABG",
    "PRIOR_PCI",
]

# Subset used for SHAP dependence plots
SHAP_COMORBIDITIES = ["DIABETES", "DIALYSIS", "COPD", "PRIOR_CHF"]

# Subset used for point-biserial correlation with HEMO
CORRELATION_COMORBIDITIES = ["DIABETES", "PRIOR_CHF", "COPD", "DIALYSIS", "HTN"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )


def load_run_metadata(path: Path) -> dict:
    return json.loads(path.read_text())


def binarize_comorbidity(series: pd.Series) -> pd.Series:
    """Convert a comorbidity column to binary (0 = None/absent, 1 = any presence).

    Works for columns that are already 0/1, categorical with a 'None' level,
    or multi-level ordinal (e.g. DIABETES: None, Non-insulin, Insulin).
    """
    if hasattr(series, "cat"):
        codes = series.cat.codes  # -1 for NaN
        categories = series.cat.categories
        # Find the index of the "None" or baseline category
        none_indices = [
            i for i, c in enumerate(categories)
            if str(c).strip().lower() in ("none", "no", "0", "0.0")
        ]
        if none_indices:
            return (codes.isin(none_indices)).astype(int).replace({1: 0, 0: 1})
        # If no "None" category, treat code 0 as baseline
        return (codes > 0).astype(int)
    # Numeric column: treat 0 as absent, >0 as present
    return (series.fillna(0) > 0).astype(int)


def prepare_h2o_frame(
    df: pd.DataFrame,
    categorical_columns,
    target: str,
    problem_type: str,
):
    """Convert a pandas DataFrame into an H2OFrame with proper column types."""
    frame = h2o.H2OFrame(df)
    for column in categorical_columns:
        if column in frame.columns:
            frame[column] = frame[column].asfactor()
    if problem_type == "classification":
        frame[target] = frame[target].asfactor()
    return frame


# ---------------------------------------------------------------------------
# Step 1: Correlation table
# ---------------------------------------------------------------------------


def compute_correlation_table(valid_df: pd.DataFrame) -> pd.DataFrame:
    """Compute point-biserial correlation between HEMO and each comorbidity."""
    records = []
    hemo = valid_df["HEMO"]

    for comorbidity in CORRELATION_COMORBIDITIES:
        binary = binarize_comorbidity(valid_df[comorbidity])
        # Drop rows where either value is missing
        mask = hemo.notna() & binary.notna()
        if mask.sum() < 10:
            logging.warning(
                "Too few valid rows for %s correlation. Skipping.", comorbidity
            )
            continue
        corr, pvalue = pointbiserialr(binary[mask], hemo[mask])
        records.append(
            {
                "comorbidity": comorbidity,
                "point_biserial_r": round(corr, 4),
                "p_value": pvalue,
                "n": int(mask.sum()),
                "prevalence": round(binary[mask].mean(), 4),
                "mean_hemo_absent": round(hemo[mask][binary[mask] == 0].mean(), 2),
                "mean_hemo_present": round(hemo[mask][binary[mask] == 1].mean(), 2),
            }
        )

    corr_df = pd.DataFrame(records)
    return corr_df


# ---------------------------------------------------------------------------
# Step 2: SHAP dependence plots
# ---------------------------------------------------------------------------


def plot_shap_dependence(
    hemo_values: np.ndarray,
    hemo_shap: np.ndarray,
    comorbidity_binary: np.ndarray,
    comorbidity_name: str,
    output_path: Path,
) -> None:
    """Scatter plot: HEMO value vs. HEMO SHAP contribution, coloured by comorbidity."""
    fig, ax = plt.subplots(figsize=(8, 5))

    absent_mask = comorbidity_binary == 0
    present_mask = comorbidity_binary == 1

    ax.scatter(
        hemo_values[absent_mask],
        hemo_shap[absent_mask],
        c="#1f77b4",
        alpha=0.15,
        s=6,
        label=f"{comorbidity_name} absent",
        rasterized=True,
    )
    ax.scatter(
        hemo_values[present_mask],
        hemo_shap[present_mask],
        c="#d62728",
        alpha=0.15,
        s=6,
        label=f"{comorbidity_name} present",
        rasterized=True,
    )

    ax.set_xlabel("Hemoglobin (HEMO) value (g/dL)")
    ax.set_ylabel("SHAP contribution of HEMO")
    ax.set_title(f"HEMO SHAP dependence coloured by {comorbidity_name}")
    ax.legend(loc="upper right", framealpha=0.9, markerscale=3)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logging.info("Saved SHAP dependence plot to %s", output_path)


# ---------------------------------------------------------------------------
# Step 3: Narrative
# ---------------------------------------------------------------------------

NARRATIVE_TEXT = """\
Confounding Analysis: HEMO and Comorbidity Interactions
=======================================================

Objective
---------
This analysis examines whether the hemoglobin (HEMO) feature's contribution
to the GBM mortality prediction model is independent of comorbidity status,
or whether its apparent predictive value is confounded by correlated
comorbidities.

Approach
--------
The GBM model includes comorbidity features (DIABETES, PRIOR_CHF, COPD,
DIALYSIS, HTN, PRIOR_BYPASS, PRIOR_CABG, PRIOR_PCI) as separate predictors
alongside HEMO. Because the model learns conditional relationships, the SHAP
values for HEMO represent its marginal contribution to the predicted
log-odds of mortality *conditional on all other features*, including
comorbidities. This means that to the extent the model has learned the
comorbidity effects independently, the HEMO SHAP values already account
for confounding.

Analysis Components
-------------------
1. Point-biserial correlations (hemo_comorbidity_correlations.csv):
   Quantifies the raw linear association between HEMO and each binary
   comorbidity indicator. Moderate-to-strong correlations suggest that
   patients with certain comorbidities tend to have systematically lower
   (or higher) hemoglobin levels, which could confound naive analyses.

2. SHAP dependence plots (shap_hemo_by_*.png):
   For each of four key comorbidities (DIABETES, DIALYSIS, COPD,
   PRIOR_CHF), a scatter plot shows:
     - x-axis: the patient's HEMO value
     - y-axis: the model's SHAP contribution attributed to HEMO
     - colour: whether the comorbidity is present or absent

   If the HEMO-SHAP relationship is similar regardless of comorbidity
   status (i.e., the red and blue point clouds overlap substantially),
   this supports the interpretation that HEMO's predictive value is
   largely independent of that comorbidity. Conversely, if the point
   clouds diverge, the model may be capturing an interaction between
   HEMO and the comorbidity.

Interpretation
--------------
Together, the correlation table and SHAP dependence plots allow the
reviewer to assess:

  (a) Whether low hemoglobin is merely a marker for sicker patients
      (high comorbidity burden), or whether it carries independent
      predictive information for postoperative mortality.

  (b) Whether the GBM model treats HEMO differently for patients with
      vs. without specific comorbidities (effect modification).

Because SHAP values are computed conditional on all other model features,
a consistent HEMO-SHAP curve across comorbidity strata provides evidence
that the model has successfully disentangled the HEMO effect from
comorbidity-driven confounding.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    configure_logging()

    # --- Validate paths ---
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Run metadata not found: {METADATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load run metadata ---
    run_meta = load_run_metadata(METADATA_PATH)
    target = run_meta["target"]
    predictors = run_meta["predictors"]
    problem_type = run_meta["problem_type"]
    train_ratio = float(run_meta.get("train_ratio", 0.8))
    seed = int(run_meta.get("seed", 12345))
    stratified = bool(run_meta.get("stratified", True))

    # --- Resolve model path from run metadata ---
    model_path_str = run_meta.get("gbm_model_path")
    if not model_path_str:
        raise FileNotFoundError(
            "No 'gbm_model_path' in run metadata. Re-run run_h2o_automl.py first."
        )
    MODEL_PATH = Path(model_path_str)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"GBM model not found: {MODEL_PATH}")

    # --- Load dataset ---
    logging.info("Loading dataset from %s", DATA_PATH)
    dataset = pd.read_parquet(DATA_PATH)
    working = dataset[predictors + [target]].dropna(subset=[target])

    # --- Recreate validation split ---
    logging.info("Recreating train/valid split (seed=%d, ratio=%.2f, stratified=%s)",
                 seed, train_ratio, stratified)
    _train_df, valid_df = train_valid_split(
        working,
        target,
        train_ratio,
        seed,
        stratify=stratified,
        problem_type=problem_type,
    )
    logging.info("Validation set: %d rows", len(valid_df))

    # --- Initialise H2O and load model ---
    logging.info("Initialising H2O cluster...")
    h2o.init()

    logging.info("Loading GBM model from %s", MODEL_PATH)
    model = h2o.load_model(str(MODEL_PATH))

    # --- Build H2O frame ---
    categorical_columns = working.select_dtypes(include="category").columns
    valid_h2o = prepare_h2o_frame(
        valid_df,
        categorical_columns=categorical_columns,
        target=target,
        problem_type=problem_type,
    )

    # =========================================================================
    # Step 1: Correlation table
    # =========================================================================
    logging.info("Computing point-biserial correlations between HEMO and comorbidities...")
    corr_df = compute_correlation_table(valid_df)
    corr_path = OUTPUT_DIR / "hemo_comorbidity_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    logging.info("Saved correlation table to %s", corr_path)
    logging.info("\n%s", corr_df.to_string(index=False))

    # =========================================================================
    # Step 2: SHAP dependence plots
    # =========================================================================
    logging.info("Computing SHAP contributions on validation set...")
    contributions = model.predict_contributions(valid_h2o)
    contrib_df = contributions.as_data_frame(use_pandas=True)

    if "HEMO" not in contrib_df.columns:
        raise ValueError(
            "HEMO not found in SHAP contribution columns. "
            f"Available: {sorted(contrib_df.columns)}"
        )

    hemo_shap = contrib_df["HEMO"].to_numpy()
    hemo_values = valid_df["HEMO"].to_numpy()

    for comorbidity in SHAP_COMORBIDITIES:
        logging.info("Generating SHAP dependence plot for HEMO by %s...", comorbidity)
        binary = binarize_comorbidity(valid_df[comorbidity]).to_numpy()

        plot_path = OUTPUT_DIR / f"shap_hemo_by_{comorbidity.lower()}.png"
        plot_shap_dependence(
            hemo_values=hemo_values,
            hemo_shap=hemo_shap,
            comorbidity_binary=binary,
            comorbidity_name=comorbidity,
            output_path=plot_path,
        )

    # =========================================================================
    # Step 3: Narrative
    # =========================================================================
    narrative_path = OUTPUT_DIR / "narrative.txt"
    narrative_path.write_text(NARRATIVE_TEXT)
    logging.info("Saved narrative to %s", narrative_path)

    # --- Shut down H2O ---
    logging.info("Shutting down H2O cluster.")
    h2o.cluster().shutdown(prompt=False)

    logging.info("Confounding analysis complete. Outputs in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
