"""
Subgroup analysis for the vascular-anemia risk model.

Trains separate H2O AutoML models for clinically meaningful subgroups defined
by PROCEDURE_GROUP, generates SHAP / PDP interpretability artefacts for each,
derives age-stratified hemoglobin thresholds, and produces cross-subgroup
comparison tables.

Outputs
-------
output/task2_subgroups/
    primary_aneurysm/    -- baseline.csv, shap_summary.png, pdp_age.{png,csv},
                            pdp_hemo.{png,csv}, thresholds.csv
    primary_bypass/
    supplemental_open_aaa/
    supplemental_infra/
    supplemental_supra/
    comparison_summary.csv
    auc_comparison.csv
    feature_ranking_comparison.csv

Usage
-----
export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"
uv run python src/analysis_subgroups.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

import h2o
from h2o.automl import H2OAutoML
from h2o.frame import H2OFrame

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "processed" / "merged_vqi_2012_2020.parquet"
METADATA_PATH = REPO_ROOT / "data" / "processed" / "merged_vqi_2012_2020_metadata.csv"
OUTPUT_ROOT = REPO_ROOT / "output" / "task2_subgroups"

TARGET = "DEAD"
SEED = 12345
TRAIN_RATIO = 0.8
MAX_RUNTIME_SECS = 600
NFOLDS = 5
PDP_NBINS = 40
HEMOGLOBIN_CUTOFFS = [0.05, 0.10, 0.20, 0.30]

AGE_BANDS = [
    ("Under 40", -np.inf, 40),
    ("40-49", 40, 50),
    ("50-59", 50, 60),
    ("60-69", 60, 70),
    ("70-79", 70, 80),
    ("80+", 80, np.inf),
]

# ---------------------------------------------------------------------------
# Subgroup definitions
# ---------------------------------------------------------------------------
SUBGROUPS: List[Dict] = [
    # PRIMARY (2-group)
    {
        "label": "primary_aneurysm",
        "display_name": "Aneurysm",
        "group_type": "primary",
        "filter": lambda df: df[df["PROCEDURE_GROUP"] == "OPEN_AAA"],
    },
    {
        "label": "primary_bypass",
        "display_name": "Bypass",
        "group_type": "primary",
        "filter": lambda df: df[df["PROCEDURE_GROUP"].isin(["INFRA", "SUPRA"])],
    },
    # SUPPLEMENTAL (3-group)
    {
        "label": "supplemental_open_aaa",
        "display_name": "OPEN_AAA",
        "group_type": "supplemental",
        "filter": lambda df: df[df["PROCEDURE_GROUP"] == "OPEN_AAA"],
    },
    {
        "label": "supplemental_infra",
        "display_name": "INFRA",
        "group_type": "supplemental",
        "filter": lambda df: df[df["PROCEDURE_GROUP"] == "INFRA"],
    },
    {
        "label": "supplemental_supra",
        "display_name": "SUPRA",
        "group_type": "supplemental",
        "filter": lambda df: df[df["PROCEDURE_GROUP"] == "SUPRA"],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def determine_predictors(metadata: pd.DataFrame, dataset_columns: pd.Index) -> List[str]:
    """Return input columns that exist in the dataset."""
    predictors = metadata.loc[metadata["feature_role"] == "input", "column"].tolist()
    predictors = [c for c in predictors if c in dataset_columns]
    if not predictors:
        raise ValueError("No predictor columns found in the metadata.")
    return predictors


def train_valid_split(
    dataset: pd.DataFrame,
    target_column: str,
    train_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split for classification on *target_column*."""
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    valid_indices: List[int] = []
    for _, group in dataset.groupby(target_column, dropna=False, observed=False):
        group_idx = group.index.to_numpy().copy()
        rng.shuffle(group_idx)
        train_count = int(round(len(group_idx) * train_ratio))
        if len(group_idx) > 1 and train_count >= len(group_idx):
            train_count = len(group_idx) - 1
        valid_count = len(group_idx) - train_count
        if valid_count == 0 and len(group_idx) > 0:
            train_count = max(0, train_count - 1)
        train_indices.extend(group_idx[:train_count])
        valid_indices.extend(group_idx[train_count:])
    return dataset.loc[train_indices].copy(), dataset.loc[valid_indices].copy()


def prepare_h2o_frame(
    df: pd.DataFrame,
    categorical_columns: Sequence[str],
    target: str,
) -> H2OFrame:
    """Convert a pandas DataFrame to an H2OFrame with correct types."""
    frame = h2o.H2OFrame(df)
    for column in categorical_columns:
        if column in frame.columns:
            frame[column] = frame[column].asfactor()
    frame[target] = frame[target].asfactor()
    return frame


def extract_best_gbm(aml: H2OAutoML):
    """Return the best GBM model from the AutoML leaderboard, or None."""
    leaderboard = aml.leaderboard.as_data_frame()
    mask = leaderboard["model_id"].str.startswith("GBM")
    matches = leaderboard.loc[mask, "model_id"]
    if matches.empty:
        return None
    return h2o.get_model(matches.iloc[0])


# ---------------------------------------------------------------------------
# Baseline characteristics
# ---------------------------------------------------------------------------
def compute_baseline(df: pd.DataFrame, subgroup_label: str, out_dir: Path) -> Dict:
    """Compute and save baseline characteristics table."""
    n = len(df)
    age_mean = df["AGE"].mean()
    age_sd = df["AGE"].std()
    male_pct = (df["SEX"].astype(float).eq(1).sum() / n * 100) if "SEX" in df.columns else np.nan
    hemo_mean = df["HEMO"].mean()
    hemo_sd = df["HEMO"].std()
    mortality_rate = df[TARGET].astype(float).mean()

    baseline = pd.DataFrame(
        [
            {
                "subgroup": subgroup_label,
                "n": n,
                "age_mean": round(age_mean, 2),
                "age_sd": round(age_sd, 2),
                "male_pct": round(male_pct, 2),
                "hemo_mean": round(hemo_mean, 2),
                "hemo_sd": round(hemo_sd, 2),
                "mortality_rate": round(mortality_rate, 4),
            }
        ]
    )
    baseline_path = out_dir / "baseline.csv"
    baseline.to_csv(baseline_path, index=False)
    logging.info("Saved baseline characteristics to %s", baseline_path)
    return baseline.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def generate_shap_summary(
    model,
    valid_frame: H2OFrame,
    valid_df: pd.DataFrame,
    predictors: List[str],
    out_dir: Path,
) -> pd.DataFrame:
    """Generate SHAP summary plot and return the mean |SHAP| table."""
    contributions = model.predict_contributions(valid_frame)
    contrib_df = contributions.as_data_frame(use_pandas=True)
    if "BiasTerm" in contrib_df.columns:
        contrib_df = contrib_df.drop(columns=["BiasTerm"])
    available = [c for c in predictors if c in contrib_df.columns]
    if not available:
        logging.warning("No SHAP contribution columns matched predictors; skipping SHAP.")
        return pd.DataFrame()
    contrib_df = contrib_df[available]

    # Mean |SHAP| table
    shap_magnitudes = contrib_df.abs().mean().sort_values(ascending=False).rename("mean_abs_shap")
    shap_table = shap_magnitudes.reset_index().rename(columns={"index": "feature"})

    # Feature values for coloring -- convert categoricals to numeric codes
    feature_values = valid_df[available].copy()
    for col in feature_values.select_dtypes(include="category").columns:
        feature_values[col] = feature_values[col].cat.codes.replace(-1, np.nan)

    shap.summary_plot(
        contrib_df.to_numpy(),
        feature_values,
        feature_names=available,
        show=False,
        plot_type="dot",
        max_display=min(20, len(available)),
    )
    shap_path = out_dir / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(shap_path, dpi=300)
    plt.close()
    logging.info("Saved SHAP summary plot to %s", shap_path)

    return shap_table


# ---------------------------------------------------------------------------
# PDP
# ---------------------------------------------------------------------------
def generate_pdp(
    model,
    frame: H2OFrame,
    column: str,
    data_hist: pd.Series,
    out_dir: Path,
) -> pd.DataFrame:
    """Generate PDP for *column*. Save CSV + PNG. Return PDP dataframe."""
    plots = model.partial_plot(
        frame=frame,
        cols=[column],
        nbins=PDP_NBINS,
        plot=False,
    )
    pdp_df = plots[0].as_data_frame()
    # H2O names the x-column after the feature; normalise to "center"
    x_col = [
        c for c in pdp_df.columns
        if c not in ("mean_response", "stddev_response", "std_error_mean_response")
    ][0]
    pdp_df = pdp_df.rename(columns={x_col: "center"})

    csv_path = out_dir / f"pdp_{column.lower()}.csv"
    pdp_df.to_csv(csv_path, index=False)
    logging.info("Saved PDP table for %s to %s", column, csv_path)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    sns.lineplot(
        data=pdp_df,
        x="center",
        y="mean_response",
        ax=ax1,
        color="#1f77b4",
        linewidth=2,
    )
    ax1.set_xlabel(column)
    ax1.set_ylabel("Mean predicted mortality")
    ax1.set_title(f"Partial Dependence -- {column}")

    ax2 = ax1.twinx()
    sns.histplot(data_hist.dropna(), bins=PDP_NBINS, ax=ax2, alpha=0.3, color="#ff7f0e")
    ax2.set_ylabel("Count")
    fig.tight_layout()
    fig_path = out_dir / f"pdp_{column.lower()}.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logging.info("Saved PDP plot for %s to %s", column, fig_path)

    return pdp_df


# ---------------------------------------------------------------------------
# Hemoglobin thresholds
# ---------------------------------------------------------------------------
def contiguous_interval(values: pd.Series) -> Optional[Tuple[int, int]]:
    """Return (start, end) indices for the longest contiguous True run."""
    idx = np.where(values)[0]
    if len(idx) == 0:
        return None
    starts = [idx[0]]
    ends: List[int] = []
    for i in range(1, len(idx)):
        if idx[i] != idx[i - 1] + 1:
            ends.append(idx[i - 1])
            starts.append(idx[i])
    ends.append(idx[-1])
    lengths = [end - start + 1 for start, end in zip(starts, ends)]
    best = int(np.argmax(lengths))
    return starts[best], ends[best]


def derive_hemoglobin_thresholds(
    model,
    dataset: pd.DataFrame,
    predictors: List[str],
    categorical_columns: Sequence[str],
    out_dir: Path,
) -> pd.DataFrame:
    """Age-stratified hemoglobin threshold derivation via PDP."""
    results: List[Dict] = []
    for label, lower, upper in AGE_BANDS:
        mask = dataset["AGE"].between(lower, upper, inclusive="left")
        group_df = dataset.loc[mask, predictors + [TARGET]]
        if len(group_df) < 10:
            logging.warning(
                "Age band '%s' has only %d rows -- skipping threshold derivation.",
                label,
                len(group_df),
            )
            for cutoff in HEMOGLOBIN_CUTOFFS:
                results.append(
                    {
                        "age_group": label,
                        "cutoff": cutoff,
                        "hemoglobin_min": None,
                        "hemoglobin_max": None,
                        "mean_predicted_mortality": None,
                    }
                )
            continue

        group_h2o = prepare_h2o_frame(group_df, categorical_columns, TARGET)

        try:
            pdp_frames = model.partial_plot(
                frame=group_h2o, cols=["HEMO"], nbins=PDP_NBINS, plot=False
            )
        except Exception as exc:
            logging.warning("PDP failed for age band '%s': %s", label, exc)
            for cutoff in HEMOGLOBIN_CUTOFFS:
                results.append(
                    {
                        "age_group": label,
                        "cutoff": cutoff,
                        "hemoglobin_min": None,
                        "hemoglobin_max": None,
                        "mean_predicted_mortality": None,
                    }
                )
            continue

        pdp_df = pdp_frames[0].as_data_frame()
        x_col = [
            c for c in pdp_df.columns
            if c not in ("mean_response", "stddev_response", "std_error_mean_response")
        ][0]
        pdp_df = pdp_df.rename(columns={x_col: "center"})

        for cutoff in HEMOGLOBIN_CUTOFFS:
            below_mask = pdp_df["mean_response"] < cutoff
            interval = contiguous_interval(below_mask)
            if interval is None:
                results.append(
                    {
                        "age_group": label,
                        "cutoff": cutoff,
                        "hemoglobin_min": None,
                        "hemoglobin_max": None,
                        "mean_predicted_mortality": None,
                    }
                )
                continue
            start, end = interval
            subset = pdp_df.iloc[start : end + 1]
            results.append(
                {
                    "age_group": label,
                    "cutoff": cutoff,
                    "hemoglobin_min": float(subset["center"].iloc[0]),
                    "hemoglobin_max": float(subset["center"].iloc[-1]),
                    "mean_predicted_mortality": float(subset["mean_response"].mean()),
                }
            )

    threshold_table = pd.DataFrame(results)
    threshold_path = out_dir / "thresholds.csv"
    threshold_table.to_csv(threshold_path, index=False)
    logging.info("Saved hemoglobin thresholds to %s", threshold_path)
    return threshold_table


# ---------------------------------------------------------------------------
# Single-subgroup pipeline
# ---------------------------------------------------------------------------
def run_subgroup(
    subgroup_def: Dict,
    full_dataset: pd.DataFrame,
    predictors: List[str],
    categorical_columns: List[str],
) -> Optional[Dict]:
    """
    Full modelling + interpretability pipeline for one subgroup.

    Returns a summary dict or None on failure.
    """
    label = subgroup_def["label"]
    display_name = subgroup_def["display_name"]
    group_type = subgroup_def["group_type"]
    logging.info("=" * 70)
    logging.info("SUBGROUP: %s (%s / %s)", label, group_type, display_name)
    logging.info("=" * 70)

    out_dir = OUTPUT_ROOT / label
    ensure_directory(out_dir)

    # ----- filter to subgroup -----
    sub_df = subgroup_def["filter"](full_dataset).copy()
    working = sub_df[predictors + [TARGET]].dropna(subset=[TARGET]).copy()
    logging.info("Subgroup '%s' has %d rows after dropping missing target.", label, len(working))
    if len(working) < 100:
        logging.warning("Subgroup '%s' too small (%d rows); skipping.", label, len(working))
        return None

    # ----- (a) baseline characteristics -----
    baseline = compute_baseline(sub_df, display_name, out_dir)

    # ----- (b) train / valid split -----
    train_df, valid_df = train_valid_split(working, TARGET, TRAIN_RATIO, SEED)
    logging.info("Train: %d  Valid: %d", len(train_df), len(valid_df))

    # ----- (c) train H2O AutoML and extract best GBM -----
    train_h2o = prepare_h2o_frame(train_df, categorical_columns, TARGET)
    valid_h2o = prepare_h2o_frame(valid_df, categorical_columns, TARGET)

    aml = H2OAutoML(
        max_runtime_secs=MAX_RUNTIME_SECS,
        seed=SEED,
        nfolds=NFOLDS,
        balance_classes=True,
        stopping_metric="AUTO",
        project_name=f"subgroup_{label}",
    )
    aml.train(x=predictors, y=TARGET, training_frame=train_h2o, leaderboard_frame=valid_h2o)

    gbm_model = extract_best_gbm(aml)
    if gbm_model is None:
        logging.warning("No GBM model found for subgroup '%s'; using leader instead.", label)
        gbm_model = aml.leader

    # ----- (d) validation AUC -----
    perf = gbm_model.model_performance(valid_h2o)
    auc_val = perf.auc()
    logging.info("Subgroup '%s' validation AUC: %.4f", label, auc_val)

    # ----- (e) SHAP summary plot -----
    shap_table = generate_shap_summary(gbm_model, valid_h2o, valid_df, predictors, out_dir)

    # ----- (f) PDP for AGE and HEMO -----
    generate_pdp(gbm_model, valid_h2o, "AGE", valid_df["AGE"], out_dir)
    generate_pdp(gbm_model, valid_h2o, "HEMO", valid_df["HEMO"], out_dir)

    # ----- (g) age-stratified hemoglobin thresholds -----
    derive_hemoglobin_thresholds(
        gbm_model, working, predictors, categorical_columns, out_dir
    )

    # ----- build summary -----
    top10_features = (
        shap_table.head(10)["feature"].tolist() if len(shap_table) > 0 else []
    )
    summary = {
        "subgroup": label,
        "display_name": display_name,
        "group_type": group_type,
        "n": len(sub_df),
        "n_working": len(working),
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "mortality_rate": baseline.get("mortality_rate"),
        "auc": auc_val,
        "top10_shap_features": top10_features,
    }
    return summary


# ---------------------------------------------------------------------------
# Cross-subgroup comparisons
# ---------------------------------------------------------------------------
def build_comparisons(summaries: List[Dict]) -> None:
    """Create cross-subgroup comparison tables and a narrative summary."""

    # ---- AUC comparison ----
    auc_rows = [
        {
            "subgroup": s["display_name"],
            "group_type": s["group_type"],
            "n": s["n"],
            "mortality_rate": s["mortality_rate"],
            "validation_auc": s["auc"],
        }
        for s in summaries
    ]
    auc_df = pd.DataFrame(auc_rows)
    auc_path = OUTPUT_ROOT / "auc_comparison.csv"
    auc_df.to_csv(auc_path, index=False)
    logging.info("Saved AUC comparison to %s", auc_path)

    # ---- Feature ranking comparison (top 10 SHAP per subgroup) ----
    ranking_rows: List[Dict] = []
    for s in summaries:
        for rank, feat in enumerate(s["top10_shap_features"], 1):
            ranking_rows.append(
                {
                    "subgroup": s["display_name"],
                    "group_type": s["group_type"],
                    "shap_rank": rank,
                    "feature": feat,
                }
            )
    ranking_df = pd.DataFrame(ranking_rows)
    ranking_path = OUTPUT_ROOT / "feature_ranking_comparison.csv"
    ranking_df.to_csv(ranking_path, index=False)
    logging.info("Saved feature ranking comparison to %s", ranking_path)

    # ---- Brief textual comparison summary ----
    lines: List[str] = []
    lines.append("Cross-subgroup comparison summary")
    lines.append("=" * 50)

    # AUC spread
    aucs = {s["display_name"]: s["auc"] for s in summaries}
    best = max(aucs, key=aucs.get)  # type: ignore[arg-type]
    worst = min(aucs, key=aucs.get)  # type: ignore[arg-type]
    lines.append(
        f"AUC range: {min(aucs.values()):.4f} ({worst}) to "
        f"{max(aucs.values()):.4f} ({best})"
    )

    # Does AGE remain dominant?
    age_ranks: Dict[str, Optional[int]] = {}
    for s in summaries:
        feats = s["top10_shap_features"]
        age_ranks[s["display_name"]] = (feats.index("AGE") + 1) if "AGE" in feats else None
    age_dominant = all(r is not None and r <= 3 for r in age_ranks.values())
    lines.append(
        f"AGE in top-3 SHAP for all subgroups: {'Yes' if age_dominant else 'No'}"
    )
    for name, rank in age_ranks.items():
        lines.append(f"  {name}: AGE rank = {rank}")

    # Do hemoglobin thresholds differ?
    lines.append("")
    lines.append("Hemoglobin threshold comparison (10% cutoff, 60-69 age band):")
    for s in summaries:
        thresh_path = OUTPUT_ROOT / s["subgroup"] / "thresholds.csv"
        if thresh_path.exists():
            tdf = pd.read_csv(thresh_path)
            row = tdf[(tdf["age_group"] == "60-69") & (tdf["cutoff"] == 0.10)]
            if not row.empty and pd.notna(row.iloc[0]["hemoglobin_min"]):
                lines.append(
                    f"  {s['display_name']}: HEMO {row.iloc[0]['hemoglobin_min']:.1f}"
                    f" - {row.iloc[0]['hemoglobin_max']:.1f} g/dL"
                )
            else:
                lines.append(f"  {s['display_name']}: no threshold found")
        else:
            lines.append(f"  {s['display_name']}: threshold file not found")

    summary_text = "\n".join(lines)
    logging.info("\n%s", summary_text)

    # Write as CSV with a single text column for downstream tooling
    comp_df = pd.DataFrame(
        [{"item": line} for line in lines if line.strip()]
    )
    comp_path = OUTPUT_ROOT / "comparison_summary.csv"
    comp_df.to_csv(comp_path, index=False)
    logging.info("Saved comparison summary to %s", comp_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    configure_logging()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    ensure_directory(OUTPUT_ROOT)

    # ---- load data + metadata ----
    logging.info("Loading dataset from %s", DATA_PATH)
    dataset = pd.read_parquet(DATA_PATH)
    logging.info("Dataset shape: %s", dataset.shape)

    metadata = pd.read_csv(METADATA_PATH)
    predictors = determine_predictors(metadata, dataset.columns)
    logging.info("Using %d predictors.", len(predictors))

    # Identify categorical predictors
    categorical_columns = [
        c for c in predictors
        if isinstance(dataset[c].dtype, pd.CategoricalDtype)
    ]
    logging.info("Categorical predictors: %d", len(categorical_columns))

    # ---- initialise H2O once ----
    logging.info("Initialising H2O cluster...")
    h2o.init()

    # ---- run each subgroup ----
    summaries: List[Dict] = []
    for subgroup_def in SUBGROUPS:
        try:
            result = run_subgroup(subgroup_def, dataset, predictors, categorical_columns)
            if result is not None:
                summaries.append(result)
        except Exception:
            logging.exception(
                "Failed to process subgroup '%s'. Continuing with next subgroup.",
                subgroup_def["label"],
            )

    # ---- cross-subgroup comparison ----
    if len(summaries) >= 2:
        try:
            build_comparisons(summaries)
        except Exception:
            logging.exception("Failed to build cross-subgroup comparisons.")
    else:
        logging.warning(
            "Only %d subgroup(s) completed; skipping cross-subgroup comparison.",
            len(summaries),
        )

    # ---- shutdown ----
    logging.info("Shutting down H2O cluster.")
    h2o.cluster().shutdown(prompt=False)
    logging.info("Done.")


if __name__ == "__main__":
    main()
