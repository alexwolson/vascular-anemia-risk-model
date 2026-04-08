"""
Generate interpretability artifacts (ROC, SHAP, PDP, hemoglobin thresholds)
for the AutoML-trained GBM model.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

import h2o
from h2o.frame import H2OFrame

from run_h2o_automl import train_valid_split


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "processed" / "merged_vqi_2012_2020.parquet"
DEFAULT_METADATA_PATH = REPO_ROOT / "artifacts" / "h2o_automl" / "DEAD_run_metadata.json"
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures"
DEFAULT_TABLES_DIR = REPO_ROOT / "tables"
AGE_BANDS = [
    ("Under 40", -np.inf, 40),
    ("40-49", 40, 50),
    ("50-59", 50, 60),
    ("60-69", 60, 70),
    ("70-79", 70, 80),
    ("80+", 80, np.inf),
]


@dataclass(frozen=True)
class RunMetadata:
    target: str
    problem_type: str
    predictors: List[str]
    train_ratio: float
    seed: int
    stratified: bool
    metric: str
    metric_value: float
    gbm_model_path: Optional[Path]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate interpretability outputs for the trained GBM model."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Parquet dataset used for modelling (default: harmonised dataset).",
    )
    parser.add_argument(
        "--run-metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="JSON metadata emitted by run_h2o_automl.py (default: DEAD run metadata).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Override path to the GBM model. Defaults to the path recorded in run metadata.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory to write figures (default: figures/).",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
        help="Directory to write tables (default: tables/).",
    )
    parser.add_argument(
        "--hemoglobin-cutoffs",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30],
        help="Mortality cutoffs for age-stratified threshold derivation (default: 5%, 10%, 20%, 30%).",
    )
    parser.add_argument(
        "--pdp-nbins",
        type=int,
        default=40,
        help="Number of bins for partial dependence calculations (default: 40).",
    )
    parser.add_argument(
        "--no-shutdown",
        action="store_true",
        help="Skip shutting down the H2O cluster after outputs are generated.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=getattr(logging, level)
    )


def load_run_metadata(path: Path) -> RunMetadata:
    payload = json.loads(path.read_text())
    return RunMetadata(
        target=payload["target"],
        problem_type=payload["problem_type"],
        predictors=list(payload["predictors"]),
        train_ratio=float(payload.get("train_ratio", 0.8)),
        seed=int(payload.get("seed", 12345)),
        stratified=bool(payload.get("stratified", True)),
        metric=str(payload["metric"]),
        metric_value=float(payload["metric_value"]),
        gbm_model_path=Path(payload["gbm_model_path"]) if payload.get("gbm_model_path") else None,
    )


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_h2o_frame(
    df: pd.DataFrame,
    categorical_columns: Iterable[str],
    target: str,
    problem_type: str,
) -> H2OFrame:
    frame = h2o.H2OFrame(df)
    for column in categorical_columns:
        if column in frame.columns:
            frame[column] = frame[column].asfactor()
    if problem_type == "classification":
        frame[target] = frame[target].asfactor()
    return frame


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    path: Path,
) -> Dict[str, float]:
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — GBM (Validation Set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return {"auc": float(roc_auc)}


def generate_shap_outputs(
    model,
    valid_frame: H2OFrame,
    valid_df: pd.DataFrame,
    predictors: Sequence[str],
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    contributions = model.predict_contributions(valid_frame)
    contrib_df = contributions.as_data_frame(use_pandas=True)
    if "BiasTerm" in contrib_df.columns:
        contrib_df = contrib_df.drop(columns=["BiasTerm"])
    available_features = [column for column in predictors if column in contrib_df.columns]
    if not available_features:
        logging.warning("No SHAP contribution columns matched the predictor list; skipping SHAP outputs.")
        return
    contrib_df = contrib_df[available_features]

    shap_magnitudes = (
        contrib_df.abs().mean().sort_values(ascending=False).rename("mean_abs_shap")
    )
    shap_table = shap_magnitudes.reset_index().rename(columns={"index": "feature"})
    shap_table_path = tables_dir / "dead_gbm_shap_summary.csv"
    shap_table.to_csv(shap_table_path, index=False)
    logging.info("Saved SHAP summary table to %s", shap_table_path)

    feature_values = valid_df[predictors].copy()
    for column in feature_values.select_dtypes(include="category").columns:
        feature_values[column] = feature_values[column].cat.codes.replace(-1, np.nan)

    shap.summary_plot(
        contrib_df.to_numpy(),
        feature_values[available_features],
        feature_names=available_features,
        show=False,
        plot_type="dot",
        max_display=min(20, len(available_features)),
    )
    shap_fig_path = figures_dir / "dead_gbm_shap_summary.png"
    plt.tight_layout()
    plt.savefig(shap_fig_path, dpi=300)
    plt.close()
    logging.info("Saved SHAP summary plot to %s", shap_fig_path)


def partial_dependence_plot(
    model,
    data: H2OFrame,
    column: str,
    nbins: int,
    figures_dir: Path,
    tables_dir: Path,
    data_hist: pd.Series,
) -> pd.DataFrame:
    plots = model.partial_plot(
        frame=data,
        cols=[column],
        nbins=nbins,
        plot=False,
    )
    pdp_df = plots[0].as_data_frame()
    # H2O names the x-column after the feature (lowercase); normalise for downstream code
    x_col = [c for c in pdp_df.columns if c not in ("mean_response", "stddev_response", "std_error_mean_response")][0]
    pdp_df = pdp_df.rename(columns={x_col: "center"})
    pdp_table_path = tables_dir / f"dead_gbm_pdp_{column.lower()}.csv"
    pdp_df.to_csv(pdp_table_path, index=False)
    logging.info("Saved PDP table for %s to %s", column, pdp_table_path)

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
    ax1.set_title(f"Partial Dependence — {column}")

    ax2 = ax1.twinx()
    sns.histplot(
        data_hist.dropna(),
        bins=nbins,
        ax=ax2,
        alpha=0.3,
        color="#ff7f0e",
    )
    ax2.set_ylabel("Count")
    fig.tight_layout()
    figure_path = figures_dir / f"dead_gbm_pdp_{column.lower()}.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    logging.info("Saved PDP plot for %s to %s", column, figure_path)

    return pdp_df


def contiguous_interval(values: pd.Series) -> Optional[Tuple[int, int]]:
    """Return the start/end indices (inclusive) for the longest contiguous True run."""
    idx = np.where(values)[0]
    if len(idx) == 0:
        return None
    starts = [idx[0]]
    ends = []
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
    predictors: Sequence[str],
    target: str,
    run_meta: RunMetadata,
    nbins: int,
    cutoffs: Sequence[float],
    tables_dir: Path,
) -> None:
    results = []
    for label, lower, upper in AGE_BANDS:
        mask = dataset["AGE"].between(lower, upper, inclusive="left")
        group_df = dataset.loc[mask, list(predictors) + [target]]
        if group_df.empty:
            for cutoff in cutoffs:
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

        group_h2o = prepare_h2o_frame(
            group_df,
            categorical_columns=group_df.select_dtypes(include="category").columns,
            target=target,
            problem_type=run_meta.problem_type,
        )

        pdp_frames = model.partial_plot(
            frame=group_h2o, cols=["HEMO"], nbins=nbins, plot=False
        )
        pdp_df = pdp_frames[0].as_data_frame()
        x_col = [c for c in pdp_df.columns if c not in ("mean_response", "stddev_response", "std_error_mean_response")][0]
        pdp_df = pdp_df.rename(columns={x_col: "center"})

        for cutoff in cutoffs:
            mask = pdp_df["mean_response"] < cutoff
            interval = contiguous_interval(mask)
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
    threshold_path = tables_dir / "dead_gbm_hemoglobin_thresholds.csv"
    threshold_table.to_csv(threshold_path, index=False)
    logging.info("Saved hemoglobin thresholds to %s", threshold_path)


def main() -> None:
    args = parse_arguments()
    configure_logging(args.log_level)

    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")
    if not args.run_metadata_path.exists():
        raise FileNotFoundError(f"Run metadata not found: {args.run_metadata_path}")

    run_meta = load_run_metadata(args.run_metadata_path)
    model_path = args.model_path or run_meta.gbm_model_path
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(
            "GBM model path not available. Provide --model-path or ensure run metadata "
            "contains 'gbm_model_path'."
        )

    ensure_directory(args.figures_dir)
    ensure_directory(args.tables_dir)

    logging.info("Loading dataset from %s", args.data_path)
    dataset = pd.read_parquet(args.data_path)

    working = dataset[run_meta.predictors + [run_meta.target]].dropna(subset=[run_meta.target])
    train_df, valid_df = train_valid_split(
        working,
        run_meta.target,
        run_meta.train_ratio,
        run_meta.seed,
        stratify=run_meta.stratified,
        problem_type=run_meta.problem_type,
    )

    logging.info("Initialising H2O cluster...")
    h2o.init()

    logging.info("Loading GBM model from %s", model_path)
    model = h2o.load_model(str(model_path))

    categorical_predictors = working.select_dtypes(include="category").columns
    valid_h2o = prepare_h2o_frame(
        valid_df,
        categorical_columns=categorical_predictors,
        target=run_meta.target,
        problem_type=run_meta.problem_type,
    )

    predictions = model.predict(valid_h2o).as_data_frame(use_pandas=True)
    if run_meta.problem_type == "classification":
        prob_column = [col for col in predictions.columns if col.startswith("p")][-1]
        y_score = predictions[prob_column].to_numpy()
        y_true = pd.to_numeric(valid_df[run_meta.target], errors="coerce").to_numpy()
        roc_fig_path = args.figures_dir / "dead_gbm_roc_curve.png"
        roc_stats = plot_roc_curve(y_true, y_score, roc_fig_path)
        roc_table_path = args.tables_dir / "dead_gbm_roc_metrics.json"
        roc_table_path.write_text(json.dumps(roc_stats, indent=2))
        logging.info("Saved ROC metrics to %s", roc_table_path)

    generate_shap_outputs(
        model,
        valid_h2o,
        valid_df,
        run_meta.predictors,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
    )

    _ = partial_dependence_plot(
        model,
        valid_h2o,
        "AGE",
        args.pdp_nbins,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
        data_hist=valid_df["AGE"],
    )
    _ = partial_dependence_plot(
        model,
        valid_h2o,
        "HEMO",
        args.pdp_nbins,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
        data_hist=valid_df["HEMO"],
    )

    derive_hemoglobin_thresholds(
        model,
        dataset[run_meta.predictors + [run_meta.target]],
        run_meta.predictors,
        run_meta.target,
        run_meta,
        nbins=args.pdp_nbins,
        cutoffs=args.hemoglobin_cutoffs,
        tables_dir=args.tables_dir,
    )

    if not args.no_shutdown:
        logging.info("Shutting down H2O cluster.")
        h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()


