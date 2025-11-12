"""Train H2O AutoML models for each output variable in the VQI dataset."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from pathlib import Path

import pandas as pd

import h2o
from h2o.automl import H2OAutoML


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "processed" / "merged_vqi_2012_2020.parquet"
DEFAULT_METADATA_PATH = (
    REPO_ROOT / "data" / "processed" / "merged_vqi_2012_2020_metadata.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "h2o_automl"


@dataclass(frozen=True)
class TargetSpec:
    """Configuration for a single response variable."""

    name: str
    problem_type: str  # "classification" or "regression"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train H2O AutoML models for each output variable defined in the "
            "processed dataset metadata."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the harmonised dataset parquet file (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help=(
            "Path to the metadata CSV containing column roles "
            f"(default: {DEFAULT_METADATA_PATH})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to persist AutoML leaderboards and summary (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio for the train/validation split (default: 0.8).",
    )
    parser.add_argument(
        "--max-runtime-secs",
        type=int,
        default=600,
        help="Maximum runtime in seconds for each AutoML run (default: 600).",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Maximum number of models to train per AutoML run (default: unlimited).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for data splitting and AutoML reproducibility (default: 12345).",
    )
    parser.add_argument(
        "--nfolds",
        type=int,
        default=5,
        help="Number of internal cross-validation folds for AutoML (default: 5).",
    )
    parser.add_argument(
        "--categorical-threshold",
        type=int,
        default=10,
        help=(
            "Treat response columns with <= this many unique values as classification "
            "(default: 10)."
        ),
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="Enable class balancing for classification targets.",
    )
    parser.add_argument(
        "--no-shutdown",
        action="store_true",
        help="Skip shutting down the H2O cluster when finishing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Configure the script logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=getattr(logging, level)
    )


def read_metadata(metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    required_columns = {"column", "feature_role"}
    if not required_columns.issubset(metadata.columns):
        missing = ", ".join(sorted(required_columns - set(metadata.columns)))
        raise ValueError(
            f"Metadata file '{metadata_path}' missing required columns: {missing}"
        )
    return metadata


def determine_predictors(metadata: pd.DataFrame, dataset_columns: Iterable[str]) -> List[str]:
    predictors = metadata.loc[metadata["feature_role"] == "input", "column"].tolist()
    predictors = [column for column in predictors if column in dataset_columns]
    if not predictors:
        raise ValueError("No predictor columns found in the metadata.")
    return predictors


def determine_targets(
    metadata: pd.DataFrame, dataset: pd.DataFrame, categorical_threshold: int
) -> List[TargetSpec]:
    targets = []
    for column in metadata.loc[metadata["feature_role"] == "output", "column"]:
        if column not in dataset.columns:
            logging.warning("Target column '%s' not present in dataset. Skipping.", column)
            continue
        series = dataset[column]
        unique = series.dropna().unique()
        problem_type = (
            "classification"
            if series.dtype.kind in ("O", "b", "U")
            or series.dtype.name == "category"
            or len(unique) <= categorical_threshold
            else "regression"
        )
        targets.append(TargetSpec(name=column, problem_type=problem_type))
    if not targets:
        raise ValueError("No output targets discovered from the metadata.")
    return targets


def build_automl(
    args: argparse.Namespace, target: TargetSpec
) -> H2OAutoML:
    kwargs = {
        "max_runtime_secs": args.max_runtime_secs,
        "seed": args.seed,
        "nfolds": args.nfolds,
        "stopping_metric": "AUTO",
        "balance_classes": args.balance_classes if target.problem_type == "classification" else False,
        "project_name": f"h2o_automl_{target.name}",
    }
    if args.max_models is not None:
        kwargs["max_models"] = args.max_models
    return H2OAutoML(**kwargs)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_frame(
    frame, train_ratio: float, seed: int
) -> Tuple[object, object]:
    splits = frame.split_frame(ratios=[train_ratio], seed=seed)
    if len(splits) < 2:
        raise RuntimeError("H2O split_frame did not return the expected train/validation split.")
    return splits[0], splits[1]


def extract_primary_metric(perf, problem_type: str) -> Tuple[str, float | None]:
    if problem_type == "classification":
        auc = getattr(perf, "auc", None)
        if callable(auc):
            value = auc()
            if value is not None:
                return "auc", value
        logloss = getattr(perf, "logloss", None)
        if callable(logloss):
            value = logloss()
            if value is not None:
                return "logloss", value
    else:
        rmse = getattr(perf, "rmse", None)
        if callable(rmse):
            value = rmse()
            if value is not None:
                return "rmse", value
        mae = getattr(perf, "mae", None)
        if callable(mae):
            value = mae()
            if value is not None:
                return "mae", value
    return "metric", None


def run_for_target(
    args: argparse.Namespace,
    h2o_frame,
    predictors: List[str],
    target: TargetSpec,
) -> dict:
    logging.info("=== Training target '%s' (%s) ===", target.name, target.problem_type)
    # Create a view that we can safely modify without affecting subsequent runs.
    frame = h2o_frame[:, :]
    if target.problem_type == "classification":
        frame[target.name] = frame[target.name].asfactor()
    train, valid = split_frame(frame, args.train_ratio, args.seed)
    aml = build_automl(args, target)
    aml.train(x=predictors, y=target.name, training_frame=train, leaderboard_frame=valid)

    leaderboard = aml.leaderboard.as_data_frame()
    leaderboard_path = args.output_dir / f"{target.name}_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    logging.info("Saved leaderboard to %s", leaderboard_path)

    perf = aml.leader.model_performance(valid)
    metric_name, metric_value = extract_primary_metric(perf, target.problem_type)
    summary = {
        "target": target.name,
        "problem_type": target.problem_type,
        "metric": metric_name,
        "metric_value": metric_value,
        "leader_model_id": aml.leader.model_id,
        "leaderboard_path": str(leaderboard_path),
    }
    return summary


def main() -> None:
    args = parse_arguments()
    configure_logging(args.log_level)

    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")
    if not args.metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {args.metadata_path}")

    logging.info("Loading dataset from %s", args.data_path)
    dataset = pd.read_parquet(args.data_path)

    logging.info("Loading metadata from %s", args.metadata_path)
    metadata = read_metadata(args.metadata_path)

    predictors = determine_predictors(metadata, dataset.columns)
    targets = determine_targets(metadata, dataset, args.categorical_threshold)
    ensure_output_dir(args.output_dir)

    logging.info("Initialising H2O cluster...")
    h2o.init()

    # Convert to H2OFrame; ensure categorical columns are preserved.
    h2o_frame = h2o.H2OFrame(dataset)

    summaries = []
    for target in targets:
        try:
            summary = run_for_target(args, h2o_frame, predictors, target)
            summaries.append(summary)
        except Exception as error:  # pragma: no cover - defensive logging
            logging.exception("Failed to train target '%s': %s", target.name, error)

    if summaries:
        summary_path = args.output_dir / "summary.csv"
        pd.DataFrame(summaries).to_csv(summary_path, index=False)
        logging.info("Wrote summary to %s", summary_path)
    else:
        logging.warning("No models were successfully trained.")

    if not args.no_shutdown:
        logging.info("Shutting down H2O cluster.")
        h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()


