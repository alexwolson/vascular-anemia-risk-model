"""Train H2O AutoML models aligned with the replication checklist."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from pathlib import Path

import numpy as np
import pandas as pd

import h2o
from h2o.automl import H2OAutoML


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "processed" / "merged_vqi_2012_2020.parquet"
DEFAULT_METADATA_PATH = (
    REPO_ROOT / "data" / "processed" / "merged_vqi_2012_2020_metadata.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "h2o_automl"
DEFAULT_MODEL_DIR = REPO_ROOT / "models"


@dataclass(frozen=True)
class TargetSpec:
    """Configuration for a single response variable."""

    name: str
    problem_type: str  # "classification" or "regression"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train H2O AutoML models for selected outcomes using the processed VQI dataset.",
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
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory to persist trained model artifacts (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["DEAD"],
        help="Subset of target columns to train. Use 'all' to include every output (default: DEAD).",
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
        "--disable-stratify",
        action="store_true",
        help="Disable stratified 80/20 split for classification targets.",
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
    metadata: pd.DataFrame,
    dataset: pd.DataFrame,
    categorical_threshold: int,
    selected_targets: Optional[Iterable[str]],
) -> List[TargetSpec]:
    targets: List[TargetSpec] = []
    selected = set(selected_targets) if selected_targets is not None else None
    for column in metadata.loc[metadata["feature_role"] == "output", "column"]:
        if selected is not None and column not in selected:
            continue
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


def build_automl(args: argparse.Namespace, target: TargetSpec) -> H2OAutoML:
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


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def train_valid_split(
    dataset: pd.DataFrame,
    target_column: str,
    train_ratio: float,
    seed: int,
    stratify: bool,
    problem_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if stratify and problem_type == "classification":
        train_indices: List[int] = []
        valid_indices: List[int] = []
        for _, group in dataset.groupby(target_column, dropna=False):
            group_idx = group.index.to_numpy()
            rng.shuffle(group_idx)
            train_count = int(round(len(group_idx) * train_ratio))
            if len(group_idx) > 1 and train_count >= len(group_idx):
                train_count = len(group_idx) - 1
            valid_count = len(group_idx) - train_count
            if valid_count == 0 and len(group_idx) > 0:
                train_count = max(0, train_count - 1)
                valid_count = len(group_idx) - train_count
            train_indices.extend(group_idx[:train_count])
            valid_indices.extend(group_idx[train_count:])
        return dataset.loc[train_indices].copy(), dataset.loc[valid_indices].copy()

    indices = dataset.index.to_numpy()
    rng.shuffle(indices)
    cutoff = int(round(len(indices) * train_ratio))
    cutoff = min(len(indices), max(1, cutoff))
    train_idx = indices[:cutoff]
    valid_idx = indices[cutoff:]
    if len(valid_idx) == 0 and len(indices) > 1:
        train_idx = indices[:-1]
        valid_idx = indices[-1:]
    return dataset.loc[train_idx].copy(), dataset.loc[valid_idx].copy()


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


def first_model_with_prefix(
    leaderboard: pd.DataFrame, prefixes: Tuple[str, ...]
) -> Optional[str]:
    mask = leaderboard["model_id"].str.startswith(prefixes)
    matches = leaderboard.loc[mask, "model_id"]
    if matches.empty:
        return None
    return matches.iloc[0]


def summarise_model_families(leaderboard: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    families: Dict[str, Tuple[str, ...]] = {
        "StackedEnsemble": ("StackedEnsemble",),
        "GBM": ("GBM",),
        "XRT": ("XRT", "DRF"),
        "GLM": ("GLM",),
        "DeepLearning": ("DeepLearning",),
    }
    records = []
    for label, prefixes in families.items():
        mask = leaderboard["model_id"].str.startswith(prefixes)
        if not mask.any():
            continue
        row = leaderboard.loc[mask].iloc[0]
        records.append(
            {
                "family": label,
                "model_id": row["model_id"],
                "metric": row.get(metric_column),
            }
        )
    return pd.DataFrame(records)


def run_for_target(
    args: argparse.Namespace,
    dataset: pd.DataFrame,
    predictors: List[str],
    target: TargetSpec,
) -> dict:
    logging.info("=== Training target '%s' (%s) ===", target.name, target.problem_type)

    columns = predictors + [target.name]
    working = dataset[columns].dropna(subset=[target.name])

    stratify = not args.disable_stratify
    train_df, valid_df = train_valid_split(
        working,
        target.name,
        args.train_ratio,
        args.seed,
        stratify=stratify,
        problem_type=target.problem_type,
    )

    train = h2o.H2OFrame(train_df)
    valid = h2o.H2OFrame(valid_df)

    categorical_predictors = [
        column for column in predictors if pd.api.types.is_categorical_dtype(dataset[column])
    ]
    for column in categorical_predictors:
        train[column] = train[column].asfactor()
        valid[column] = valid[column].asfactor()

    if target.problem_type == "classification":
        train[target.name] = train[target.name].asfactor()
        valid[target.name] = valid[target.name].asfactor()

    aml = build_automl(args, target)
    aml.train(x=predictors, y=target.name, training_frame=train, leaderboard_frame=valid)

    leaderboard = aml.leaderboard.as_data_frame()
    leaderboard_path = args.output_dir / f"{target.name}_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    logging.info("Saved leaderboard to %s", leaderboard_path)

    perf = aml.leader.model_performance(valid)
    metric_name, metric_value = extract_primary_metric(perf, target.problem_type)

    family_summary = summarise_model_families(leaderboard, metric_name)
    family_summary_path = args.output_dir / f"{target.name}_top_models.csv"
    family_summary.to_csv(family_summary_path, index=False)
    logging.info("Saved model family summary to %s", family_summary_path)

    gbm_model_id = first_model_with_prefix(leaderboard, ("GBM",))
    saved_model_path: Optional[Path] = None
    if gbm_model_id:
        target_model_dir = args.model_dir / target.name
        ensure_directory(target_model_dir)
        gbm_model = h2o.get_model(gbm_model_id)
        saved_model_path = Path(
            h2o.save_model(gbm_model, path=str(target_model_dir), force=True)
        )
        logging.info("Saved GBM model to %s", saved_model_path)

    run_metadata_path = args.output_dir / f"{target.name}_run_metadata.json"
    run_metadata = {
        "target": target.name,
        "problem_type": target.problem_type,
        "predictors": predictors,
        "train_rows": int(train.nrows),
        "valid_rows": int(valid.nrows),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "stratified": stratify and target.problem_type == "classification",
        "metric": metric_name,
        "metric_value": metric_value,
        "leader_model_id": aml.leader.model_id,
        "gbm_model_id": gbm_model_id,
        "gbm_model_path": str(saved_model_path) if saved_model_path else None,
        "leaderboard_path": str(leaderboard_path),
        "family_summary_path": str(family_summary_path),
    }
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2))
    logging.info("Saved run metadata to %s", run_metadata_path)

    summary = {
        "target": target.name,
        "problem_type": target.problem_type,
        "metric": metric_name,
        "metric_value": metric_value,
        "leader_model_id": aml.leader.model_id,
        "leaderboard_path": str(leaderboard_path),
        "family_summary_path": str(family_summary_path),
        "gbm_model_path": str(saved_model_path) if saved_model_path else "",
        "run_metadata_path": str(run_metadata_path),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "stratified": stratify and target.problem_type == "classification",
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
    if args.targets and len(args.targets) == 1 and args.targets[0].lower() == "all":
        requested_targets: Optional[Iterable[str]] = None
    else:
        requested_targets = args.targets
    targets = determine_targets(
        metadata,
        dataset,
        args.categorical_threshold,
        selected_targets=requested_targets,
    )

    ensure_directory(args.output_dir)
    ensure_directory(args.model_dir)

    logging.info("Initialising H2O cluster...")
    h2o.init()

    summaries = []
    for target in targets:
        try:
            summary = run_for_target(args, dataset, predictors, target)
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


