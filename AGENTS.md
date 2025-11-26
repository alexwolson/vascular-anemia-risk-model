# Repository Guidelines

## Project Structure & Module Organization
Source code lives in `src/`:
- `build_vqi_dataset.py`: Harmonizes raw VQI spreadsheets into a parquet/CSV pair under `data/processed/`.
- `validate_dataset_schema.py`: Verifies the harmonized dataset against the replication checklist and writes a report to `data/processed/`.
- `run_h2o_automl.py`: Orchestrates H2O AutoML training, outputting models and metadata to `artifacts/h2o_automl/`.
- `generate_interpretability.py`: Produces ROC curves, SHAP plots, and partial dependence plots for trained models, saving to `figures/` and `tables/`.

Input data belongs in `data/raw/`; derived assets stay under `data/processed/`. Treat `artifacts/` as disposable experiment output. `figures/` and `tables/` hold final publication-ready assets. Keep any exploratory notebooks in `artifacts/notebooks/` (git-ignored by default) or `notebooks/` to keep the root tidy.

## Build, Test, and Development Commands
- `uv sync` — install Python 3.11 dependencies from `pyproject.toml`/`uv.lock`.
- `uv run python src/build_vqi_dataset.py` — rebuild the harmonized dataset.
- `uv run python src/validate_dataset_schema.py` — check dataset integrity and generate a schema report.
- `uv run python src/run_h2o_automl.py --max-runtime-secs 900 --balance-classes` — launch reference AutoML runs.
- `uv run python src/generate_interpretability.py` — generate plots and tables from the latest run metadata.
- `uv run python -m pytest tests` — execute the test suite.
Prefer `uv run …` so the resolved environment matches CI.

## Coding Style & Naming Conventions
Target Python 3.11, use type hints (see existing dataclasses) and `Path` objects for filesystem work. Follow Black/PEP8 formatting (4-space indentation, 88-char soft limit) and keep imports grouped stdlib/third-party/local. Name modules and files with snake_case; exported classes should be `PascalCase`, functions `snake_case`, and constants UPPER_SNAKE_CASE. Align logging with the structured format already configured in `run_h2o_automl.py`; avoid bare prints outside notebooks.

## Testing Guidelines
Place unit tests under `tests/` mirroring the `src/` tree (e.g., `tests/test_build_vqi_dataset.py`). Use `pytest` fixtures to stage lightweight sample CSV/Parquet files rather than large real datasets, and verify both schema (columns + dtypes) and behavioral contracts (e.g., metadata contains the declared feature roles). Add regression tests for AutoML configuration helpers so CLI arguments stay wired correctly. Aim for meaningful coverage of preprocessing utilities before introducing new modeling code.

## Commit & Pull Request Guidelines
Write imperative, scoped commit messages (`Add cohort metadata validator`, not `Added` or `fix`). Bundle related changes and reference issue IDs in the subject when applicable. PRs should describe the motivation, summarize functional changes, and list verification steps (commands/tests run). Attach screenshots or table snippets when modifying artifacts that influence clinical reporting. Ensure CI (dataset build + pytest) is green before requesting review.

## Data & Security Notes
Never commit PHI or full VQI exports—keep sensitive spreadsheets confined to `data/raw/` and rely on `.gitignore`. Document any schema changes in the PR body and sanitize sample rows used in tests. Rotate H2O API keys or credentials via environment variables, not plain-text files, and mention required env vars in the PR description so reviewers can reproduce runs safely.
