# Repository Guidelines

## Project Structure & Module Organization
Source code lives in `src/`, with `build_vqi_dataset.py` harmonizing raw VQI spreadsheets into a parquet/CSV pair under `data/processed/`, and `run_h2o_automl.py` orchestrating H2O AutoML training outputs inside `artifacts/h2o_automl/`. Input data belongs in `data/raw/`; derived assets stay under `data/processed/` so regeneration is deterministic. Treat `artifacts/` as disposable experiment output—do not check large models into Git. Keep any exploratory notebooks or diagnostics in `artifacts/notebooks/` (git-ignored by default) to keep the root tidy.

## Build, Test, and Development Commands
- `uv sync` — install Python 3.11 dependencies from `pyproject.toml`/`uv.lock`.
- `uv run python src/build_vqi_dataset.py` — rebuild the harmonized dataset; rerun whenever `data/raw/` changes.
- `uv run python src/run_h2o_automl.py --max-runtime-secs 900 --balance-classes` — launch reference AutoML runs and write leaderboards to `artifacts/h2o_automl/`.
- `uv run python -m pytest tests` — execute the test suite (see below for layout expectations).
Prefer `uv run …` so the resolved environment matches CI.

## Coding Style & Naming Conventions
Target Python 3.11, use type hints (see existing dataclasses) and `Path` objects for filesystem work. Follow Black/PEP8 formatting (4-space indentation, 88-char soft limit) and keep imports grouped stdlib/third-party/local. Name modules and files with snake_case; exported classes should be `PascalCase`, functions `snake_case`, and constants UPPER_SNAKE_CASE. Align logging with the structured format already configured in `run_h2o_automl.py`; avoid bare prints outside notebooks.

## Testing Guidelines
Place unit tests under `tests/` mirroring the `src/` tree (e.g., `tests/test_build_vqi_dataset.py`). Use `pytest` fixtures to stage lightweight sample CSV/Parquet files rather than large real datasets, and verify both schema (columns + dtypes) and behavioral contracts (e.g., metadata contains the declared feature roles). Add regression tests for AutoML configuration helpers so CLI arguments stay wired correctly. Aim for meaningful coverage of preprocessing utilities before introducing new modeling code.

## Commit & Pull Request Guidelines
Write imperative, scoped commit messages (`Add cohort metadata validator`, not `Added` or `fix`). Bundle related changes and reference issue IDs in the subject when applicable. PRs should describe the motivation, summarize functional changes, and list verification steps (commands/tests run). Attach screenshots or table snippets when modifying artifacts that influence clinical reporting. Ensure CI (dataset build + pytest) is green before requesting review.

## Data & Security Notes
Never commit PHI or full VQI exports—keep sensitive spreadsheets confined to `data/raw/` and rely on `.gitignore`. Document any schema changes in the PR body and sanitize sample rows used in tests. Rotate H2O API keys or credentials via environment variables, not plain-text files, and mention required env vars in the PR description so reviewers can reproduce runs safely.
