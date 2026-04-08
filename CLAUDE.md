# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML research project deriving age-specific preoperative hemoglobin thresholds associated with postoperative mortality in open vascular surgery (VQI registry 2012–2020, N=85,431). Primary model: GBM via H2O AutoML, interpreted with SHAP and partial dependence plots. Under active revision for journal resubmission.

## Commands

```bash
uv sync                                                             # Install/reinstall dependencies (Python 3.11+)

# Pipeline steps (in order)
uv run python src/build_vqi_dataset.py                             # Harmonise raw VQI Excel → parquet
uv run python src/validate_dataset_schema.py                       # Validate schema
uv run python src/run_h2o_automl.py --targets DEAD \
  --max-runtime-secs 900 --balance-classes                         # Train AutoML (long-running ~15 min)
uv run python src/generate_interpretability.py \
  --run-metadata-path artifacts/h2o_automl/DEAD_run_metadata.json  # ROC, SHAP, PDP, threshold tables

# Reviewer-response analyses
export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"               # Required for H2O
uv run python src/analysis_subgroups.py                            # Subgroup models (long-running)
uv run python src/analysis_hgb_by_age.py                           # Hgb distribution by age
uv run python src/analysis_confounding.py                          # SHAP confounding analysis
uv run python src/analysis_table1.py                               # Corrected Table 1
uv run python src/analysis_asa_emergent.py                         # ASA class / urgency
uv run python src/analysis_mortality_timepoint.py                  # Mortality endpoint

# Testing
uv run python -m pytest -q                                         # All tests
uv run python -m pytest tests/test_dataset_schema.py -q            # Single test file
```

Make targets mirror the pipeline: `make data`, `make validate`, `make models`, `make interpretability`, `make test`, `make clean`.

H2O requires Java 17: `export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"` before running any H2O script.

## Architecture

**Data flow:**
```
data/raw/VQI_Database_MTAEdits.xlsx   (PHI — git-ignored; sheets: INFRA_Database, SUPRA_Database, OPEN_AAA_Database_)
  → src/build_vqi_dataset.py          harmonises 3 cohorts, applies sentinel-value cleaning, writes parquet
  → data/processed/merged_vqi_2012_2020.parquet   (N=85,431; HEMO cleaned to [2,20] g/dL)

data/processed/
  → src/run_h2o_automl.py             80/20 stratified split (seed=12345), H2O AutoML, saves best GBM
  → artifacts/h2o_automl/DEAD_run_metadata.json   (predictor list, GBM model path, AUC)
  → models/DEAD/<model_id>            (serialised H2O GBM — git-ignored)

artifacts/h2o_automl/DEAD_run_metadata.json
  → src/generate_interpretability.py  ROC curve, SHAP summary, PDPs for AGE and HEMO,
                                      age-stratified Hgb threshold table
  → figures/   (PNG)   tables/   (CSV/JSON)
```

**Reviewer-response analysis scripts** (`src/analysis_*.py`) are standalone and write to `output/task*/`. They depend on the processed parquet and, where H2O is needed, on models serialised under `models/`.

**`artifacts/` is disposable experiment output.** `figures/` and `tables/` are the publication-ready finals.

## Key Internals

**Sentinel-value cleaning** (`build_vqi_dataset.py:clean_sentinel_values`): HEMO outside [2, 20] g/dL → NaN; PREOP_CREAT > 25 → NaN; zero HTCM/WEIGHT_KG → NaN; TXFUSION > 50 → NaN. The current parquet already has these applied.

**Hemoglobin threshold derivation** (`generate_interpretability.py:derive_hemoglobin_thresholds`): groups the full dataset by age band, runs H2O `partial_plot` on each group (nbins=40), then identifies the longest contiguous Hgb interval where predicted mortality < cutoff. Because the parquet is already clean, PDP grid points are bounded by the observed (clinically valid) Hgb range within each age group.

**Train/validation split** (`run_h2o_automl.py:train_valid_split`): stratified 80/20 by outcome class, seed=12345. The split is reproduced identically in `generate_interpretability.py` so PDPs and SHAP values are computed only on validation rows.

**`generate_interpretability.py` imports `train_valid_split` directly from `run_h2o_automl`** — both scripts must be run from `src/` or with `src/` on the path (the Makefile handles this).

## Coding Style

- Python 3.11+, type hints required, `Path` objects for filesystem work
- Black/PEP8: 4-space indent, 88-char soft limit, grouped imports (stdlib / third-party / local)
- Structured logging via `logging` module — no bare `print()` outside scripts that are explicitly written as pipeline scripts (where `print()` is acceptable)
- Names: `snake_case` modules/functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants

## Testing

Tests in `tests/` mirror `src/` structure. Use pytest fixtures with lightweight sample CSV/Parquet files — no real VQI data. Tests verify column schemas + dtypes, run metadata structure, and train/validation split contracts.

Run a single test: `uv run python -m pytest tests/test_train_valid_split.py -q`

## Data & PHI

Never commit PHI or VQI exports. `data/raw/` is git-ignored. Sentinel cleaning rules are in `build_vqi_dataset.py:clean_sentinel_values` — update there if cleaning logic changes, then rebuild the parquet.
