# Preoperative Hemoglobin ML Replication

This repository rebuilds the analysis from _“Preoperative hemoglobin in vascular surgery”_, with the goal of deriving age-specific hemoglobin thresholds that minimize postoperative mortality for open vascular procedures.

## Study Overview

- **Study aim**: Use machine learning to estimate age-specific preoperative hemoglobin ranges associated with <10% predicted mortality following open vascular surgery.
- **Data source**: Vascular Quality Initiative (VQI) registry extracts (2012–2020) for infrainguinal bypass, suprainguinal bypass, and open abdominal aortic aneurysm (AAA) repair.
- **Primary outcome**: All-cause mortality during registry follow-up (`DEAD`).
- **Primary model**: Gradient Boosting Machine (GBM) selected via H2O AutoML, interpreted with SHAP values and partial dependence plots.

## Repository Layout

```
artifacts/              # Experiment outputs (leaderboards, notebooks, figures)
config/                 # Environment manifests and model/data configuration
data/
  raw/                  # PHI/PII-restricted VQI extracts (not versioned)
  processed/            # Harmonized analytic dataset + metadata
figures/                # Generated plots (ROC, SHAP, PDP)
models/                 # Serialized H2O models (ignored in git)
notebooks/              # Exploratory and replication notebooks
src/                    # Reproducible pipeline scripts
tables/                 # Published table exports (CSV/LaTeX)
```

## Quick Start

1. Install the locked environment:

   ```
   uv sync
   ```

2. Rebuild the harmonized dataset (expects VQI Excel extracts under `data/raw/`):

   ```
   uv run python src/build_vqi_dataset.py
   ```

3. Train AutoML models for the mortality endpoint:

   ```
   uv run python src/run_h2o_automl.py --max-runtime-secs 900 --balance-classes
   ```

4. (Upcoming) Generate interpretability outputs and tables via `make figures` / `make tables`.

## Reproducibility Status

- ✅ Dataset harmonization script covering the 30 preoperative and 14 postoperative variables.
- ⚠️ AutoML pipeline and interpretability outputs under active development (see `align.plan.md`).
- ⏳ SHAP, PDP, and age-stratified threshold workflows pending implementation.

Please see `ML_Hgb_replication_checklist.md` for the full replication specification.
