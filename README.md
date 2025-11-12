# Vascular Anemia Risk Model Utilities

This repository contains utilities for working with the harmonised VQI dataset and training risk models.

## H2O AutoML Training

The script at `src/run_h2o_automl.py` fits one H2O AutoML model per output variable defined in the processed metadata and writes leaderboards plus a summary CSV to an artifacts directory.

Run the workflow with `uv run`:

```
uv run python src/run_h2o_automl.py \
  --data-path ./data/processed/merged_vqi_2012_2020.parquet \
  --metadata-path ./data/processed/merged_vqi_2012_2020_metadata.csv \
  --output-dir ./artifacts/h2o_automl
```

Key options:

- `--max-runtime-secs`: time budget per AutoML run (default 600).
- `--max-models`: cap the number of models per run.
- `--train-ratio`: adjust the train/validation split (default 0.8).
- `--balance-classes`: enable class balancing for classification targets.
- `--no-shutdown`: keep the H2O cluster running after the script finishes.

Each AutoML run writes `<TARGET>_leaderboard.csv` and appends a row to `summary.csv` inside the chosen output directory. Classification targets are automatically detected when the response column has a categorical dtype or fewer than the configurable number of distinct values (default 10).


