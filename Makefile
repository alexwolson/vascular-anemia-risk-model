REPO_ROOT := $(CURDIR)
PYTHON := uv run python

DATA_PATH := data/processed/merged_vqi_2012_2020.parquet
RUN_METADATA := artifacts/h2o_automl/DEAD_run_metadata.json

.PHONY: data validate models interpretability figures tables test clean

data:
	$(PYTHON) src/build_vqi_dataset.py

validate: data
	$(PYTHON) src/validate_dataset_schema.py

models: data
	$(PYTHON) src/run_h2o_automl.py --targets DEAD --max-runtime-secs 900 --balance-classes

interpretability: models
	$(PYTHON) src/generate_interpretability.py --run-metadata-path $(RUN_METADATA)

figures: interpretability

tables: interpretability

test:
	uv run python -m pytest -q

clean:
	rm -f $(RUN_METADATA) artifacts/h2o_automl/*.csv figures/*.png tables/*.csv tables/*.json

