from pathlib import Path
import json

import pytest

RUN_METADATA = Path("artifacts/h2o_automl/DEAD_run_metadata.json")


@pytest.mark.parametrize("lower,upper", [(0.75, 0.81)])
def test_dead_metric_within_bounds(lower: float, upper: float) -> None:
    if not RUN_METADATA.exists():
        pytest.skip("Run metadata not generated yet; execute run_h2o_automl.py first.")
    payload = json.loads(RUN_METADATA.read_text())
    assert payload["target"] == "DEAD"
    metric_value = float(payload["metric_value"])
    assert lower <= metric_value <= upper
    assert payload["train_ratio"] == pytest.approx(0.8, rel=0, abs=1e-6)
    assert isinstance(payload["seed"], int)
    assert isinstance(payload["stratified"], bool)

