import pandas as pd

from src.run_h2o_automl import train_valid_split


def test_train_valid_split_stratified() -> None:
    size = 500
    positives = int(size * 0.2)
    negatives = size - positives
    df = pd.DataFrame(
        {
            "feature": range(size),
            "target": [1] * positives + [0] * negatives,
        }
    )
    train, valid = train_valid_split(
        df,
        target_column="target",
        train_ratio=0.8,
        seed=42,
        stratify=True,
        problem_type="classification",
    )
    assert len(train) + len(valid) == size
    valid_rate = valid["target"].mean()
    overall_rate = df["target"].mean()
    assert abs(valid_rate - overall_rate) < 0.02


def test_train_valid_split_regression() -> None:
    df = pd.DataFrame({"target": range(50)})
    train, valid = train_valid_split(
        df,
        target_column="target",
        train_ratio=0.7,
        seed=1,
        stratify=False,
        problem_type="regression",
    )
    assert len(train) + len(valid) == 50
    assert len(train) >= len(valid)

