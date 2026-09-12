import numpy as np
import pandas as pd
import pytest

from src import train
from src.config import LABEL_HORIZON


@pytest.fixture
def market_data():
    return pd.DataFrame({
        "Date": pd.bdate_range("2021-11-01", periods=80),
        "Close": np.arange(80, dtype=float) + 100,
        "Volume": np.arange(80, dtype=float) + 1000,
    })


def test_training_labels_end_before_test_period(monkeypatch, market_data):
    monkeypatch.setattr(train, "download_data", lambda: market_data.copy())
    monkeypatch.setattr(train, "TRAIN_SPLIT_DATE", "2022-01-01")

    X_train, y_train, X_test, y_test = train.prepare_dataset()

    first_test = market_data.index[market_data["Date"] >= "2022-01-01"][0]
    assert X_train.index.max() == first_test - LABEL_HORIZON - 1
    assert X_test.index.min() == first_test
    assert X_test.index.max() == len(market_data) - LABEL_HORIZON - 1
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
    assert list(X_train.columns) == train.FEATURE_COLUMNS

    # Changing held-out prices cannot change any training features or labels.
    market_data.loc[first_test:, "Close"] *= 10
    changed_X, changed_y, _, _ = train.prepare_dataset()
    pd.testing.assert_frame_equal(X_train, changed_X)
    pd.testing.assert_series_equal(y_train, changed_y)


@pytest.mark.parametrize("split_date", ["2020-01-01", "2023-01-01", "2021-11-15"])
def test_empty_partition_has_actionable_error(monkeypatch, market_data, split_date):
    monkeypatch.setattr(train, "download_data", lambda: market_data.copy())
    monkeypatch.setattr(train, "TRAIN_SPLIT_DATE", split_date)

    with pytest.raises(ValueError, match="label-horizon purging"):
        train.prepare_dataset()
