import json

import joblib
import pandas as pd
import pytest

from src.predict import (
    PredictionArtifacts,
    format_latest_data_date,
    get_latest_feature_payload,
    load_best_model,
    make_feature_frame,
    normalize_ticker,
    predict_one,
    predict_ticker,
    validate_feature_columns,
    validate_market_data_columns,
    validate_model_metadata,
)


def test_make_feature_frame_preserves_feature_order_and_casts_values():
    payload = {"b": "2.5", "a": 1}
    feature_columns = ["a", "b"]

    result = make_feature_frame(payload, feature_columns)

    assert list(result.columns) == feature_columns
    assert result.to_dict(orient="records") == [{"a": 1.0, "b": 2.5}]


def test_make_feature_frame_rejects_missing_features():
    with pytest.raises(ValueError, match="Missing required features"):
        make_feature_frame({"a": 1}, ["a", "b"])


def test_make_feature_frame_rejects_empty_feature_columns():
    with pytest.raises(ValueError, match="Feature columns cannot be empty"):
        make_feature_frame({"a": 1}, [])


def test_predict_one_returns_prediction_with_probability():
    class DummyModel:
        def predict(self, X):
            assert isinstance(X, pd.DataFrame)
            return [1]

        def predict_proba(self, X):
            return [[0.25, 0.75]]

    artifacts = PredictionArtifacts(
        model=DummyModel(),
        model_name="dummy",
        feature_columns=["return_1d", "return_5d"],
    )

    result = predict_one({"return_1d": 0.01, "return_5d": 0.02}, artifacts)

    assert result == {
        "model_name": "dummy",
        "prediction": 1,
        "probability": 0.75,
        "features_used": ["return_1d", "return_5d"],
    }


def test_predict_one_uses_positive_class_metadata_for_probability():
    class ReversedClassesModel:
        classes_ = [1, 0]

        def predict(self, X):
            return [1]

        def predict_proba(self, X):
            return [[0.8, 0.2]]

    artifacts = PredictionArtifacts(ReversedClassesModel(), "dummy", ["x"])

    assert predict_one({"x": 1}, artifacts)["probability"] == pytest.approx(0.8)


def test_normalize_ticker_strips_and_uppercases_symbol():
    assert normalize_ticker(" aapl ") == "AAPL"


def test_normalize_ticker_rejects_empty_symbol():
    with pytest.raises(ValueError, match="Ticker is required"):
        normalize_ticker(" ")


def test_get_latest_feature_payload_downloads_data_and_builds_features(monkeypatch):
    def fake_download_data(ticker, start, end):
        assert ticker == "AAPL"
        return pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=11),
                "Close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                "Volume": [1000, 1000, 1000, 1000, 1000, 1500, 1500, 1500, 1500, 1500, 2000],
            }
        )

    monkeypatch.setattr("src.predict.download_data", fake_download_data)

    payload, latest_date = get_latest_feature_payload(
        "aapl",
        ["return_1d", "return_5d", "return_10d", "ma_ratio", "volume_change_5d"],
    )

    assert latest_date == "2024-01-11"
    assert payload["return_1d"] == pytest.approx(1 / 109)
    assert payload["return_5d"] == pytest.approx(5 / 105)
    assert payload["return_10d"] == pytest.approx(0.1)
    assert payload["volume_change_5d"] == pytest.approx(1 / 3)


def test_validate_market_data_columns_rejects_missing_columns():
    df = pd.DataFrame({"Date": ["2024-01-01"], "Close": [100]})

    with pytest.raises(ValueError, match="missing required columns"):
        validate_market_data_columns(df)


def test_validate_model_metadata_rejects_missing_fields():
    with pytest.raises(ValueError, match="missing required fields"):
        validate_model_metadata({"model_name": "random_forest"})


def test_validate_feature_columns_rejects_empty_list():
    with pytest.raises(ValueError, match="Feature columns cannot be empty"):
        validate_feature_columns([])


def test_format_latest_data_date_returns_iso_date_for_timestamps():
    assert format_latest_data_date(pd.Timestamp("2024-01-31 15:30:00")) == "2024-01-31"


def test_load_best_model_rejects_metadata_without_model_path(tmp_path):
    metadata_path = tmp_path / "best_model.json"
    metadata_path.write_text(json.dumps({"model_name": "random_forest"}))

    with pytest.raises(ValueError, match="missing required fields"):
        load_best_model(metadata_path)


def test_load_best_model_uses_default_feature_columns_when_missing(tmp_path):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "best_model.json"

    joblib.dump({"hello": "world"}, model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "dummy",
                "model_path": str(model_path),
            }
        )
    )

    artifacts = load_best_model(metadata_path)

    assert artifacts.model_name == "dummy"
    assert artifacts.feature_columns == [
        "return_1d",
        "return_5d",
        "return_10d",
        "ma_ratio",
        "volume_change_5d",
    ]


def test_predict_ticker_returns_ticker_and_latest_data_date(monkeypatch):
    class DummyModel:
        def predict(self, X):
            return [0]

        def predict_proba(self, X):
            return [[0.8, 0.2]]

    def fake_get_latest_feature_payload(ticker, feature_columns):
        assert ticker == "MSFT"
        return {"return_1d": 0.01, "return_5d": 0.02}, "2024-01-31"

    monkeypatch.setattr("src.predict.get_latest_feature_payload", fake_get_latest_feature_payload)

    artifacts = PredictionArtifacts(
        model=DummyModel(),
        model_name="dummy",
        feature_columns=["return_1d", "return_5d"],
    )

    result = predict_ticker("msft", artifacts)

    assert result == {
        "ticker": "MSFT",
        "model_name": "dummy",
        "prediction": 0,
        "probability": 0.2,
        "features_used": ["return_1d", "return_5d"],
        "latest_data_date": "2024-01-31",
    }
