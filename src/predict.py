from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data_loader import download_data
from src.features import add_features

DEFAULT_FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "ma_ratio",
    "volume_change_5d",
]

# src/predict.py -> parent is src, parent.parent is project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "best_model.json"


@dataclass
class PredictionArtifacts:
    model: Any
    model_name: str
    feature_columns: list[str]


REQUIRED_MARKET_DATA_COLUMNS = {"Date", "Close", "Volume"}
REQUIRED_METADATA_FIELDS = {"model_name", "model_path"}


def normalize_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker is required")
    return ticker


def validate_market_data_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_MARKET_DATA_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Downloaded market data is missing required columns: {sorted(missing)}")


def validate_model_metadata(meta: dict[str, Any]) -> None:
    missing = REQUIRED_METADATA_FIELDS.difference(meta)
    if missing:
        raise ValueError(f"Model metadata is missing required fields: {sorted(missing)}")


def validate_feature_columns(feature_columns: list[str]) -> None:
    if not feature_columns:
        raise ValueError("Feature columns cannot be empty")


def format_latest_data_date(value: Any) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def load_best_model(
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> PredictionArtifacts:
    """
    Load the best trained model using a metadata file.

    Example best_model.json:
    {
        "model_name": "logistic_regression",
        "model_path": "models/logistic_regression.joblib",
        "feature_columns": [
            "return_1d",
            "return_5d",
            "return_10d",
            "ma_ratio",
            "volume_change_5d"
        ]
    }
    """
    metadata_path = Path(metadata_path)

    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    validate_model_metadata(meta)

    model_path = Path(meta["model_path"])
    model_name = meta["model_name"]
    feature_columns = meta.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    validate_feature_columns(feature_columns)

    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)

    return PredictionArtifacts(
        model=model,
        model_name=model_name,
        feature_columns=feature_columns,
    )


def make_feature_frame(payload: dict, feature_columns: list[str]) -> pd.DataFrame:
    validate_feature_columns(feature_columns)
    missing = [col for col in feature_columns if col not in payload]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    row = {col: float(payload[col]) for col in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def get_latest_feature_payload(
    ticker: str,
    feature_columns: list[str],
    lookback_days: int = 90,
) -> tuple[dict, str]:
    ticker = normalize_ticker(ticker)
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=lookback_days)

    df = download_data(ticker=ticker, start=start.isoformat(), end=end.isoformat())
    if df.empty:
        raise ValueError(f"No market data found for ticker: {ticker}")

    validate_market_data_columns(df)
    df = add_features(df)
    df = df.dropna(subset=feature_columns).copy()
    if df.empty:
        raise ValueError(f"Not enough market data to create features for ticker: {ticker}")

    latest_row = df.iloc[-1]
    payload = {column: latest_row[column] for column in feature_columns}
    latest_date = format_latest_data_date(latest_row["Date"])

    return payload, latest_date


def predict_one(payload: dict, artifacts: PredictionArtifacts, ticker: str | None = None) -> dict:
    X = make_feature_frame(payload, artifacts.feature_columns)

    pred = int(artifacts.model.predict(X)[0])

    probability = None
    if hasattr(artifacts.model, "predict_proba"):
        proba = artifacts.model.predict_proba(X)[0]
        classes = getattr(artifacts.model, "classes_", None)
        if classes is not None:
            positive = next((i for i, label in enumerate(classes) if label == 1), None)
            probability = float(proba[positive]) if positive is not None else 0.0
        elif len(proba) > 1:
            # Most sklearn classifiers use [negative, positive] when classes
            # metadata is unavailable.
            probability = float(proba[1])

    result = {
        "model_name": artifacts.model_name,
        "prediction": pred,
        "probability": probability,
        "features_used": artifacts.feature_columns,
    }

    if ticker is not None:
        result["ticker"] = normalize_ticker(ticker)

    return result


def predict_ticker(ticker: str, artifacts: PredictionArtifacts) -> dict:
    ticker = normalize_ticker(ticker)
    payload, latest_date = get_latest_feature_payload(ticker, artifacts.feature_columns)
    result = predict_one(payload, artifacts, ticker=ticker)
    result["latest_data_date"] = latest_date
    return result
