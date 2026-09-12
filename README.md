# Stock Movement ML System

An end-to-end machine learning system for predicting short-term stock movement, with model serving via FastAPI and Docker.

---

## Overview

This project includes:

* Data ingestion from Yahoo Finance
* Feature engineering on price and volume signals
* Model training and selection
* Model serialization and metadata tracking
* REST API for ticker-based prediction using FastAPI
* Dockerized deployment

---

## Problem Definition

* Task: Binary classification
* Goal: Predict whether a stock will increase by more than 1% over the next 5 days

Label:

```
future_5d_return > 1%
```

---

## Features

* return_1d
* return_5d
* return_10d
* ma_ratio
* volume_change_5d

---

## Project Structure

```
stock-movement-ml-system/
├── app/
│   └── main.py
├── src/
│   ├── train.py
│   ├── train_lr.py
│   ├── train_rf.py
│   ├── train_torch.py
│   └── predict.py
├── tests/
│   ├── test_features.py
│   ├── test_labels.py
│   └── test_predict.py
├── models/
│   ├── *.joblib
│   └── best_model.json
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Models

* Logistic Regression
* Random Forest
* PyTorch MLP

Model selection is based on F1 score.
The best model is saved along with metadata in:

```
models/best_model.json
```

Example:

```json
{
  "model_name": "random_forest",
  "model_path": "models/random_forest.joblib",
  "feature_columns": [
    "return_1d",
    "return_5d",
    "return_10d",
    "ma_ratio",
    "volume_change_5d"
  ]
}
```

---

## Training

Run:

```
python -m src.train
```

This will:

* Download training data
* Build features and labels
* Train baseline models
* Evaluate performance
* Save the best model
* Write metadata to best_model.json

Training uses a chronological split at `TRAIN_SPLIT_DATE`. Rows whose five-session
target reaches the test period are excluded from training, so training labels
cannot use held-out prices. Feature warmup and rows without complete future
returns are also excluded. Both partitions must contain usable rows.

---

## API

Start locally:

```
uvicorn app.main:app --reload
```

Docs:

```
http://localhost:8000/docs
```

### Endpoints

**GET /health**

```
{"status": "ok", "model_loaded": true}
```

---

**POST /predict**

The customer only provides a stock ticker. The API downloads recent market data,
builds the required model features internally, and predicts using the latest
available feature row.

Request:

```json
{
  "ticker": "AAPL"
}
```

Response:

```json
{
  "ticker": "AAPL",
  "model_name": "random_forest",
  "prediction": 1,
  "probability": 0.67,
  "features_used": [
    "return_1d",
    "return_5d",
    "return_10d",
    "ma_ratio",
    "volume_change_5d"
  ],
  "latest_data_date": "2024-12-30 00:00:00"
}
```

### Batch predictions

**POST /predict/batch** accepts up to 50 tickers and returns independent results. A
problem with one ticker is returned in that ticker's `error` field without failing
the whole request:

```json
{"tickers": ["AAPL", "MSFT", "NVDA"]}
```

Recent market downloads are cached in-process to reduce duplicate requests. Clear
the cache after a new trading session with **POST /cache/clear**. The cache is local
to one API process and is not a replacement for a shared production cache.

## Testing

Install dependencies and run:

```
python -m pip install -r requirements.txt
python -m pytest -q
```

## Docker

```
docker compose up --build
```

The API is then available at `http://localhost:8000`, with interactive docs at
`http://localhost:8000/docs`.

---

## Docker

Build:

```
docker build -t stock-ml-api .
```

Run:

```
docker run -p 8000:8000 stock-ml-api
```

Then open:

```
http://localhost:8000/docs
```

---

## Notes

* The current training configuration uses SPY data as the initial baseline
* The API accepts any ticker supported by Yahoo Finance and computes features at request time
* Model quality for non-SPY tickers has not been validated yet
* Model paths are relative for Docker compatibility
* Models are included in the image for reproducible serving
* API loads model once at startup

---

## Tests

Run:

```
python -m pytest
```

Current tests cover:

* Feature generation
* Label generation
* Prediction input handling
* Ticker-based feature creation
* Training split isolation and empty-partition validation

---

## Future Work

* Add experiment tracking
* Add CI/CD
* Improve features
* Validate model performance across multiple tickers
* Deploy to cloud
