from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src.predict import load_best_model, predict_ticker, clear_prediction_cache, normalize_ticker


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tickers: list[str]


class PredictResponse(BaseModel):
    ticker: str
    model_name: str
    prediction: int
    probability: Optional[float]
    features_used: list[str]
    latest_data_date: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifacts = load_best_model()
    yield


app = FastAPI(
    title="Stock Movement ML API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "artifacts")}


@app.post("/cache/clear")
def clear_cache():
    clear_prediction_cache()
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        result = predict_ticker(request.ticker, app.state.artifacts)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")


@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest):
    try:
        tickers = list(dict.fromkeys(normalize_ticker(t) for t in request.tickers))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not tickers or len(tickers) > 50:
        raise HTTPException(status_code=400, detail="Provide between 1 and 50 tickers")
    results = []
    for ticker in tickers:
        try:
            results.append(predict_ticker(ticker, app.state.artifacts))
        except (ValueError, FileNotFoundError) as exc:
            results.append({"ticker": ticker, "error": str(exc)})
    return {"results": results}
