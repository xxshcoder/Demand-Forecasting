"""FastAPI prediction service. The model always comes from the registry alias."""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException

from ..config import get_admin_token, load_params
from ..features.build import build_single_row_features
from .logging_db import init_db, log_prediction
from .model_loader import ModelLoader
from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    ReloadResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loader = ModelLoader(load_params())
    app.state.loader = loader
    init_db()
    if os.getenv("SKIP_MODEL_REFRESH") != "1":
        loader.refresh()
    if os.getenv("ENABLE_MODEL_POLLING", "1") == "1":
        loader.start_polling()
    yield
    loader.stop_polling()


app = FastAPI(title="Demand Forecasting API", version="1.0.0", lifespan=lifespan)


def get_loader() -> ModelLoader:
    return app.state.loader


@app.get("/health", response_model=HealthResponse)
def health(loader: ModelLoader = Depends(get_loader)) -> HealthResponse:
    if not loader.is_loaded():
        raise HTTPException(status_code=503, detail="No production model loaded")
    return HealthResponse(status="ok", model_loaded=True)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(loader: ModelLoader = Depends(get_loader)) -> ModelInfoResponse:
    if not loader.is_loaded():
        raise HTTPException(status_code=503, detail="No production model loaded")
    return ModelInfoResponse(**loader.info())


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, loader: ModelLoader = Depends(get_loader)) -> PredictResponse:
    if not loader.is_loaded():
        raise HTTPException(status_code=503, detail="No production model loaded")
    info = loader.info()
    started = time.perf_counter()
    features = build_single_row_features(
        recent_sales=request.recent_sales,
        forecast_date=request.forecast_date,
        price=request.price,
        promo=request.promo,
        store_id=request.store_id,
        item_id=request.item_id,
        feature_list=info.get("feature_list"),
    )
    raw_prediction = np.asarray(loader.get_model().predict(features)).ravel()
    if raw_prediction.size != 1 or not np.isfinite(raw_prediction[0]):
        raise HTTPException(status_code=500, detail="Model returned an invalid prediction")
    prediction = float(np.clip(raw_prediction[0], 0, None))
    latency_ms = (time.perf_counter() - started) * 1000
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(UTC).isoformat()

    log_prediction(
        request_id=request_id,
        store_id=request.store_id,
        item_id=request.item_id,
        forecast_date=request.forecast_date,
        features_json=json.dumps(features.iloc[0].to_dict(), default=str),
        prediction=prediction,
        model_version=str(info.get("model_version", "unknown")),
        latency_ms=latency_ms,
    )
    return PredictResponse(
        prediction=round(prediction, 3),
        model_name=str(info.get("model_name", "demand-forecaster")),
        model_version=str(info.get("model_version", "unknown")),
        request_id=request_id,
        timestamp=timestamp,
    )


@app.post("/admin/reload", response_model=ReloadResponse)
def reload_model(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    loader: ModelLoader = Depends(get_loader),
) -> ReloadResponse:
    expected = get_admin_token()
    if not expected or x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing admin token")
    loader.reload()
    return ReloadResponse(
        reloaded=True,
        info=ModelInfoResponse(**loader.info()) if loader.is_loaded() else None,
        timestamp=datetime.now(UTC),
    )
