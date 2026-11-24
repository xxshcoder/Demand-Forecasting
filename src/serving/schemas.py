"""Pydantic v2 request and response contracts for the prediction API."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..config import load_params


def _history_length() -> int:
    return max(int(lag) for lag in load_params()["features"]["lags"])


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    store_id: int = Field(gt=0)
    item_id: int = Field(gt=0)
    forecast_date: date
    recent_sales: list[float]
    price: float | None = None
    promo: int = 0

    @field_validator("store_id", "item_id", mode="before")
    @classmethod
    def require_real_integers(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("store_id and item_id must be integers")
        return value

    @field_validator("forecast_date", mode="before")
    @classmethod
    def parse_iso_date(cls, value: Any) -> Any:
        if isinstance(value, str):
            return date.fromisoformat(value)
        return value

    @field_validator("recent_sales")
    @classmethod
    def validate_history(cls, values: list[float]) -> list[float]:
        expected = _history_length()
        if len(values) != expected:
            raise ValueError(f"recent_sales must contain exactly {expected} values")
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("recent_sales values must be finite and non-negative")
        return values

    @field_validator("promo")
    @classmethod
    def validate_promo(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError("promo must be 0 or 1")
        return value

    @field_validator("price")
    @classmethod
    def validate_price(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("price must be finite and non-negative")
        return value


class PredictResponse(BaseModel):
    prediction: float
    model_name: str
    model_version: str
    request_id: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    alias: str
    training_date: str | None = None
    run_id: str | None = None
    validation_metrics: dict[str, float] = Field(default_factory=dict)
    feature_list: list[str] = Field(default_factory=list)
    data_version: str | None = None
    loaded_at: str | None = None


class ReloadResponse(BaseModel):
    reloaded: bool
    info: ModelInfoResponse | None = None
    timestamp: datetime
