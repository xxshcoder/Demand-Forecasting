"""Shared pytest fixtures. Everything here is tiny and deterministic."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.config import load_params

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def params() -> dict:
    return load_params()


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def tiny_raw() -> pd.DataFrame:
    frame = pd.read_csv(FIXTURES / "tiny_raw.csv")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.rename(columns={"store": "store_id", "item": "item_id"})


@pytest.fixture
def tiny_processed() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=70, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "store_id": 1,
            "item_id": 1,
            "sales": (10 + (dates.day_of_week >= 5) * 3).astype("int64"),
        }
    )


class StubModel:
    """A model implementing the pyfunc predict contract, without MLflow."""

    def predict(self, features: pd.DataFrame):
        return pd.Series(features["lag_1"].to_numpy(), index=features.index).to_numpy()


@pytest.fixture
def stub_model() -> StubModel:
    return StubModel()


@pytest.fixture
def stub_loader(stub_model) -> SimpleNamespace:
    return SimpleNamespace(
        get_model=lambda: stub_model,
        is_loaded=lambda: True,
        info=lambda: {
            "model_name": "demand-forecaster",
            "model_version": "test-1",
            "alias": "production",
            "run_id": "test-run",
            "training_date": "2026-01-01T00:00:00",
            "validation_metrics": {"mae": 1.0},
            "feature_list": ["lag_1"],
            "data_version": "test-hash",
        },
    )
