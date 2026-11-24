from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app
from src.serving.schemas import PredictRequest


@pytest.fixture
def client(monkeypatch, stub_loader):
    monkeypatch.setenv("SKIP_MODEL_REFRESH", "1")
    monkeypatch.setenv("ENABLE_MODEL_POLLING", "0")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///test_predictions.db")

    def override_loader():
        return stub_loader

    app.dependency_overrides[get_loader_dependency()] = override_loader
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def get_loader_dependency():
    from src.serving.app import get_loader

    return get_loader


def valid_payload() -> dict:
    return {
        "store_id": 1,
        "item_id": 5,
        "forecast_date": "2018-01-01",
        "recent_sales": [10.0] * 28,
        "price": 9.99,
        "promo": 0,
    }


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info_fields(client):
    body = client.get("/model-info").json()
    assert body["model_name"] == "demand-forecaster"
    assert body["model_version"] == "test-1"
    assert body["validation_metrics"] == {"mae": 1.0}
    assert body["data_version"] == "test-hash"


def test_valid_prediction(client):
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prediction"] == 10.0
    assert body["model_version"] == "test-1"
    assert len(body["request_id"]) == 36
    assert "T" in body["timestamp"]


@pytest.mark.parametrize(
    "mutations",
    [
        {"recent_sales": [10.0] * 27},
        {"recent_sales": [-1.0] + [10.0] * 27},
        {"store_id": 1.5},
        {"forecast_date": "01/01/2018"},
        {"promo": 2},
        {"unexpected": 1},
    ],
)
def test_invalid_requests_return_422(client, mutations):
    payload = valid_payload()
    payload.update(mutations)
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_model_returns_503(client, stub_model):
    from types import SimpleNamespace

    app.dependency_overrides[get_loader_dependency()] = lambda: SimpleNamespace(
        get_model=lambda: stub_model, is_loaded=lambda: False, info=lambda: {}
    )
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json=valid_payload()).status_code == 503


def test_admin_reload_requires_token(client, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    assert client.post("/admin/reload").status_code == 401


def test_request_model_validates_history_length():
    with pytest.raises(ValueError):
        PredictRequest(store_id=1, item_id=1, forecast_date="2024-01-01", recent_sales=[1, 2])
