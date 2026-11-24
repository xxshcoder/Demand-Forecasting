from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd

from src.models import promote as promote_module
from src.models.promote import should_promote


class _FakeClient:
    def get_model_version_by_alias(self, name, alias):
        return SimpleNamespace(version="3", run_id="run-3", name=name, aliases=["production"])

    def search_model_versions(self, query):
        return []

    def set_registered_model_alias(self, name, alias, version):
        self.last_alias = (name, alias, version)


def metrics(mae: float, rmse: float = 10.0) -> dict:
    return {"mae": mae, "rmse": rmse, "mape": mae * 10, "wape": mae * 10}


def test_refuses_worse_model(params):
    ok, reason = should_promote(metrics(12), metrics(10), {"moving_average": metrics(15)}, params)
    assert not ok and "improvement" in reason


def test_refuses_model_that_does_not_beat_baselines(params):
    ok, reason = should_promote(metrics(16), metrics(10), {"moving_average": metrics(15)}, params)
    assert not ok and "baseline" in reason


def test_refuses_improvement_below_threshold(params):
    ok, reason = should_promote(metrics(9.98), metrics(10), {"moving_average": metrics(15)}, params)
    assert not ok and "required" in reason


def test_refuses_secondary_regression(params):
    ok, reason = should_promote(
        metrics(9.0, 12.0), metrics(10.0, 10.0), {"moving_average": metrics(15)}, params
    )
    assert not ok and "RMSE" in reason


def test_refuses_nan_metrics(params):
    ok, reason = should_promote(
        metrics(float("nan")), metrics(10), {"moving_average": metrics(15)}, params
    )
    assert not ok and "NaN" in reason
    ok, _ = should_promote(
        metrics(1), metrics(float("inf")), {"moving_average": metrics(15)}, params
    )
    assert not ok


def test_promotes_first_model_when_it_beats_baseline(params):
    ok, reason = should_promote(metrics(9), None, {"moving_average": metrics(15)}, params)
    assert ok and "first model" in reason


def test_promotes_clearly_better_model(params):
    ok, reason = should_promote(metrics(8), metrics(10), {"moving_average": metrics(15)}, params)
    assert ok and "Promote" in reason


def test_refuses_with_no_baselines(params):
    ok, reason = should_promote(metrics(8), metrics(10), None, params)
    assert not ok and "baseline" in reason


def test_registry_helpers_use_aliases(params, monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(promote_module, "_client", lambda: client)
    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        set_registry_uri=lambda uri: None,
        tracking=SimpleNamespace(MlflowClient=lambda: client),
        register_model=lambda model_uri, name: SimpleNamespace(version="4"),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    info = promote_module.get_production_model_info(params)
    assert info["version"] == "3" and info["run_id"] == "run-3"
    assert promote_module.register_model("run-x", params) == "4"
    promote_module.set_alias("4", "candidate", params)
    assert client.last_alias[-1] == "4"


def test_score_production_uses_only_model_features(params, stub_model, monkeypatch):
    from src.features import build as feature_build

    fake_mlflow = SimpleNamespace(pyfunc=SimpleNamespace(load_model=lambda uri: stub_model))
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr(
        feature_build, "model_feature_columns", lambda frame, params=None: ["lag_1"]
    )
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10),
            "store_id": 1,
            "item_id": 1,
            "sales": range(10),
            "lag_1": range(10),
        }
    )
    result = promote_module.score_model_on_validation(
        "models:/demand-forecaster@production", frame, params
    )
    assert set(result) == {"mae", "rmse", "mape", "wape"}
