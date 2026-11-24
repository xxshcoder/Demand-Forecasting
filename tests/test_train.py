from __future__ import annotations

import copy
import json
from types import SimpleNamespace

from src.features.build import build_features
from src.models import train as train_module


class FakeRun:
    def __enter__(self):
        return SimpleNamespace(info=SimpleNamespace(run_id="fake-run"))

    def __exit__(self, *args):
        return False


class FakeMLflow:
    def __init__(self):
        self.calls: list[str] = []

    def set_tracking_uri(self, uri):
        self.calls.append("set_tracking_uri")

    def set_registry_uri(self, uri):
        self.calls.append("set_registry_uri")

    def set_experiment(self, name):
        self.calls.append("set_experiment")

    def start_run(self, run_name):
        self.calls.append(f"run:{run_name}")
        return FakeRun()

    def log_params(self, values):
        self.calls.append("log_params")

    def log_metrics(self, values):
        self.calls.append("log_metrics")

    def set_tags(self, values):
        self.calls.append("set_tags")

    def log_dict(self, value, path):
        self.calls.append("log_dict")

    def log_artifact(self, path):
        self.calls.append("log_artifact")

    def infer_signature(self, example, predictions):
        return "signature"

    sklearn = SimpleNamespace(log_model=lambda **kwargs: None)
    pyfunc = SimpleNamespace(log_model=lambda **kwargs: None)


def test_train_all_logs_five_runs(tiny_processed, params, monkeypatch, tmp_path):
    cfg = copy.deepcopy(params)
    cfg["split"]["val_days"] = 10
    cfg["split"]["test_days"] = 5
    cfg["models"]["hgb"]["max_iter"] = 5
    cfg["models"]["nn"]["epochs"] = 1
    cfg["models"]["nn"]["early_stop_patience"] = 10
    frame = build_features(tiny_processed, cfg).dropna().reset_index(drop=True)
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    frame.to_parquet(processed_dir / "features.parquet", index=False)
    tiny_processed.to_parquet(processed_dir / "sales.parquet", index=False)

    monkeypatch.setattr(train_module, "ROOT", tmp_path)
    monkeypatch.setattr(train_module, "resolve_path", lambda path: tmp_path / path)
    monkeypatch.chdir(tmp_path)
    fake = FakeMLflow()
    fake.models = fake
    monkeypatch.setattr(train_module, "mlflow", fake)

    summary = train_module.train_all(cfg)
    assert set(summary["models"]) == {"last_week", "moving_average", "linear", "hgb", "nn"}
    assert sum(call.startswith("run:") for call in fake.calls) == 5
    assert (tmp_path / "reports" / "train_summary.json").exists()
    stored = json.loads((tmp_path / "reports" / "train_summary.json").read_text(encoding="utf-8"))
    assert stored["validation_rows"] == 10 * 1  # one series in the tiny fixture
    for name in summary["models"]:
        assert (tmp_path / "reports" / "predictions" / f"{name}.csv").exists()
        assert "mae" in summary["models"][name]["metrics"]
