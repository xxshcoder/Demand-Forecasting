from __future__ import annotations

from types import SimpleNamespace

from src.serving.model_loader import ModelLoader


class FakeClient:
    def __init__(self, version: str = "1", fail: bool = False):
        self.version = version
        self.fail = fail

    def get_model_version_by_alias(self, name, alias):
        if self.fail:
            raise RuntimeError("registry down")
        return SimpleNamespace(version=self.version, run_id=f"run-{self.version}")


def test_loader_swaps_when_alias_version_changes(params, stub_model, monkeypatch):
    loader = ModelLoader(params=params, tracking_uri="sqlite:///test.db")
    client = FakeClient("1")
    monkeypatch.setattr(loader, "_client", lambda: client)
    monkeypatch.setattr(
        loader,
        "_build_snapshot",
        lambda version, run_id: (stub_model, {"model_version": version, "run_id": run_id}),
    )
    assert loader.refresh() is True
    assert loader.is_loaded()
    assert loader.info()["model_version"] == "1"
    assert loader.get_model() is stub_model
    assert loader.refresh() is True  # no change
    client.version = "2"
    assert loader.reload() is True
    assert loader.info()["model_version"] == "2"
    loader.stop_polling()


def test_loader_starts_unloaded_when_registry_unreachable(params, monkeypatch):
    loader = ModelLoader(params=params, tracking_uri="sqlite:///test.db")
    monkeypatch.setattr(loader, "_client", lambda: FakeClient(fail=True))
    assert loader.refresh() is False
    assert not loader.is_loaded()
    assert loader.info() == {}


def test_polling_thread_start_and_stop(params, monkeypatch):
    loader = ModelLoader(params=params, tracking_uri="sqlite:///test.db")
    loader.refresh_seconds = 1
    monkeypatch.setattr(loader, "refresh", lambda: True)
    loader.start_polling()
    assert loader._thread is not None
    loader.stop_polling()
    assert loader._thread is None
