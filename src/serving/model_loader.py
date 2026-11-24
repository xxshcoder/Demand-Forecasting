"""Registry-backed model loader with an atomically swapped cache and polling."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any

from ..config import get_mlflow_tracking_uri, get_model_refresh_seconds, load_params


class ModelLoader:
    """Load ``models:/demand-forecaster@production`` and keep it up to date."""

    def __init__(self, params: dict | None = None, tracking_uri: str | None = None):
        self.params = params or load_params()
        self.tracking_uri = tracking_uri or get_mlflow_tracking_uri()
        self.model_name = self.params["mlflow"]["registered_model_name"]
        self.alias = self.params["mlflow"]["production_alias"]
        self.refresh_seconds = get_model_refresh_seconds(self.params)
        self._snapshot: tuple[Any, dict] | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _client(self):
        import mlflow

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.tracking_uri)
        return mlflow.tracking.MlflowClient()

    def _build_snapshot(self, version: str, run_id: str) -> tuple[Any, dict]:
        import mlflow

        model = mlflow.pyfunc.load_model(f"models:/{self.model_name}@{self.alias}")
        client = self._client()
        run = client.get_run(run_id)
        metrics = {
            key.removeprefix("val_"): float(value)
            for key, value in run.data.metrics.items()
            if key.startswith("val_")
        }
        feature_param = run.data.params.get("feature_list", "")
        training_date = None
        if run.info.start_time:
            training_date = datetime.fromtimestamp(run.info.start_time / 1000, tz=UTC).isoformat()
        info = {
            "model_name": self.model_name,
            "model_version": str(version),
            "alias": self.alias,
            "training_date": training_date,
            "run_id": run_id,
            "validation_metrics": metrics,
            "feature_list": [name for name in feature_param.split(",") if name],
            "data_version": run.data.tags.get("dvc_data_hash"),
            "loaded_at": datetime.now(UTC).isoformat(),
        }
        return model, info

    def refresh(self) -> bool:
        """Check the alias version and swap the cached model only if it changed."""
        try:
            client = self._client()
            version = client.get_model_version_by_alias(self.model_name, self.alias)
        except Exception as error:  # noqa: BLE001 - registry may not be reachable yet
            print(f"Model registry unavailable: {error}")
            return False
        with self._lock:
            current = self._snapshot
            if current and current[1]["model_version"] == str(version.version):
                return True
        try:
            snapshot = self._build_snapshot(str(version.version), version.run_id)
        except Exception as error:  # noqa: BLE001 - keep serving the previous version
            print(f"Could not load model version {version.version}: {error}")
            return False
        with self._lock:
            self._snapshot = snapshot
        return True

    def reload(self) -> bool:
        """Force a re-check, even when the version has not changed."""
        return self.refresh()

    def is_loaded(self) -> bool:
        return self._snapshot is not None

    def get_model(self):
        return self._snapshot[0] if self._snapshot else None

    def info(self) -> dict:
        return dict(self._snapshot[1]) if self._snapshot else {}

    def start_polling(self) -> None:
        if self._thread is not None:
            return

        def loop() -> None:
            while not self._stop.wait(self.refresh_seconds):
                self.refresh()

        self._thread = threading.Thread(target=loop, name="model-refresh", daemon=True)
        self._thread.start()

    def stop_polling(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
