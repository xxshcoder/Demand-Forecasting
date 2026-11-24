"""Central configuration: params.yaml plus environment variables.

No model, split, data or monitoring setting is hard-coded in the code that
uses it; all of those values are read from ``params.yaml``. Environment
variables are used only for deployment-specific settings (service URLs,
credentials and secrets).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

try:  # .env is convenient locally; harmless when python-dotenv is absent.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional dependency
    pass

ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"
DEFAULT_LOCAL_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_REMOTE_TRACKING_URI = "http://localhost:5000"
DEFAULT_LOCAL_DATABASE_URL = "sqlite:///app.db"
DEFAULT_REMOTE_DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/app"


def load_params(path: Path | str = PARAMS_PATH) -> dict[str, Any]:
    """Load and return the single source of truth for all settings."""
    with Path(path).open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    if not isinstance(params, dict):
        raise ValueError(f"params file {path} must contain a YAML mapping")
    return params


def get_env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    """Read an environment variable, optionally failing with a clear message."""
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def get_mlflow_tracking_uri() -> str:
    """Return the configured MLflow URI.

    Compose sets this to the MLflow service. When running locally without
    Compose we use a SQLite-backed store, which (unlike a plain file store)
    supports the model registry aliases used by the serving layer.
    """
    return get_env("MLFLOW_TRACKING_URI", DEFAULT_LOCAL_TRACKING_URI) or DEFAULT_LOCAL_TRACKING_URI


def get_database_url() -> str:
    """Return the application (prediction log) database URL."""
    return get_env("DATABASE_URL", DEFAULT_LOCAL_DATABASE_URL) or DEFAULT_LOCAL_DATABASE_URL


def get_admin_token() -> str | None:
    """Return the optional token protecting /admin/reload."""
    return get_env("ADMIN_TOKEN")


def get_dataset_label() -> str:
    """Return the dataset provenance label recorded with every run."""
    return get_env("DATASET_LABEL", "synthetic") or "synthetic"


def get_model_refresh_seconds(params: dict[str, Any] | None = None) -> int:
    params = params or load_params()
    override = get_env("MODEL_REFRESH_SECONDS")
    if override:
        return int(override)
    return int(params["serving"]["model_refresh_seconds"])


def resolve_path(path: str | Path) -> Path:
    """Resolve a configured relative path against the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate
