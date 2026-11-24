"""Best-effort prediction logging. A database outage never breaks a prediction."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

from sqlalchemy import Column, Date, Float, Integer, MetaData, String, Table, create_engine, insert

from ..config import get_database_url

logger = logging.getLogger(__name__)
metadata = MetaData()
prediction_logs = Table(
    "prediction_logs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("request_id", String(64), nullable=False),
    Column("ts", String(64), nullable=False),
    Column("store_id", Integer, nullable=False),
    Column("item_id", Integer, nullable=False),
    Column("forecast_date", Date, nullable=False),
    Column("features_json", String, nullable=False),
    Column("prediction", Float, nullable=False),
    Column("model_version", String(64), nullable=False),
    Column("latency_ms", Float, nullable=False),
)
_engines: dict[str, object] = {}


def get_engine():
    url = get_database_url()
    if url not in _engines:
        options = {}
        if url.startswith("sqlite"):
            options["connect_args"] = {"check_same_thread": False}
        _engines[url] = create_engine(url, **options)
    return _engines[url]


def init_db() -> None:
    """Create the table if it does not exist yet."""
    try:
        metadata.create_all(get_engine())
    except Exception:  # noqa: BLE001
        logger.exception("Could not initialise prediction log database")


def log_prediction(
    *,
    request_id: str,
    store_id: int,
    item_id: int,
    forecast_date: date,
    features_json: str,
    prediction: float,
    model_version: str,
    latency_ms: float,
) -> None:
    """Insert one prediction row; swallow and log any database error."""
    try:
        with get_engine().begin() as connection:
            connection.execute(
                insert(prediction_logs).values(
                    request_id=request_id,
                    ts=datetime.now(UTC).isoformat(),
                    store_id=store_id,
                    item_id=item_id,
                    forecast_date=forecast_date,
                    features_json=features_json,
                    prediction=float(prediction),
                    model_version=model_version,
                    latency_ms=float(latency_ms),
                )
            )
    except Exception:  # noqa: BLE001
        logger.exception("Could not write prediction log (request %s)", request_id)
