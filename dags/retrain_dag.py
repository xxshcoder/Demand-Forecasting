"""Weekly retraining DAG. All business logic lives in ``src``; tasks are thin."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from src.config import ROOT, load_params, resolve_path

TASK_ORDER = [
    "ingest",
    "validate",
    "build_features",
    "train",
    "evaluate",
    "compare",
    "register_and_promote",
]
DAG_ID = "demand_forecast_retrain"

try:
    from airflow.decorators import dag, task  # type: ignore[import-not-found]

    AIRFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - local tests fall back to AST checks
    AIRFLOW_AVAILABLE = False
    dag = task = None  # type: ignore[assignment]


def _ingest() -> str:
    params = load_params()
    path = resolve_path(params["data"]["raw_path"])
    if not path.exists():
        raise FileNotFoundError(f"Raw data missing at {path}; run scripts/make_synthetic_data.py")
    return str(path)


def _validate() -> str:
    from src.data.prepare import prepare

    prepare()
    return str(resolve_path(load_params()["data"]["processed_path"]))


def _build_features() -> str:
    from src.features.build import build_features
    from src.features.split import time_split

    params = load_params()
    processed = resolve_path(params["data"]["processed_path"])
    import pandas as pd

    frame = pd.read_parquet(processed)
    enriched = build_features(frame, params)
    warmup = [c for c in enriched.columns if c.startswith("lag_") or c.startswith("roll_")]
    enriched = enriched.dropna(subset=warmup).reset_index(drop=True)
    output = resolve_path("data/processed/features.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output, index=False)
    reference = resolve_path("data/reference/train_features.parquet")
    reference.parent.mkdir(parents=True, exist_ok=True)
    train, _, _ = time_split(
        enriched, int(params["split"]["val_days"]), int(params["split"]["test_days"])
    )
    train.to_parquet(reference, index=False)
    return str(output)


def _train() -> str:
    from src.models.train import train_all

    summary = train_all()
    return summary["models"]


def _evaluate() -> str:
    from src.models.evaluate import evaluate_saved_predictions

    result = evaluate_saved_predictions()
    return str(len(result["metrics"]))


def _compare() -> str:
    from src.models.compare import main as compare_main

    compare_main()
    return str(ROOT / "reports" / "model_comparison.md")


def _register_and_promote(models: dict) -> str:
    from src.models.promote import promote_from_run

    params = load_params()
    summary = json.loads((ROOT / "reports" / "train_summary.json").read_text(encoding="utf-8"))
    primary = params["promotion"]["primary_metric"].upper()
    learned = {
        name: entry
        for name, entry in summary["models"].items()
        if entry["model_type"] != "baseline"
    }
    best_name = min(learned, key=lambda name: learned[name]["metrics"][primary])
    decision = promote_from_run(learned[best_name]["run_id"], params=params)
    return decision["reason"]


if AIRFLOW_AVAILABLE:

    @dag(
        dag_id=DAG_ID,
        schedule="@weekly",
        start_date=datetime(2024, 1, 1),
        catchup=False,
        default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
        tags=["mlops", "demand-forecasting"],
    )
    def demand_forecast_retrain():
        @task(task_id="ingest")
        def ingest() -> str:
            return _ingest()

        @task(task_id="validate")
        def validate(path: str) -> str:
            return _validate()

        @task(task_id="build_features")
        def build_features(path: str) -> str:
            return _build_features()

        @task(task_id="train")
        def train(path: str) -> dict:
            return _train()

        @task(task_id="evaluate")
        def evaluate(models: dict) -> str:
            return _evaluate()

        @task(task_id="compare")
        def compare(rows: str) -> str:
            return _compare()

        @task(task_id="register_and_promote")
        def register_and_promote(comparison: str) -> str:
            return _register_and_promote(
                json.loads((ROOT / "reports" / "train_summary.json").read_text())["models"]
            )

        raw = ingest()
        processed = validate(raw)
        features = build_features(processed)
        models = train(features)
        evaluation = evaluate(models)
        comparison = compare(evaluation)
        register_and_promote(comparison)

    demand_forecast_retrain_dag = demand_forecast_retrain()
else:  # pragma: no cover
    demand_forecast_retrain_dag = None
