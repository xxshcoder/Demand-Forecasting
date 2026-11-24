"""Pure promotion gate plus MLflow registry operations."""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import pandas as pd

from ..config import ROOT, get_mlflow_tracking_uri, load_params

FINITE_METRICS = ("mae", "rmse", "mape", "wape")


def _usable(metrics: dict[str, Any] | None) -> bool:
    if not metrics:
        return False
    return all(
        key in metrics
        and isinstance(metrics[key], (int, float))
        and math.isfinite(float(metrics[key]))
        for key in FINITE_METRICS
    )


def should_promote(
    candidate_metrics: dict[str, Any] | None,
    production_metrics: dict[str, Any] | None,
    baseline_metrics: dict[str, dict[str, Any]] | None,
    params: dict,
) -> tuple[bool, str]:
    """Decide whether a candidate may take the production alias.

    Rules (all must hold) are documented in plan.md section 4.6. Metrics must
    come from the same, latest validation window.
    """
    cfg = params["promotion"]
    primary = str(cfg["primary_metric"]).lower()
    if not _usable(candidate_metrics) or (
        production_metrics is not None and not _usable(production_metrics)
    ):
        return False, "Refused: NaN/inf or missing metrics."
    if not baseline_metrics:
        return False, "Refused: no baseline metrics available for comparison."

    valid_baselines = {name: values for name, values in baseline_metrics.items() if _usable(values)}
    if not valid_baselines:
        return False, "Refused: no finite baseline metrics available."
    best_baseline_name = min(
        valid_baselines, key=lambda name: float(valid_baselines[name][primary])
    )
    best_baseline = float(valid_baselines[best_baseline_name][primary])
    candidate = float(candidate_metrics[primary])  # type: ignore[index]
    if not candidate < best_baseline:
        return (
            False,
            f"Refused: candidate {primary}={candidate:.4f} does not beat best baseline "
            f"{best_baseline_name}={best_baseline:.4f}.",
        )

    if not production_metrics:
        return (
            True,
            f"Promote first model: beats {best_baseline_name} "
            f"({candidate:.4f} < {best_baseline:.4f}).",
        )

    production = float(production_metrics[primary])  # type: ignore[index]
    improvement = (production - candidate) / abs(production) if production else float("inf")
    minimum = float(cfg["min_relative_improvement"])
    if improvement < minimum:
        return (
            False,
            f"Refused: relative {primary} improvement {improvement:.4%} < required {minimum:.4%}.",
        )

    candidate_rmse = float(candidate_metrics["rmse"])  # type: ignore[index]
    production_rmse = float(production_metrics["rmse"])  # type: ignore[index]
    maximum = float(cfg["max_secondary_regression"])
    if production_rmse and candidate_rmse > production_rmse * (1 + maximum):
        return (
            False,
            f"Refused: RMSE regression {(candidate_rmse / production_rmse - 1):.4%} "
            f"> allowed {maximum:.4%}.",
        )
    reason = (
        f"Promote: {primary} improved {improvement:.4%} over production and beats "
        f"{best_baseline_name}; RMSE within the allowed regression."
    )
    return True, reason


def _client():
    import mlflow

    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    mlflow.set_registry_uri(get_mlflow_tracking_uri())
    return mlflow.tracking.MlflowClient()


def get_production_model_info(params: dict | None = None) -> dict | None:
    """Return info about the version carrying the production alias, if any."""
    params = params or load_params()
    client = _client()
    try:
        version = client.get_model_version_by_alias(
            params["mlflow"]["registered_model_name"], params["mlflow"]["production_alias"]
        )
    except Exception:  # noqa: BLE001 - registry may be empty or unreachable
        return None
    return {
        "version": version.version,
        "run_id": version.run_id,
        "name": version.name,
        "aliases": list(getattr(version, "aliases", []) or []),
    }


def register_model(run_id: str, params: dict | None = None) -> str:
    """Register a model from an MLflow run and return the new version string."""
    import mlflow

    params = params or load_params()
    client = _client()
    versions = client.search_model_versions(f"run_id='{run_id}'")
    for version in versions:
        if version.name == params["mlflow"]["registered_model_name"]:
            return str(version.version)
    result = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model", name=params["mlflow"]["registered_model_name"]
    )
    return str(result.version)


def set_alias(version: str, alias: str, params: dict | None = None) -> None:
    """Attach a registry alias to a version."""
    params = params or load_params()
    client = _client()
    client.set_registered_model_alias(params["mlflow"]["registered_model_name"], alias, version)


def score_model_on_validation(model_uri: str, features: pd.DataFrame, params: dict) -> dict:
    """Re-score a model on the latest validation window with the shared features."""
    import mlflow

    from ..features.build import model_feature_columns
    from .evaluate import last_n_days
    from .metrics import all_metrics

    loaded = mlflow.pyfunc.load_model(model_uri)
    frame = last_n_days(features, int(params["split"]["val_days"]))
    columns = model_feature_columns(frame, params)
    predictions = loaded.predict(frame[columns])
    return all_metrics(frame["sales"], predictions)


def promote_from_run(
    run_id: str, params: dict | None = None, features: pd.DataFrame | None = None
) -> dict:
    """Register, re-score production, apply the gate and write a decision file."""
    import mlflow

    params = params or load_params()
    summary = json.loads((ROOT / "reports" / "train_summary.json").read_text(encoding="utf-8"))
    candidate_metrics = (
        summary["models"][run_id]["metrics"] if run_id in summary["models"] else None
    )
    if candidate_metrics is None:
        for entry in summary["models"].values():
            if entry["run_id"] == run_id:
                candidate_metrics = entry["metrics"]
                break
    evaluation = json.loads((ROOT / "reports" / "evaluation.json").read_text(encoding="utf-8"))
    baseline_metrics = {
        name: values
        for name, values in evaluation["metrics"].items()
        if values["model_type"] == "baseline"
    }
    if features is None:
        features = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")

    version = register_model(run_id, params)
    set_alias(version, "candidate", params)
    production = get_production_model_info(params)
    production_metrics = None
    if production and production["version"] != version:
        try:
            production_metrics = score_model_on_validation(
                f"models:/{params['mlflow']['registered_model_name']}@{params['mlflow']['production_alias']}",
                features,
                params,
            )
        except Exception as error:  # noqa: BLE001
            production_metrics = None
            print(f"Could not re-score production model: {error}")

    approved, reason = should_promote(
        candidate_metrics, production_metrics, baseline_metrics, params
    )
    if approved:
        set_alias(version, params["mlflow"]["production_alias"], params)
    decision = {
        "run_id": run_id,
        "candidate_version": version,
        "candidate_metrics": candidate_metrics,
        "production": production,
        "production_metrics": production_metrics,
        "baseline_metrics": baseline_metrics,
        "approved": approved,
        "reason": reason,
    }
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    # Keep the training experiment at exactly one run per model; decisions are
    # audited in their own experiment.
    mlflow.set_experiment(params["mlflow"]["promotion_experiment_name"])
    with mlflow.start_run(run_name="promotion_decision"):
        mlflow.set_tags({"decision": str(approved), "candidate_version": version, "reason": reason})
        mlflow.log_dict(decision, "promotion_decision.json")
    (ROOT / "reports" / "promotion_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(reason)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    promote_from_run(args.run_id)


if __name__ == "__main__":
    main()
