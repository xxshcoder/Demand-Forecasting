"""Train every model, log one MLflow run per model and write a run summary."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from ..config import ROOT, get_dataset_label, get_mlflow_tracking_uri, load_params, resolve_path
from ..data.prepare import prepare as prepare_data
from ..features.build import build_features, model_feature_columns
from ..features.split import time_split
from .baselines import LastWeekBaseline, MovingAverageBaseline, make_baseline_pyfunc
from .metrics import all_metrics as compute_metrics
from .nn import TorchForecaster, make_nn_pyfunc
from .sklearn_models import (
    build_hgb_pipeline,
    build_linear_pipeline,
    make_sklearn_pyfunc,
    predict_array,
)


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:  # pragma: no cover - git not installed
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _data_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path.exists():
            continue
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:16]


def _flatten(prefix: str, values: dict) -> dict:
    result = {}
    for key, value in values.items():
        if isinstance(value, dict):
            result.update(_flatten(f"{prefix}.{key}", value))
        elif isinstance(value, (list, tuple)):
            result[f"{prefix}.{key}"] = ",".join(str(item) for item in value)
        else:
            result[f"{prefix}.{key}"] = value
    return result


def load_feature_frame(params: dict) -> pd.DataFrame:
    """Load built features, building them on demand for a local `make train`."""
    path = resolve_path("data/processed/features.parquet")
    if path.exists():
        return pd.read_parquet(path)
    processed_path = resolve_path(params["data"]["processed_path"])
    if not processed_path.exists():
        prepare_data(params=params)
    processed = pd.read_parquet(processed_path)
    features = build_features(processed, params)
    warmup = [c for c in features.columns if c.startswith("lag_") or c.startswith("roll_")]
    features = features.dropna(subset=warmup).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False)
    # The drift monitor needs the training-window distribution; save it here
    # too so a direct `make train` works without the DVC features stage.
    reference = resolve_path("data/reference/train_features.parquet")
    reference.parent.mkdir(parents=True, exist_ok=True)
    train_window, _, _ = time_split(
        features, int(params["split"]["val_days"]), int(params["split"]["test_days"])
    )
    train_window.to_parquet(reference, index=False)
    return features


def _log_run(
    model_name: str,
    model_type: str,
    model,
    feature_names: list[str],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    predictions,
    metrics: dict,
    params: dict,
    run_meta: dict,
) -> str:
    cfg = params
    with mlflow.start_run(run_name=model_name) as active:
        run_id = active.info.run_id
        logged = _flatten(
            "",
            {k: cfg[k] for k in ("data", "features", "split", "baselines", "models", "promotion")},
        )
        logged["feature_list"] = ",".join(feature_names)
        mlflow.log_params(logged)
        mlflow.log_metrics({f"val_{key}": float(value) for key, value in metrics.items()})
        mlflow.set_tags(
            {
                "git_sha": run_meta["git_sha"],
                "dvc_data_hash": run_meta["data_hash"],
                "model_type": model_type,
                "dataset": run_meta["dataset"],
            }
        )
        mlflow.log_dict({"feature_list": list(feature_names)}, "feature_list.json")
        sample = X_val.head(int(run_meta["sample_rows"]))
        sample["prediction"] = predictions[: len(sample)]
        sample["actual"] = y_val.head(len(sample)).to_numpy()
        sample.to_csv("predictions_sample.csv", index=False)
        mlflow.log_artifact("predictions_sample.csv")
        Path("predictions_sample.csv").unlink(missing_ok=True)
        reference = resolve_path("data/reference/train_features.parquet")
        if reference.exists():
            mlflow.log_artifact(str(reference))
        if model is not None:
            signature = mlflow.models.infer_signature(
                X_val.head(run_meta["sample_rows"]), predictions[: run_meta["sample_rows"]]
            )
            if model_type == "nn":
                with tempfile.TemporaryDirectory() as tmp:
                    model.save(Path(tmp) / "model.pt")
                    mlflow.pyfunc.log_model(
                        name="model",
                        python_model=make_nn_pyfunc(model),
                        artifacts={"nn_model": str(Path(tmp) / "model.pt")},
                        signature=signature,
                        input_example=X_val.head(run_meta["sample_rows"]),
                    )
            elif model_type == "baseline":
                mlflow.pyfunc.log_model(
                    name="model",
                    python_model=make_baseline_pyfunc(model),
                    signature=signature,
                    input_example=X_val.head(run_meta["sample_rows"]),
                )
            else:
                mlflow.pyfunc.log_model(
                    name="model",
                    python_model=make_sklearn_pyfunc(model),
                    signature=signature,
                    input_example=X_val.head(run_meta["sample_rows"]),
                )
    return run_id


def train_all(params: dict | None = None) -> dict:
    """Fit all five models on the same training window and log them to MLflow."""
    params = params or load_params()
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    mlflow.set_registry_uri(get_mlflow_tracking_uri())
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    features = load_feature_frame(params)
    train, validation, _ = time_split(
        features, int(params["split"]["val_days"]), int(params["split"]["test_days"])
    )
    feature_columns = model_feature_columns(features, params)
    X_train, y_train = train[feature_columns], train["sales"]
    X_val, y_val = validation[feature_columns], validation["sales"]

    run_meta = {
        "git_sha": _git_sha(),
        "data_hash": _data_hash(
            [
                resolve_path(params["data"]["raw_path"]),
                resolve_path(params["data"]["processed_path"]),
            ]
        ),
        "dataset": get_dataset_label(),
        "sample_rows": 5,
    }
    summary = {
        "tracking_uri": get_mlflow_tracking_uri(),
        "experiment": params["mlflow"]["experiment_name"],
        "registered_model_name": params["mlflow"]["registered_model_name"],
        "feature_list": feature_columns,
        "data_version": run_meta["data_hash"],
        "git_sha": run_meta["git_sha"],
        "dataset": run_meta["dataset"],
        "train_range": [str(train["date"].min()), str(train["date"].max())],
        "validation_range": [str(validation["date"].min()), str(validation["date"].max())],
        "validation_rows": int(len(validation)),
        "models": {},
    }
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    predictions_dir = reports / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("last_week", "baseline", LastWeekBaseline(params=params)),
        ("moving_average", "baseline", MovingAverageBaseline(params=params)),
        ("linear", "sklearn", build_linear_pipeline(feature_columns, params)),
        ("hgb", "sklearn", build_hgb_pipeline(feature_columns, params)),
        ("nn", "nn", TorchForecaster(params, feature_columns)),
    ]
    for name, model_type, model in specs:
        if model_type == "baseline":
            predictions = predict_array(model, X_val)
        elif model_type == "sklearn":
            model.fit(X_train, y_train)
            predictions = predict_array(model, X_val)
        else:
            model.fit(train, y_train)
            predictions = predict_array(model, X_val)
        metrics = compute_metrics(y_val, predictions)
        run_id = _log_run(
            name,
            model_type,
            model,
            feature_columns,
            X_val,
            y_val,
            predictions,
            metrics,
            params,
            run_meta,
        )
        pd.DataFrame(
            {
                "date": validation["date"].to_numpy(),
                "actual": y_val.to_numpy(),
                "prediction": predictions,
            }
        ).to_csv(predictions_dir / f"{name}.csv", index=False)
        summary["models"][name] = {
            "run_id": run_id,
            "model_type": model_type,
            "metrics": metrics,
            "model_uri": f"runs:/{run_id}/model",
        }
        print(
            f"{name:16s} MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} "
            f"MAPE={metrics['mape']:.2f}%"
        )

    (reports / "train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    train_all()


if __name__ == "__main__":
    main()
