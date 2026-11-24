"""Validation-window evaluation helpers (baselines and all trained models)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import ROOT, load_params
from .baselines import LastWeekBaseline, MovingAverageBaseline
from .metrics import all_metrics


def last_n_days(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    """Return rows from the final ``days`` calendar days of a frame."""
    cutoff = frame["date"].max() - pd.Timedelta(days=days - 1)
    return frame[frame["date"] >= cutoff]


def baseline_features(processed: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Derive the two baseline inputs from past sales only (no same-day sales)."""
    params = params or load_params()
    lag = int(params["baselines"]["last_week_lag"])
    window = int(params["baselines"]["moving_average_window"])
    frame = processed.copy()
    grouped = frame.groupby(["store_id", "item_id"], sort=False)["sales"]
    frame[f"lag_{lag}"] = grouped.shift(lag)
    frame[f"roll_mean_{window}"] = grouped.transform(
        lambda series: series.shift(1).rolling(window=window, min_periods=window).mean()
    )
    return frame


def evaluate_baselines(
    processed: pd.DataFrame,
    params: dict | None = None,
    out_path: str | Path = "reports/baseline_metrics.json",
) -> dict:
    """Score both baselines on the same latest validation window and persist it."""
    params = params or load_params()
    features = baseline_features(processed, params)
    validation = last_n_days(features, int(params["split"]["val_days"]))
    results: dict[str, dict[str, float]] = {}
    for model in (LastWeekBaseline(params=params), MovingAverageBaseline(params=params)):
        rows = validation.dropna(subset=[model.column])
        results[model.name] = all_metrics(rows["sales"], model.predict(rows))

    destination = ROOT / out_path if not Path(out_path).is_absolute() else Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def evaluate_saved_predictions(params: dict | None = None) -> dict:
    """Recompute metrics from every model's saved validation predictions."""
    params = params or load_params()
    summary_path = ROOT / "reports" / "train_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = {}
    for name, entry in summary["models"].items():
        path = ROOT / "reports" / "predictions" / f"{name}.csv"
        frame = pd.read_csv(path)
        results[name] = {
            **all_metrics(frame["actual"], frame["prediction"]),
            "run_id": entry["run_id"],
            "model_type": entry["model_type"],
        }
    output = {
        "validation_range": summary["validation_range"],
        "validation_rows": summary["validation_rows"],
        "feature_list": summary["feature_list"],
        "data_version": summary["data_version"],
        "dataset": summary["dataset"],
        "metrics": results,
    }
    (ROOT / "reports" / "evaluation.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    params = load_params()
    processed = pd.read_parquet(ROOT / params["data"]["processed_path"])
    baseline_metrics = evaluate_baselines(processed, params)
    result = evaluate_saved_predictions(params)
    print(f"Baselines: {baseline_metrics}")
    print(
        f"Evaluated {len(result['metrics'])} models on {result['validation_rows']:,} identical rows"
    )


if __name__ == "__main__":
    main()
