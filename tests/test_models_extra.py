from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

from src.features.build import build_features, model_feature_columns
from src.models import compare, evaluate
from src.models.sklearn_models import build_hgb_pipeline, build_linear_pipeline, predict_array


def _trained_features(tiny_processed, params):
    cfg = copy.deepcopy(params)
    cfg["models"]["hgb"]["max_iter"] = 5
    frame = build_features(tiny_processed, cfg).dropna().reset_index(drop=True)
    return frame, model_feature_columns(frame, cfg), cfg


def test_sklearn_model_factories_fit_and_predict(tiny_processed, params):
    frame, columns, cfg = _trained_features(tiny_processed, params)
    X, y = frame[columns], frame["sales"]
    for model in (build_linear_pipeline(columns, cfg), build_hgb_pipeline(columns, cfg)):
        model.fit(X, y)
        predictions = predict_array(model, X)
        assert predictions.shape == (len(X),)
        assert np.isfinite(predictions).all()
        assert (predictions >= 0).all()


def test_evaluate_and_compare_with_saved_predictions(tiny_processed, params, monkeypatch, tmp_path):
    frame, columns, cfg = _trained_features(tiny_processed, params)
    reports = tmp_path / "reports"
    (reports / "predictions").mkdir(parents=True)
    summary = {
        "validation_range": ["2024-03-02", "2024-03-10"],
        "validation_rows": len(frame),
        "feature_list": columns,
        "data_version": "hash",
        "dataset": "synthetic",
        "models": {},
    }
    for name, model_type in [
        ("last_week", "baseline"),
        ("moving_average", "baseline"),
        ("linear", "sklearn"),
        ("hgb", "sklearn"),
        ("nn", "nn"),
    ]:
        summary["models"][name] = {"run_id": f"run-{name}", "model_type": model_type, "metrics": {}}
        pd.DataFrame(
            {
                "date": frame["date"],
                "actual": frame["sales"],
                "prediction": frame["sales"].to_numpy() + (1 if name == "nn" else 2),
            }
        ).to_csv(reports / "predictions" / f"{name}.csv", index=False)
    (reports / "train_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    monkeypatch.setattr(compare, "ROOT", tmp_path)

    result = evaluate.evaluate_saved_predictions(cfg)
    assert set(result["metrics"]) == set(summary["models"])
    table = compare.build_comparison(cfg)
    assert table.loc["moving_average", "MAE"] == 2
    compare.main()
    assert (reports / "model_comparison.csv").exists()
    comparison_md = (reports / "model_comparison.md").read_text(encoding="utf-8")
    assert "Best learned model" in comparison_md
    # evaluate.json belongs to the evaluate stage; compare must not mutate it.
    assert "interpretation" not in (reports / "evaluation.json").read_text(encoding="utf-8")


def test_baseline_metrics_writer(tiny_processed, params, monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    result = evaluate.evaluate_baselines(
        tiny_processed, params, out_path=tmp_path / "baselines.json"
    )
    assert set(result) == {"last_week", "moving_average"}
    assert all(np.isfinite(value) for values in result.values() for value in values.values())
    assert (tmp_path / "baselines.json").exists()
