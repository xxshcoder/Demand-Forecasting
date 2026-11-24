from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.monitoring import report as report_module


def test_report_writes_markdown_and_html(tmp_path, params, monkeypatch):
    reference = pd.DataFrame({"store_id": np.ones(100), "lag_1": np.arange(100) % 10})
    reference.to_parquet(tmp_path / "reference.parquet", index=False)
    (tmp_path / "data" / "reference").mkdir(parents=True)
    reference.to_parquet(tmp_path / "data" / "reference" / "train_features.parquet", index=False)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "train_summary.json").write_text(
        json.dumps({"feature_list": ["store_id", "lag_1"]}), encoding="utf-8"
    )
    monkeypatch.setattr(report_module, "ROOT", tmp_path)
    recent = pd.DataFrame({"store_id": np.ones(100), "lag_1": np.arange(100) % 10 + 30})
    monkeypatch.setattr(report_module, "load_recent_predictions", lambda days: recent)

    result = report_module.build_report(params)
    assert result["flagged"] == ["lag_1"]
    markdown = (tmp_path / "reports" / "drift_report.md").read_text(encoding="utf-8")
    assert "lag_1" in markdown and "YES" in markdown
    assert "<html" in (tmp_path / "reports" / "drift_report.html").read_text(encoding="utf-8")
