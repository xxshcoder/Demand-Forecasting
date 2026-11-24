"""Build markdown/HTML drift reports from prediction logs and reference data."""

from __future__ import annotations

import base64
import io
import json
from datetime import UTC, datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import select  # noqa: E402

from ..config import ROOT, load_params  # noqa: E402
from ..serving.logging_db import get_engine, prediction_logs  # noqa: E402
from .drift import check_drift  # noqa: E402


def load_recent_predictions(days: int) -> pd.DataFrame:
    """Read the most recent logged feature vectors from the app database."""
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    try:
        with get_engine().connect() as connection:
            rows = (
                connection.execute(select(prediction_logs).where(prediction_logs.c.ts >= cutoff))
                .mappings()
                .all()
            )
    except Exception as error:  # noqa: BLE001
        print(f"Prediction log unavailable: {error}")
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    features = [json.loads(row["features_json"]) for row in rows]
    return pd.DataFrame(features)


def _histogram(reference: pd.Series, recent: pd.Series) -> str:
    figure, axis = plt.subplots(figsize=(5, 2.4))
    axis.hist(reference.dropna(), bins=20, alpha=0.6, label="reference")
    axis.hist(recent.dropna(), bins=20, alpha=0.6, label="recent")
    axis.set_title("reference vs recent")
    axis.legend(fontsize=8)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def build_report(params: dict | None = None) -> dict:
    params = params or load_params()
    summary_path = ROOT / "reports" / "train_summary.json"
    feature_list = []
    if summary_path.exists():
        feature_list = json.loads(summary_path.read_text(encoding="utf-8")).get("feature_list", [])
    reference_path = ROOT / "data" / "reference" / "train_features.parquet"
    if not reference_path.exists():
        raise FileNotFoundError("Reference data missing; run the features stage first")
    reference = pd.read_parquet(reference_path)
    recent = load_recent_predictions(int(params["monitoring"]["recent_window_days"]))
    features = [column for column in feature_list if column in reference.columns]
    results = check_drift(reference, recent, features, params) if not recent.empty else {}
    flagged = [name for name, values in results.items() if values["drift"]]

    lines = [
        "# Drift report",
        "",
        f"Reference rows: {len(reference):,}. Recent logged rows: {len(recent):,}.",
        "" if recent.empty else f"Drift flagged for: {', '.join(flagged) if flagged else 'none'}.",
        "",
        "| feature | PSI | KS statistic | KS p-value | drift |",
        "|---|---:|---:|---:|:---:|",
    ]
    for name, values in results.items():
        lines.append(
            f"| {name} | {values['psi']:.4f} | {values['ks_statistic']:.4f} | "
            f"{values['ks_pvalue']:.4g} | {'YES' if values['drift'] else 'no'} |"
        )
    if not results:
        lines.append("| _no recent predictions_ | - | - | - | - |")
    markdown = "\n".join(lines) + "\n"
    (ROOT / "reports" / "drift_report.md").write_text(markdown, encoding="utf-8")

    html_rows = "\n".join(
        f"<tr><td>{name}</td><td>{v['psi']:.4f}</td><td>{v['ks_statistic']:.4f}</td>"
        f"<td>{v['ks_pvalue']:.4g}</td><td>{'YES' if v['drift'] else 'no'}</td></tr>"
        for name, v in results.items()
    )
    plots = ""
    for name in features[:6]:
        if name in recent.columns:
            plots += f"<h3>{name}</h3><img src='{_histogram(reference[name], recent[name])}'/>"
    html = (
        "<html><head><title>Drift report</title></head><body><h1>Drift report</h1>"
        f"<p>Reference rows: {len(reference):,}; recent rows: {len(recent):,}.</p>"
        "<table border='1'><tr><th>feature</th><th>PSI</th><th>KS statistic</th><th>KS p-value</th>"
        f"<th>drift</th></tr>{html_rows}</table>{plots}</body></html>"
    )
    (ROOT / "reports" / "drift_report.html").write_text(html, encoding="utf-8")
    if flagged:
        print(f"WARNING: drift detected in {len(flagged)} features: {', '.join(flagged)}")
    else:
        print("No drift detected (or no recent predictions).")
    return {"features": results, "flagged": flagged, "recent_rows": len(recent)}


def main() -> None:
    build_report()


if __name__ == "__main__":
    main()
