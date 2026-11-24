# Resume numbers (measured artifacts)

These values are computed from real artifacts by `scripts/`—never invented.

- Dataset used for reported metrics: `synthetic` (see `docs/DECISIONS.md`).
- MAPE improvement vs moving-average baseline: `36.0%` — computed as
  `(19.629 - 12.556) / 19.629 * 100` from `reports/model_comparison.md`.
- Best learned model and MAE: `hgb`, `2.120` — from `reports/model_comparison.md`.
- Validation rows: `45,000` — from `reports/evaluation.json`, window 2018-10-03 to 2018-12-31.
- Median API latency: `1,074.7 ms` — `reports/latency.json` field `p50_ms`, 1,000 requests at
  concurrency 10, 0 errors, 9.06 req/s (local single-worker measurement, not a production benchmark).

Copy/paste template:

> Built an end-to-end demand forecasting MLOps pipeline (DVC, MLflow, Airflow, FastAPI, drift monitoring)
> on 45,000 validation rows (913,000 processed daily series-day rows); HGB reduced MAPE by 36.0%
> versus the moving-average baseline, and the served model returned predictions with 1,075 ms median latency.
