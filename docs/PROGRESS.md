# Progress log

One line per completed plan step (date — step — result).

- 2026-11-24 — 1.1 Scaffold — repo initialised, layout, params, tooling config, config loader created.
- 2026-11-24 — 1.2 Get data — synthetic deterministic dataset generated (908,442 rows) after no Kaggle credentials were available locally; decision logged.
- 2026-11-24 — 1.3 EDA notebook — exploration notebook with trend/seasonality/distribution plots and findings created.
- 2026-11-24 — 1.4 Cleaning and validation — load/clean/validate/prepare modules and unit tests created; targeted tests pass.
- 2026-11-24 — 1.5 DVC — repository initialised, local remote configured, raw CSV added to DVC (`.dvc` files un-ignored so Git tracks them).
- 2026-11-24 — 1.6 Metrics and baselines — masked MAPE/WAPE, last-week and moving-average baselines with tests; verified in the full suite.
- 2026-11-24 — 2.1–2.4 — features, leakage/parity tests, sklearn models, PyTorch MLP with train-only early stopping, MLflow training, comparison report implemented.
- 2026-11-24 — 3.2–3.5 — promotion gate, FastAPI service with registry loader/polling, prediction logging, Airflow DAG, Dockerfiles and Compose stack implemented.
- 2026-11-24 — 4.1–4.4 — PSI/KS drift monitoring, report generation, load-test script, CI workflow implemented.
- 2026-11-24 — Verification — `ruff check`/`format --check` clean; 59/59 unit and API tests pass; coverage 83.17% (target 80%).
- 2026-11-24 — 2.5 Full pipeline — `dvc repro` rebuilt prepare → features → train → evaluate → compare from raw data; five MLflow runs logged; comparison table generated from 45,000 identical validation rows (HGB MAE 2.120 vs moving-average 3.351).
- 2026-11-24 — DVC fix — `reports/baseline_metrics.json` moved from the `train` stage to `evaluate`; `compare.py` no longer rewrites the `evaluate` stage's `evaluation.json`, preventing a permanently dirty pipeline. `dvc status` clean, `dvc repro` no-op, `dvc push` uploaded 15 files.
- 2026-11-24 — 3.2 Promotion verified — HGB promoted to `production` (v1, beats moving average); a second run with linear (v2) was refused (`-22.08%` relative MAE). Decisions audited in a separate MLflow experiment.
- 2026-11-24 — 3.3 Serving verified — local uvicorn served `/health`, `/model-info`, `/predict` (prediction 7.36) and `/admin/reload`; 27-value input returned 422. Registry alias v1 → v2 → v1 swapped with **no API restart**.
- 2026-11-24 — 4.1 Drift verified — 100 sampled in-distribution requests produced no drift; 100 sales×2 requests flagged all 10 sales-derived features in `reports/drift_report.md`.
- 2026-11-24 — 4.3 Latency measured — 1,000 requests / concurrency 10, 0 errors, p50 1,074.7 ms, p95 1,498.4 ms, p99 1,725.4 ms, 9.06 req/s in `reports/latency.json`.
- 2026-11-24 — Phase merges — phase 2 (features/models) and phase 3 (registry/serving/DAG) verified and merged to `main`.
- 2026-11-24 — DVC fix — report artifacts removed from DVC outputs so Git can commit them; `train` deps narrowed to the exact model files; `dvc status` clean, `dvc repro` no-op and `dvc push` up to date.
- 2026-11-24 — Final cleanup — lint/format clean and 59/59 tests at 83.17% coverage; Docker/Postgres/Kaggle limitations recorded in `docs/DECISIONS.md` and `docs/HANDOFF.md`; tagged `v1.0.0`.
- 2026-11-24 — Documentation — expanded `README.md` with architecture, setup paths, configuration, pipeline stages, API/DAG/monitoring guides, artifacts, measured results, verification status and troubleshooting.
- 2026-11-24 — Documentation — removed the public Human handoff section from `README.md`; the detailed handoff remains in `docs/HANDOFF.md`.
