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
- 2026-11-24 — Verification — `ruff check`/`format --check` clean; 59/59 unit and API tests pass; coverage 83.18% (target 80%).
