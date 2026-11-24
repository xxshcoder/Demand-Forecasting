# plan.md: Demand Forecasting MLOps Pipeline (agent execution plan)

> You are a coding agent. Build this repository from an empty folder to a finished, tested, documented project by executing the steps below **in order**. Do not skip steps. Do not start a step until the previous step's **Verify** block passes.

---

## 0. Operating rules (read first)

1. **Work in small steps.** After each numbered step: run its Verify commands, fix failures, then `git commit` with a conventional message (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`).
2. **Branching:** `main` stays green. Create a short-lived branch per phase (`phase-1-data`, `phase-2-models`, ...), merge to `main` when the phase's Verify passes.
3. **Never commit secrets or data.** Secrets live in `.env` (git-ignored), documented in `.env.example`. Data is tracked by DVC, not Git.
4. **All hyperparameters and settings live in `params.yaml`.** No magic numbers in code.
5. **No business logic in notebooks.** Notebooks are exploration only; anything reusable goes in `src/` with tests.
6. **Never leak the future.** Time-based splits only. Every feature at time `t` may use data from `t-1` and earlier only. Add a test proving it.
7. **Be honest in results.** If the neural network does not beat gradient boosting or the baselines, report that. Never fabricate metrics, latency numbers or screenshots; leave a clearly marked placeholder for the human instead.
8. **Run everything through the Makefile** where possible (`make test`, `make lint`, `make up`).
9. **If blocked** (missing credentials, no network, ambiguous requirement): choose the documented fallback in this file, note it in `docs/DECISIONS.md`, and continue. Only stop for items listed under "Human handoff" (section 10).
10. **Keep a running log** in `docs/PROGRESS.md`: one line per completed step with date, step id and result.

---

## 1. Goal

Predict daily unit sales per store and item, and prove the full ML lifecycle:

- versioned data (DVC)
- tracked experiments and a model registry (MLflow)
- honest comparison: baselines vs scikit-learn vs one neural network
- a served prediction API (FastAPI) that loads the production model from the registry
- scheduled retraining (Airflow) with a **promotion gate** that only promotes better models
- prediction logging and drift monitoring
- tests, CI, Docker Compose, README with real numbers

Forecast task definition (fixed, do not change): **one-day-ahead forecast** of `sales` for a `(store_id, item_id)` pair using features known at the end of the previous day.

---

## 2. Tech decisions (fixed)

| Area | Choice |
|---|---|
| Language | Python 3.11 |
| Packaging | `pyproject.toml` + `requirements.txt` and `requirements-dev.txt` (pinned versions) |
| Data / features | pandas, numpy, pyarrow (parquet) |
| Data validation | pandera (or pydantic-based checks if pandera install fails) |
| Data versioning | DVC (remote: local folder `../dvc-remote` by default, S3-compatible optional) |
| Models | scikit-learn: `LinearRegression`/`Ridge`, `HistGradientBoostingRegressor`; neural net: **PyTorch MLP** (choose PyTorch, not Keras) |
| Tracking / registry | MLflow server, Postgres backend store, artifacts on a shared volume. Use **registry aliases** (`production`, `candidate`), not deprecated stages |
| Serving | FastAPI, Pydantic v2, uvicorn |
| Orchestration | Airflow 2.x (LocalExecutor, Postgres) |
| Prediction log + app DB | Postgres (table `prediction_logs`) via SQLAlchemy |
| Drift | PSI and Kolmogorov-Smirnov (scipy) |
| Tests | pytest, pytest-cov, httpx/TestClient |
| Lint/format | ruff (lint + format), mypy optional |
| CI | GitHub Actions |
| Containers | Docker, Docker Compose |

Registered model name: `demand-forecaster`. Production alias: `production`.

---

## 3. Final repository layout

Create exactly this structure (extra files allowed where noted).

```
demand-forecasting-mlops/
├── .github/workflows/ci.yml
├── data/                      # DVC-tracked, git-ignored (raw/, processed/, reference/)
├── src/
│   ├── __init__.py
│   ├── config.py              # loads params.yaml + env vars
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py            # read raw csv -> DataFrame
│   │   ├── clean.py           # cleaning functions (tested)
│   │   ├── validate.py        # schema + rules
│   │   └── prepare.py         # CLI: raw -> processed (used by dvc)
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build.py           # feature engineering (batch + single-row API path)
│   │   └── split.py           # time-based split
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baselines.py       # last-week, moving average
│   │   ├── metrics.py         # MAE, RMSE, MAPE (safe)
│   │   ├── sklearn_models.py
│   │   ├── nn.py              # PyTorch MLP
│   │   ├── train.py           # trains all models, logs to MLflow
│   │   ├── evaluate.py        # evaluates on validation window
│   │   ├── compare.py         # comparison table (csv + markdown)
│   │   └── promote.py         # promotion gate + registry ops
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py             # FastAPI app
│   │   ├── schemas.py         # Pydantic models
│   │   ├── model_loader.py    # loads models:/demand-forecaster@production, auto-refresh
│   │   └── logging_db.py      # prediction log writer
│   └── monitoring/
│       ├── __init__.py
│       ├── drift.py           # PSI + KS
│       └── report.py          # builds HTML/markdown drift report
├── dags/
│   └── retrain_dag.py
├── scripts/
│   ├── make_synthetic_data.py # offline/CI dataset generator
│   ├── download_data.sh       # real dataset download helper
│   ├── load_test.py           # latency measurement
│   └── seed_demo_traffic.py   # sends sample requests to fill prediction logs
├── tests/
│   ├── conftest.py
│   ├── fixtures/              # tiny CSVs
│   ├── test_clean.py
│   ├── test_validate.py
│   ├── test_features.py
│   ├── test_split.py
│   ├── test_baselines_metrics.py
│   ├── test_promote.py
│   ├── test_api.py
│   └── test_drift.py
├── notebooks/01_eda.ipynb     # exploration only
├── docs/{PROGRESS.md,DECISIONS.md,images/}
├── reports/                   # metrics json, comparison table, drift report (git-tracked small outputs)
├── dvc.yaml
├── params.yaml
├── docker-compose.yml
├── Dockerfile                 # API image
├── Dockerfile.airflow         # Airflow image with project deps
├── Makefile
├── .env.example
├── .gitignore  .dockerignore  .dvcignore
├── LICENSE (MIT)
└── README.md
```

---

## 4. Contracts (define once, follow everywhere)

### 4.1 Dataset and schema

**Primary dataset:** Kaggle "Store Item Demand Forecasting Challenge" (`train.csv`: `date, store, item, sales`; 10 stores, 50 items, 5 years of daily data). Record source URL and license in README.
**Fallback (no network or no Kaggle credentials):** `scripts/make_synthetic_data.py` generates a realistic dataset with the same schema: trend, weekly and yearly seasonality, holiday bumps, promo effects, noise, a few missing days and outliers, 10 stores x 50 items x 5 years, seeded for determinism. The pipeline must work identically on both. State clearly in README which dataset produced the reported numbers.

Canonical raw schema: `date` (YYYY-MM-DD), `store` (int), `item` (int), `sales` (int >= 0). Optional columns when present: `price`, `promo`.

Processed schema (`data/processed/sales.parquet`): `date, store_id, item_id, sales, price (nullable), promo (0/1, default 0)`, sorted by `store_id, item_id, date`, one row per series per day (missing days re-inserted and handled per `params.yaml`).

### 4.2 Features (all computed per `store_id, item_id` series, using only past data)

- lags of `sales`: `lag_1, lag_7, lag_14, lag_28`
- rolling mean and std over windows 7, 14, 28, computed on `sales.shift(1)` (so the current day is never included): `roll_mean_7, roll_std_7, roll_mean_14, ...`
- calendar: `dow`, `month`, `day_of_month`, `is_weekend`, `is_holiday` (use the `holidays` package, country set in params)
- `price`, `promo` if available
- identifiers: `store_id`, `item_id` (as categorical/embedding or one-hot per model; document choice)

The feature list is stored in `params.yaml` and logged to MLflow. There must be **one** feature function used by both training (batch) and serving (single request) to avoid train/serve skew. Add a test asserting both paths give identical values.

### 4.3 Split

Time-based. From `params.yaml`: `val_days: 90` (last 90 days = validation), optional `test_days: 90` before it. Train strictly precedes validation, no overlap, assert in code and in a test. Drop warm-up rows where lags are NaN.

### 4.4 Metrics

MAE, RMSE, MAPE (MAPE must handle zeros: use masked MAPE over `y>0` and also report sMAPE or WAPE). **Primary promotion metric: MAE** (set in `params.yaml`). All models are evaluated on the **identical** validation window and rows.

### 4.5 API contract

`POST /predict` request:
```json
{
  "store_id": 1,
  "item_id": 5,
  "forecast_date": "2018-01-01",
  "recent_sales": [/* last 28 daily values ending the day before forecast_date, oldest first */],
  "price": 9.99,
  "promo": 0
}
```
Validation: `len(recent_sales) == 28`, values >= 0, `store_id`/`item_id` ints, date is ISO. Response:
```json
{"prediction": 42.7, "model_name": "demand-forecaster", "model_version": "7", "request_id": "uuid", "timestamp": "..."}
```
`GET /health` -> `{"status":"ok","model_loaded":true}` (HTTP 503 if no model).
`GET /model-info` -> version, alias, training date, run_id, validation metrics, feature list, data version (DVC hash).
`POST /admin/reload` (optional, token-protected via env) forces a registry re-check. The loader also polls the alias every `MODEL_REFRESH_SECONDS` (default 60) so a newly promoted model is served **without restarting or code changes**.

### 4.6 Promotion gate (core logic in `src/models/promote.py`, pure and unit-testable)

```
should_promote(candidate_metrics, production_metrics, baseline_metrics, params) -> (bool, reason)
```
Rules, all must hold:
1. Candidate beats the best baseline on the primary metric.
2. If no production model exists: promote (after rule 1).
3. Otherwise candidate primary metric must be lower than production by at least `min_relative_improvement` (default 0.5%).
4. Candidate must not be worse than production by more than `max_secondary_regression` (default 5%) on RMSE.
5. NaN/inf metrics always refuse.
Both models are scored on the **same, latest** validation window at promotion time (re-score the current production model on it, do not trust old stored metrics). Return a human-readable reason and log it to MLflow and Airflow logs.

---

## 5. `params.yaml` (create with these keys; adjust values only with a reason in DECISIONS.md)

```yaml
data:
  raw_path: data/raw/train.csv
  processed_path: data/processed/sales.parquet
  fill_missing_days: true
  outlier_clip_quantile: 0.999
  country_holidays: US
features:
  lags: [1, 7, 14, 28]
  rolling_windows: [7, 14, 28]
  use_price: true
  use_promo: true
split:
  val_days: 90
  test_days: 90
baselines:
  moving_average_window: 7
  last_week_lag: 7
models:
  linear: { alpha: 1.0 }
  hgb: { max_iter: 300, learning_rate: 0.1, max_depth: null, random_state: 42 }
  nn: { hidden: [128, 64], dropout: 0.1, lr: 0.001, epochs: 30, batch_size: 1024, seed: 42, device: cpu }
mlflow:
  experiment_name: demand-forecasting
  registered_model_name: demand-forecaster
  production_alias: production
promotion:
  primary_metric: mae
  min_relative_improvement: 0.005
  max_secondary_regression: 0.05
monitoring:
  psi_threshold: 0.2
  ks_pvalue_threshold: 0.01
  recent_window_days: 7
serving:
  model_refresh_seconds: 60
```

---

## 6. Phases and steps

### PHASE 1: Foundation, data and baselines (branch `phase-1-data`)

**1.1 Scaffold**
- `git init`, create the full layout from section 3 (empty `__init__.py` files, placeholders).
- Write `.gitignore` (data/, .env, mlruns/, __pycache__, .venv, *.parquet, dvc cache etc.), `.dockerignore`, `LICENSE`, `.env.example`, `Makefile` with targets: `setup`, `lint`, `format`, `test`, `data`, `train`, `up`, `down`, `logs`, `serve-local`, `drift`, `load-test`.
- Create venv, pin dependencies in `requirements.txt` / `requirements-dev.txt`.
- `src/config.py`: `load_params()` reads `params.yaml`; env helpers for `MLFLOW_TRACKING_URI`, `DATABASE_URL`.
- **Verify:** `make setup && make lint && pytest -q` (zero tests collected is fine at this point but must not error).

**1.2 Get data**
- `scripts/download_data.sh`: tries Kaggle CLI (`kaggle competitions download -c demand-forecasting-kernels-only` or the correct slug for the store-item dataset), prints clear instructions if credentials are missing.
- `scripts/make_synthetic_data.py --out data/raw/train.csv --seed 42` per section 4.1.
- If real data is not available within one attempt, use synthetic and log the decision in `docs/DECISIONS.md`.
- **Verify:** `data/raw/train.csv` exists with columns `date, store, item, sales`, >= 100k rows.

**1.3 EDA notebook**
- `notebooks/01_eda.ipynb`: trend, weekly/yearly seasonality, missing values, outliers, per-store/item distribution. Save 3 to 4 plots to `docs/images/`. Findings (3 to 5 bullets) go to `docs/DECISIONS.md`. No reusable logic stays in the notebook.

**1.4 Cleaning and validation (`src/data/`)**
- `load.py`: read CSV with dtype enforcement, rename `store -> store_id`, `item -> item_id`.
- `clean.py`: sort, drop duplicates, re-insert missing days per series, handle missing sales (forward fill or zero, per params; document), clip outliers at the configured quantile per series, enforce `sales >= 0`.
- `validate.py`: schema checks (types, non-null keys, non-negative sales, date continuity per series, no duplicate `(store, item, date)`), raises `DataValidationError` with a readable message.
- `prepare.py`: CLI `python -m src.data.prepare` writes `data/processed/sales.parquet`.
- **Tests** (`test_clean.py`, `test_validate.py`) on tiny fixtures: duplicates removed, gaps filled, outlier clipped, negative sales rejected, bad schema rejected.
- **Verify:** `pytest tests/test_clean.py tests/test_validate.py -q` passes.

**1.5 DVC**
- `dvc init`; configure remote (`dvc remote add -d local ../dvc-remote`).
- `dvc.yaml` stages: `prepare` (deps: raw data, `src/data/`, params `data`; outs: processed parquet), later `features`, `train`, `evaluate`.
- Track raw data: `dvc add data/raw/train.csv`.
- **Verify:** delete `data/processed/`, run `dvc repro`, processed data is rebuilt; `dvc status` is clean; `dvc push` works.

**1.6 Metrics and baselines**
- `metrics.py` (MAE, RMSE, masked MAPE, WAPE) with tests including zero handling.
- `baselines.py`: `LastWeekBaseline` (value from 7 days ago) and `MovingAverageBaseline` (window from params), same `predict(df)` interface as other models.
- Evaluate baselines on the validation window; write `reports/baseline_metrics.json`.
- **Verify:** `pytest tests/test_baselines_metrics.py -q`; baseline metrics file exists and values are sensible (MAPE not NaN).

Merge phase 1 to `main`.

---

### PHASE 2: Features and models (branch `phase-2-models`)

**2.1 Feature engineering (`src/features/`)**
- `build.py`: `build_features(df, params)` for batch; `build_single_row_features(recent_sales, forecast_date, price, promo, store_id, item_id, params)` for serving. Both call shared helpers.
- `split.py`: `time_split(df, val_days, test_days)` with assertions (train max date < val min date).
- Add `features` stage to `dvc.yaml` (output `data/processed/features.parquet`).
- **Tests** (`test_features.py`, `test_split.py`):
  - lag/rolling values correct on a hand-computed fixture
  - **no-leakage test:** changing `sales` on day `t` does not change any feature of day `t`
  - **train/serve parity test:** batch features equal single-row features for the same series and date
  - split ordering and no overlap
- **Verify:** `pytest tests/test_features.py tests/test_split.py -q`, `dvc repro` runs the features stage.

**2.2 scikit-learn models**
- `sklearn_models.py`: Ridge/Linear regression pipeline (scaler + one-hot ids) and `HistGradientBoostingRegressor` (categorical support for ids). Seeds from params.
- Wrap each as an `mlflow.pyfunc`-compatible or `mlflow.sklearn` model that accepts the **feature dataframe** (document the exact input columns via an MLflow signature).

**2.3 Neural network**
- `nn.py`: PyTorch MLP (hidden sizes, dropout, Adam, MSE or L1 loss, early stopping on validation loss within the **train** period only, never on the final validation window used for comparison; carve a small tail of train for early stopping). Standardise inputs using train statistics only; save the scaler with the model. Fixed seeds; `device` from params.
- Package as an `mlflow.pyfunc` model with a wrapper that applies the scaler and returns a numpy array.
- A tiny-data smoke test: trains 1 epoch on a fixture and predicts without error.

**2.4 Training with MLflow (`train.py`, `evaluate.py`, `compare.py`)**
- `train.py` trains: both baselines (logged for reference), linear, HGB, MLP. For **each** model, one MLflow run in experiment `demand-forecasting` logging:
  - params (all model hyperparameters, split config, feature list)
  - metrics: `val_mae`, `val_rmse`, `val_mape`, `val_wape`
  - tags: `git_sha`, `dvc_data_hash` (hash of the raw+processed data), `model_type`
  - artifacts: the model (with signature and input example), feature list JSON, predictions sample CSV
- `compare.py` builds `reports/model_comparison.csv` and `reports/model_comparison.md` (rows: last-week, moving average, linear, HGB, MLP; cols: MAE, RMSE, MAPE, WAPE) on the identical validation window. Add a one-paragraph honest interpretation line generated from the numbers (e.g. "MLP did not beat HGB").
- Add `train` and `evaluate` stages to `dvc.yaml` with `reports/` as metrics outputs.
- Local run uses `MLFLOW_TRACKING_URI` (default `http://localhost:5000` when Compose is up, else `file:./mlruns`).
- **Verify:** `make train` produces a comparison table; the MLflow UI shows 5 runs with the fields above; best learned model beats the moving-average baseline (if not, investigate for bugs before continuing, and if truly not, document honestly).

Merge phase 2 to `main`.

---

### PHASE 3: Registry, promotion, serving, orchestration (branch `phase-3-serving`)

**3.1 Docker Compose base (needed to test registry)**
- Services: `postgres` (databases `mlflow`, `airflow`, `app`; init script creates them), `mlflow` (server with Postgres backend, artifact root on shared volume `/mlartifacts`, port 5000), `api` (port 8000), `airflow-init`, `airflow-webserver`, `airflow-scheduler` (port 8080). Healthchecks on postgres, mlflow, api. Named volumes for postgres and artifacts. Mount `data/` and `src/` into Airflow containers; share the MLflow artifact volume with API and Airflow.
- `.env.example` documents every variable (`POSTGRES_*`, `MLFLOW_TRACKING_URI`, `DATABASE_URL`, `AIRFLOW__*`, `ADMIN_TOKEN`, `MODEL_REFRESH_SECONDS`).
- **Verify:** `docker compose up -d` -> `curl localhost:5000` (MLflow UI) responds; `make down` cleans up.

**3.2 Promotion logic (`src/models/promote.py`)**
- Implement `should_promote` (section 4.6), `get_production_model_info()`, `register_model(run_id)`, `promote(version)` (set alias `production`), keep alias `candidate` on the newest registered version.
- CLI: `python -m src.models.promote --run-id <id>` registers the candidate, re-scores production on the latest validation window, applies the gate, promotes or refuses, writes `reports/promotion_decision.json`.
- **Tests** (`test_promote.py`), pure unit tests with fake metrics and a mocked registry client:
  - refuses a worse model
  - refuses a model that does not beat baselines
  - refuses improvement below `min_relative_improvement`
  - refuses NaN metrics
  - promotes the first model when none exists (if it beats baseline)
  - promotes a clearly better model
- **Verify:** `pytest tests/test_promote.py -q` passes.

**3.3 FastAPI service (`src/serving/`)**
- `schemas.py` Pydantic v2 request/response models (section 4.5), strict validation, clear 422 messages.
- `model_loader.py`: loads `models:/demand-forecaster@production`, caches in memory, background thread polls alias version every `model_refresh_seconds` and swaps atomically; exposes version, run_id, metrics (read from run), training date. If registry is unreachable at start, API starts in "no model" state and `/health` returns 503 until loaded.
- `app.py`: `/predict`, `/health`, `/model-info`, `/admin/reload`; per-request `request_id`; latency measured and logged; use dependency injection so tests can supply a stub model.
- `logging_db.py`: on each prediction, insert into `prediction_logs` (`id, request_id, ts, store_id, item_id, forecast_date, features_json (input summary: the computed feature vector), prediction, model_version, latency_ms`). Auto-create table. Writing must not break the request if the DB is down (log error, continue).
- **Tests** (`test_api.py`) with a stub model (no MLflow, no DB, or SQLite): valid request 200, bad length 422, negative sales 422, missing model 503, `/model-info` fields, `/health` ok.
- **Verify:** `pytest tests/test_api.py -q`; `uvicorn src.serving.app:app` with a locally registered model answers a curl.

**3.4 Dockerfiles**
- `Dockerfile`: slim Python 3.11 image, non-root user, installs only serving deps, copies `src/` and `params.yaml`, healthcheck, `CMD uvicorn`.
- `Dockerfile.airflow`: Airflow image + project requirements + `src/` on `PYTHONPATH`.
- **Verify:** `docker build -t demand-api .` succeeds; image size noted in README.

**3.5 Airflow DAG (`dags/retrain_dag.py`)**
- DAG id `demand_forecast_retrain`, `schedule="@weekly"`, `catchup=False`, retries 1, manual trigger allowed.
- Tasks in order: `ingest -> validate -> build_features -> train -> evaluate -> compare -> register_and_promote`.
  - `ingest`: pull/refresh raw data (for demo: copy or re-read raw file; optional param to simulate new data by extending the synthetic dataset).
  - `validate`: run `src.data.validate`; fail the DAG on invalid data.
  - `build_features`, `train`, `evaluate`, `compare`: call the same `src` functions the CLI uses (TaskFlow or PythonOperator; pass run IDs via XCom).
  - `register_and_promote`: runs the gate; logs decision and reason.
- Keep task code thin: **logic lives in `src/`**, DAG only orchestrates.
- A DAG import test (`tests/test_dag.py`) checks the DAG loads without errors and has the expected task order.
- **Verify (end-to-end acceptance):**
  1. `docker compose up -d`
  2. Trigger the DAG manually (UI or `airflow dags trigger demand_forecast_retrain`)
  3. A new model version appears in the MLflow registry
  4. First run promotes it to `production`; `curl /model-info` shows that version within `model_refresh_seconds`, with **no API restart or code change**
  5. A second run with worse/equal model is **refused** and the decision reason is logged

Merge phase 3 to `main`.

---

### PHASE 4: Monitoring, tests, CI, docs (branch `phase-4-polish`)

**4.1 Drift monitoring (`src/monitoring/`)**
- `drift.py`: `psi(expected, actual, bins=10)` (quantile bins from reference, epsilon for empty bins) and `ks_test(expected, actual)` via scipy. `check_drift(reference_df, recent_df, features, params)` returns per-feature PSI, KS statistic, p-value, drift flag.
- Reference data: training feature distribution saved at train time to `data/reference/train_features.parquet` (also logged as MLflow artifact on the run).
- `report.py`: reads recent rows from `prediction_logs` (last `recent_window_days`) plus reference, outputs `reports/drift_report.md` and `reports/drift_report.html` (table + simple histograms) and exits non-zero or prints a warning if thresholds are exceeded. CLI: `python -m src.monitoring.report`; Makefile target `make drift`.
- `scripts/seed_demo_traffic.py` sends N requests (both in-distribution and an intentionally shifted batch, e.g. sales x2) so the report demonstrates drift detection.
- **Tests** (`test_drift.py`): identical distributions -> PSI near 0 and no flag; shifted distribution -> flagged; constant column handled; empty input handled.
- **Verify:** seed traffic, run `make drift`, report shows no drift for normal traffic and drift for shifted traffic.

**4.2 Test completeness**
- Ensure coverage of: data validation rules, feature functions on small fixtures (no leakage + parity), API with stub model, promotion gate refusing a worse model, drift functions, DAG import.
- `pytest --cov=src --cov-report=term-missing --cov-fail-under=80`.
- **Verify:** `make test` passes with coverage >= 80%.

**4.3 Latency measurement (`scripts/load_test.py`)**
- Uses httpx (async) to send e.g. 1,000 requests with concurrency 10 against a running API; reports p50, p95, p99 latency, throughput, error rate; writes `reports/latency.json`.
- **Verify:** `make load-test` produces `reports/latency.json` with real numbers (these feed the README and the resume bullet).

**4.4 CI (`.github/workflows/ci.yml`)**
- On push and pull request: set up Python 3.11 with pip cache, install deps, `ruff check`, `ruff format --check`, `pytest` with coverage (using the **synthetic** dataset and a file-based MLflow URI, no external services), build the Docker image (no push). Upload coverage XML as an artifact.
- **Verify:** workflow YAML is valid (`actionlint` if available); all steps pass locally via `make lint test` and `docker build`.

**4.5 README (follow the template below exactly)**
1. One-line description and problem
2. Architecture diagram (Mermaid from the spec: DVC raw data -> validation -> features -> training -> MLflow tracking/registry -> promote if better -> FastAPI; Airflow DAG triggers; API -> prediction logs -> monitoring/drift)
3. Tech stack table
4. Quick start: `git clone`, `cp .env.example .env`, `docker compose up`, URLs for MLflow (5000), Airflow (8080), API (8000)
5. API examples: real `curl` commands with real sample responses captured from the running service
6. Testing instructions and coverage badge
7. Results: model comparison table (real numbers from `reports/model_comparison.md`), latency numbers from `reports/latency.json`, screenshots (placeholders, see Human handoff), `curl` example
8. Design decisions: why each major tech was chosen and what changes at larger scale (feature store, Spark, scheduled batch scoring, Kubeflow/SageMaker equivalents)
9. Roadmap and known limitations (be honest: single-step horizon, synthetic data if used, no automated rollback beyond alias reset, etc.)
- Also record dataset source and license, and compute the MAPE improvement vs the moving-average baseline: `(MA_MAPE - best_MAPE) / MA_MAPE * 100`, and the median API latency, written to `docs/RESUME_NUMBERS.md` so the human can fill the resume bullets.

**4.6 Final cleanup**
- `make lint`, `make test`, fresh clone simulation: `git clone` to temp dir -> `cp .env.example .env` -> `docker compose up -d` -> health check passes.
- `docs/PROGRESS.md` complete; `docs/DECISIONS.md` lists every fallback and deviation.

Merge phase 4 to `main`, tag `v1.0.0`.

---

## 7. Definition of done (all must be true)

- [ ] `docker compose up` starts Postgres, MLflow, Airflow and the API with one command
- [ ] `dvc repro` rebuilds processed data, features, models and reports from raw data
- [ ] Comparison table of baselines vs linear vs HGB vs neural net on the same validation window is generated and committed
- [ ] DAG run registers a new model and the API serves it without code changes
- [ ] Promotion gate verified to refuse a worse model (unit test **and** end-to-end run)
- [ ] Prediction logs written to Postgres; drift report generated
- [ ] Tests pass with >= 80% coverage; CI workflow green config in place
- [ ] README complete with diagram and **real** numbers (no invented metrics)
- [ ] No secrets in Git; `.env.example` complete

---

## 8. Pitfalls checklist (re-check before each phase merge)

- [ ] No random splits; train max date < validation min date
- [ ] No feature uses same-day or future `sales` (rolling stats use `shift(1)`)
- [ ] Baselines exist and every model is compared against them
- [ ] Model path is never hard-coded; API loads from registry alias
- [ ] Train and serve use the same feature function
- [ ] Neural net early stopping does not peek at the comparison window
- [ ] Neural net claims are backed by the fair comparison only
- [ ] No logic left only in notebooks

---

## 9. Suggested commit sequence (guide)

```
chore: scaffold repository layout and tooling
feat: add data generator and download script
feat: add cleaning and validation with tests
build: add dvc pipeline for processed data
feat: add metrics and baseline models
feat: add feature engineering and time-based split with leakage tests
feat: add sklearn and pytorch models with mlflow tracking
feat: add model comparison report
build: add docker compose stack (postgres, mlflow, airflow, api)
feat: add promotion gate with tests
feat: add fastapi service with registry model loading
feat: add airflow retraining dag
feat: add prediction logging and drift monitoring
test: reach coverage target
ci: add github actions workflow
docs: write readme, decisions and results
```

---

## 10. Human handoff (the only things you should stop and ask for)

The agent cannot produce these; leave clearly marked placeholders in the README (e.g. `<!-- TODO(human): screenshot -->`) and list them in `docs/HANDOFF.md`:

1. Kaggle API credentials, if the real dataset should be used instead of synthetic data
2. Screenshot of the MLflow experiment comparison (`docs/images/mlflow_compare.png`)
3. Screenshot of a successful Airflow DAG run (`docs/images/airflow_dag.png`)
4. 2 to 3 minute demo video or GIF
5. Pushing to GitHub and enabling Actions, plus an optional deployment target a reviewer can open
6. Final resume bullets: fill `[X]%` and `[Y] ms` from `docs/RESUME_NUMBERS.md`

---

## 11. Interview notes to capture in `docs/DECISIONS.md` (short answers, in your own words from what was built)

- Why time-based validation matters
- How the promotion decision is made (the gate in section 4.6)
- What data drift is and how PSI/KS detect it
- What changes for much larger data (feature store, Spark, batch scoring)
- How this pipeline maps to Kubeflow or SageMaker
