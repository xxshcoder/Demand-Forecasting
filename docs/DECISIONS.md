# Decisions, fallbacks and deviations

Every fallback or deviation from `plan.md` is recorded here, with the reason.

## Data source

- **Fallback used:** Kaggle CLI/credentials were not configured in the build environment, so
  `scripts/make_synthetic_data.py --seed 42` generated the dataset. It has the same schema and
  shape (10 stores x 50 items x ~5 years, 908,442 rows after injected gaps) and includes trend,
  weekly/yearly seasonality, holiday bumps, promo effects, noise, missing days and outliers.
  All reported numbers in this build therefore describe **synthetic data**, not Kaggle data.
- The real dataset can be substituted at any time with `scripts/download_data.sh`; no code
  changes are needed. `docs/HANDOFF.md` records the credentials handoff.

## Environment

- The build workstation has Python 3.15.2 and no Docker, DVC CLI or `make` preinstalled. The
  project still targets Python 3.11 in `pyproject.toml` and in the Docker images. Docker-specific
  acceptance steps are documented but cannot be executed on this machine. DVC is installed as a
  Python package so `dvc` is available in the environment. Where `make` is unavailable, the exact
  underlying commands from the `Makefile` are run directly.
- `psycopg2-binary==2.9.11` has no Python 3.15 wheel and a source build needs `pg_config`, which
  is absent locally. Local verification therefore uses SQLite (`DATABASE_URL=sqlite:///…`); the
  Docker images and CI use Python 3.11, where the pinned wheel installs normally.
- The zero-restart alias swap was verified by changing the registry `production` alias from v1 to v2 and back while the API kept running. This tests the alias-polling mechanism only; v2 (linear) was **refused** by the promotion gate, so it was never approved as production by the pipeline.
- `scripts/seed_demo_traffic.py` samples real processed histories across all store/item series for the in-distribution batch, then doubles only `recent_sales` for the shifted batch. The first version used a single fixed series, which flagged identifier drift even for normal traffic.
- All model artifacts are logged as `mlflow.pyfunc` models rather than `mlflow.sklearn` models.
  MLflow 3 validates sklearn saves with skops, which rejects the custom baseline class, HGB's
  `TreePredictor` and the categorical-id transformer. `pyfunc` keeps one loading path for every
  model in the API and promotion gate, with an MLflow signature and input example on each run.

## DVC pipeline

- `reports/baseline_metrics.json` is declared as an output of the `evaluate` stage, not `train`:
  `src/models/evaluate.py` derives baseline lag features from processed sales and scores both
  baselines there. Declaring it under `train` made the first `dvc repro` fail after all models
  had been trained; the stage definition was corrected and the existing outputs were recorded.
- DVC adds generated report outputs to `reports/.gitignore`. That file is committed with an
  explicit comment instead of the generated patterns so the small, reviewable artifacts
  (comparison table, metrics JSON, drift/latency reports) are versioned; only the large
  per-request prediction CSVs remain ignored by the root `.gitignore`.
- Interrupting a retrain left `reports/train_summary.json` missing while DVC's cache still held
  the successful outputs. `dvc checkout` restored them and `dvc commit -f` recorded the final
  code state, after which `dvc status` was clean and `dvc repro` was a no-op.

## params.yaml additions

- `data.missing_sales_strategy: forward_fill` — the plan requires the missing-day policy to be
  configurable but the sample params block omits the key. Forward fill (then back fill, then zero)
  preserves local demand level; `zero` is also implemented and tested.

## Validation approach

- Schema and business rules use explicit pandas checks with a `DataValidationError`. This is the
  documented "pydantic-based checks" fallback if pandera is unsuitable; it keeps validation
  dependency-light and gives readable aggregated errors. Pandera is pinned as an optional
  dependency for future schema migration.
- Duplicate `(store_id, item_id, date)` keys are checked on the **processed** panel, not on raw
  input: raw data may legitimately contain duplicates that `clean_sales` removes. Raw validation
  still enforces schema, dtypes, non-null keys and non-negative sales.
- The local MLflow tracking URI defaults to `sqlite:///mlflow.db` rather than `file:./mlruns`.
  The plan's file store cannot serve registry aliases, which the API needs for zero-downtime
  model swaps; Compose uses a Postgres-backed MLflow server instead.

## Extra configuration keys

- `models.nn.embedding_dim`, `models.nn.early_stop_days`, `models.nn.early_stop_patience` —
  the sample params block did not include the neural net's embedding size or early-stopping
  controls. They must be configurable rather than hard-coded; defaults are 8, 14 and 3.
- `monitoring.psi_bins` — likewise needed for the PSI bin count (default 10).
- `DATASET_LABEL` — an environment variable (not a hyperparameter) tags runs with their
  provenance so synthetic and Kaggle results can never be confused.
- `mlflow.promotion_experiment_name` — promotion decisions are logged in a separate experiment
  so the training experiment contains exactly one run per model (the plan's five-run check).

## Interview notes

- **Why time-based validation?** Sales are autocorrelated; a random split would let the model
  learn from future observations and would overstate real one-day-ahead performance. We train
  on earlier dates, keep a 90-day test window and score every model on the identical final
  90-day validation window.
- **How is the promotion decision made?** `should_promote` is a pure function. It refuses
  NaN/inf metrics, requires the candidate to beat the best baseline on MAE, and—when a
  production model exists—requires at least a 0.5% relative MAE improvement while limiting
  RMSE regression to 5%. The current production model is re-scored on the same latest
  validation window before the gate runs.
- **What is data drift and how do PSI/KS detect it?** Drift is a change in the input feature
  distribution between training and live traffic. PSI compares histogram proportions across
  quantile bins defined on the reference data; KS compares the full empirical distributions
  and returns a statistic and p-value. Either crossing the configured threshold raises a flag.
- **What changes at much larger data?** Move the processed panel/features to a feature store,
  replace pandas group-bys with Spark/SQL transforms, compute batch predictions on a schedule
  instead of per request, and use partitioned parquet plus distributed training.
- **How does this map to Kubeflow/SageMaker?** DVC stages map to pipeline components, the
  Airflow DAG to a Kubeflow Pipeline or SageMaker Pipeline, MLflow runs/registry to SageMaker
  Experiments/Model Registry, the FastAPI service to a SageMaker endpoint, and the promotion
  gate to a pipeline conditional step or CI approval.

## EDA findings (from `notebooks/01_eda.ipynb`)

- The panel is complete after cleaning: one row per store/item/day.
- Weekly seasonality is strong: weekends differ from weekdays.
- Yearly seasonality and holidays are visible in monthly means.
- Store and item levels differ substantially, so identifiers are useful features.
- A handful of extreme values were winsorised during cleaning.
