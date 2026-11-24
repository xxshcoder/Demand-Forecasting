# Human handoff

The following cannot be produced by this agent and remain as placeholders.

1. **Kaggle credentials** — place `kaggle.json` correctly and run `scripts/download_data.sh`
   if the real dataset should replace synthetic data. Set `DATASET_LABEL=kaggle` and retrain.
2. **MLflow comparison screenshot** — `docs/images/mlflow_compare.png`
   `<!-- TODO(human): screenshot of the MLflow experiment comparison -->`
3. **Airflow DAG screenshot** — `docs/images/airflow_dag.png`
   `<!-- TODO(human): screenshot of a successful DAG run -->`
4. **Demo video or GIF** — 2 to 3 minutes showing compose up, DAG trigger and API prediction.
5. **Docker** — run `docker compose up -d`, verify the DAG/registry/API acceptance steps, and note
   the built API image size in README.
6. **GitHub** — push the repository, enable Actions, optionally provide a deployment URL.
7. **Resume bullets** — the measured values are already filled in `docs/RESUME_NUMBERS.md`
   from `reports/model_comparison.md` and `reports/latency.json`; copy the prepared bullet into
   the resume and replace the local latency with a Docker/Compose measurement if desired.
