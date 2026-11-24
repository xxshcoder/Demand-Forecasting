from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

DAG_PATH = Path(__file__).resolve().parents[1] / "dags" / "retrain_dag.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("retrain_dag", DAG_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dag_import_and_task_order():
    module = _load_module()
    assert module.DAG_ID == "demand_forecast_retrain"
    assert module.TASK_ORDER == [
        "ingest",
        "validate",
        "build_features",
        "train",
        "evaluate",
        "compare",
        "register_and_promote",
    ]
    if module.AIRFLOW_AVAILABLE:
        assert set(module.demand_forecast_retrain_dag.task_ids) == set(module.TASK_ORDER)
    else:
        # Airflow is not installed in the local test environment; verify the
        # decorated task order statically. The Airflow image imports the DAG
        # for real and the branch above runs there.
        tree = ast.parse(DAG_PATH.read_text(encoding="utf-8"))
        task_ids = [
            keyword.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Call)
            for keyword in decorator.keywords
            if keyword.arg == "task_id" and isinstance(keyword.value, ast.Constant)
        ]
        assert task_ids == module.TASK_ORDER
