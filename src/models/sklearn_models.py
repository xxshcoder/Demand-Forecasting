"""scikit-learn model factories. Identifiers are handled per model type."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _numeric_and_id_columns(feature_columns: list[str]) -> tuple[list[str], list[str]]:
    ids = [c for c in ("store_id", "item_id") if c in feature_columns]
    numeric = [c for c in feature_columns if c not in ids]
    return numeric, ids


def build_linear_pipeline(feature_columns: list[str], params: dict) -> Pipeline:
    """Scaled numeric features + one-hot store/item identifiers."""
    numeric, ids = _numeric_and_id_columns(feature_columns)
    transformers = [("numeric", StandardScaler(), numeric)]
    if ids:
        transformers.append(("ids", OneHotEncoder(handle_unknown="ignore"), ids))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", Ridge(alpha=float(params["models"]["linear"]["alpha"]))),
        ]
    )


class _CategoricalIds:
    """Cast identifier columns to pandas categorical for HGB."""

    def __init__(self, ids: list[str]):
        self.ids = ids

    def fit(self, frame: pd.DataFrame, y=None):
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in self.ids:
            result[column] = result[column].astype("category")
        return result


def build_hgb_pipeline(feature_columns: list[str], params: dict) -> Pipeline:
    """Gradient boosting with native categorical treatment for the identifiers."""
    _, ids = _numeric_and_id_columns(feature_columns)
    cfg = params["models"]["hgb"]
    hgb = HistGradientBoostingRegressor(
        max_iter=int(cfg["max_iter"]),
        learning_rate=float(cfg["learning_rate"]),
        max_depth=(int(cfg["max_depth"]) if cfg.get("max_depth") is not None else None),
        random_state=int(cfg["random_state"]),
        categorical_features=ids or None,
    )
    return Pipeline([("categorical_ids", _CategoricalIds(ids)), ("model", hgb)])


def predict_array(model, features: pd.DataFrame) -> np.ndarray:
    """Uniform prediction adapter used by training and the promotion gate."""
    return np.clip(np.asarray(model.predict(features), dtype=float), 0, None)


def make_sklearn_pyfunc(model):
    """Wrap a fitted pipeline as pyfunc; skops rejects the HGB pipeline types."""
    import mlflow.pyfunc

    class SklearnPyfuncModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return predict_array(model, model_input)

    return SklearnPyfuncModel()
