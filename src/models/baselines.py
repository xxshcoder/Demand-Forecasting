"""Simple forecasting baselines that share the model ``predict`` interface."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_params


class LastWeekBaseline:
    """Predict the sales value observed seven days earlier."""

    def __init__(self, lag: int | None = None, params: dict | None = None):
        params = params or load_params()
        self.lag = int(lag if lag is not None else params["baselines"]["last_week_lag"])
        self.name = "last_week"
        self.column = f"lag_{self.lag}"

    def fit(self, features: pd.DataFrame, y: pd.Series) -> LastWeekBaseline:
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.clip(features[self.column].to_numpy(dtype=float), 0, None)


def make_baseline_pyfunc(model):
    """Wrap a baseline as an MLflow pyfunc model (skops rejects custom classes)."""
    import mlflow.pyfunc

    class BaselinePyfuncModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model.predict(model_input)

    return BaselinePyfuncModel()


class MovingAverageBaseline:
    """Predict the rolling mean of the observed past sales."""

    def __init__(self, window: int | None = None, params: dict | None = None):
        params = params or load_params()
        self.window = int(
            window if window is not None else params["baselines"]["moving_average_window"]
        )
        self.name = "moving_average"
        self.column = f"roll_mean_{self.window}"

    def fit(self, features: pd.DataFrame, y: pd.Series) -> MovingAverageBaseline:
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.clip(features[self.column].to_numpy(dtype=float), 0, None)
