from __future__ import annotations

import copy

import numpy as np

from src.features.build import build_features, model_feature_columns
from src.models.nn import TorchForecaster


def test_tiny_data_nn_smoke_train_and_predict(tiny_processed, params):
    cfg = copy.deepcopy(params)
    cfg["models"]["nn"]["epochs"] = 1
    cfg["models"]["nn"]["early_stop_patience"] = 10
    frame = build_features(tiny_processed, cfg).dropna().reset_index(drop=True)
    columns = model_feature_columns(frame, cfg)
    model = TorchForecaster(cfg, columns)
    model.fit(frame, frame["sales"])
    predictions = model.predict(frame)
    assert predictions.shape == (len(frame),)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()
