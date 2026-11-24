from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import LastWeekBaseline, MovingAverageBaseline
from src.models.metrics import all_metrics, mae, masked_mape, rmse, wape


def test_metrics_values_and_zero_handling():
    y = np.array([0.0, 10.0, 20.0])
    pred = np.array([2.0, 8.0, 25.0])
    assert mae(y, pred) == pytest.approx(3.0)
    assert rmse(y, pred) == pytest.approx(np.sqrt((4 + 4 + 25) / 3))
    # The zero actual is skipped: |8-10|/10 and |25-20|/20.
    assert masked_mape(y, pred) == pytest.approx((0.2 + 0.25) / 2 * 100)
    # WAPE includes the zero actual in the denominator.
    assert wape(y, pred) == pytest.approx(9 / 30 * 100)
    assert np.isnan(masked_mape(np.zeros(3), np.ones(3)))


def test_all_metrics_returns_expected_keys():
    result = all_metrics([1, 2, 3], [1, 3, 3])
    assert set(result) == {"mae", "rmse", "mape", "wape"}
    assert all(np.isfinite(value) for value in result.values())


def test_baselines_use_the_configured_columns(params):
    features = pd.DataFrame({"lag_7": [3.0, 4.0], "roll_mean_7": [5.0, 6.0]})
    assert LastWeekBaseline(params=params).predict(features).tolist() == [3.0, 4.0]
    assert MovingAverageBaseline(params=params).predict(features).tolist() == [5.0, 6.0]
    assert LastWeekBaseline(lag=28).column == "lag_28"


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        mae([1, 2], [1])
