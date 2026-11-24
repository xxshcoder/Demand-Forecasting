from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import build_features, build_single_row_features, model_feature_columns


def test_lag_and_rolling_values_are_hand_computed(tiny_processed, params):
    features = build_features(tiny_processed, params)
    row = features.loc[30]
    previous = tiny_processed.loc[10:29, "sales"].to_numpy()
    assert row["lag_1"] == tiny_processed.loc[29, "sales"]
    assert row["lag_7"] == tiny_processed.loc[23, "sales"]
    assert row["roll_mean_7"] == pytest.approx(previous[-7:].mean())
    assert row["roll_std_7"] == pytest.approx(pd.Series(previous[-7:]).std())
    assert row["dow"] == pd.Timestamp(row["date"]).day_of_week
    assert row["is_weekend"] == int(row["dow"] >= 5)


def test_no_same_day_leakage(tiny_processed, params):
    before = build_features(tiny_processed, params)
    changed = tiny_processed.copy()
    changed.loc[30, "sales"] = 9999
    after = build_features(changed, params)
    feature_columns = [
        c for c in before.columns if c not in {"sales", "date", "store_id", "item_id"}
    ]
    pd.testing.assert_series_equal(before.loc[30, feature_columns], after.loc[30, feature_columns])
    # The next day's lag_1 must see the change (proving the test is meaningful).
    assert after.loc[31, "lag_1"] != before.loc[31, "lag_1"]


def test_train_serve_parity(tiny_processed, params):
    features = build_features(tiny_processed, params)
    columns = model_feature_columns(features, params)
    target_position = 40
    target_date = features.loc[target_position, "date"]
    recent = features.loc[target_position - 28 : target_position - 1, "sales"].tolist()

    single = build_single_row_features(
        recent_sales=recent,
        forecast_date=target_date,
        price=None,
        promo=None,
        store_id=1,
        item_id=1,
        params=params,
        feature_list=columns,
    )
    batch_row = features.loc[[target_position], columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(single, batch_row, check_dtype=False, atol=1e-9)


def test_single_row_requires_full_history(params):
    try:
        build_single_row_features([1, 2, 3], "2024-01-01", None, None, 1, 1, params)
    except ValueError as error:
        assert "28 values" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected a ValueError for short history")


def test_rolling_std_is_finite_on_constant_series(params):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40),
            "store_id": 1,
            "item_id": 1,
            "sales": np.ones(40, dtype="int64"),
        }
    )
    features = build_features(frame, params)
    assert np.isfinite(features["roll_std_7"]).all()
