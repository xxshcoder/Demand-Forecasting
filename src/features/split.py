"""Strictly time-based train / test / validation splits."""

from __future__ import annotations

import pandas as pd


def time_split(
    frame: pd.DataFrame, val_days: int, test_days: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by date: train < test < validation, with no overlap.

    The validation window is the final ``val_days`` calendar days; the optional
    test window is the ``test_days`` immediately before it.
    """
    if val_days <= 0 or test_days < 0:
        raise ValueError("val_days must be positive and test_days non-negative")
    if frame.empty:
        raise ValueError("Cannot split an empty frame")

    dates = pd.to_datetime(frame["date"])
    last_day = dates.max().normalize()
    val_start = last_day - pd.Timedelta(days=val_days - 1)
    test_start = val_start - pd.Timedelta(days=test_days)

    train = frame[dates < test_start].copy()
    test = (
        frame[(dates >= test_start) & (dates < val_start)].copy()
        if test_days
        else frame.iloc[0:0].copy()
    )
    validation = frame[dates >= val_start].copy()

    if train.empty or validation.empty:
        raise ValueError("Time split produced an empty train or validation window")
    train_max = pd.to_datetime(train["date"]).max()
    val_min = pd.to_datetime(validation["date"]).min()
    if not test.empty:
        assert train_max < pd.to_datetime(test["date"]).min()
        assert pd.to_datetime(test["date"]).max() < val_min
    assert train_max < val_min
    return train, validation, test
