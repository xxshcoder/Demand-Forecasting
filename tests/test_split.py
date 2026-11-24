from __future__ import annotations

import pytest

from src.features.split import time_split


def test_time_split_orders_windows_and_has_no_overlap(tiny_processed):
    train, validation, test = time_split(tiny_processed, val_days=10, test_days=5)
    assert train["date"].max() < test["date"].min()
    assert test["date"].max() < validation["date"].min()
    assert set(train["date"]) & set(validation["date"]) == set()
    assert len(validation["date"].unique()) == 10
    assert len(test["date"].unique()) == 5


def test_time_split_without_test_window(tiny_processed):
    train, validation, test = time_split(tiny_processed, val_days=10, test_days=0)
    assert test.empty
    assert train["date"].max() < validation["date"].min()


def test_time_split_rejects_invalid_windows(tiny_processed):
    with pytest.raises(ValueError):
        time_split(tiny_processed, val_days=0)
    with pytest.raises(ValueError):
        time_split(tiny_processed.iloc[0:0], val_days=10)
