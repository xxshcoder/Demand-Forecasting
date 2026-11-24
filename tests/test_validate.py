from __future__ import annotations

import pandas as pd
import pytest

from src.data.validate import DataValidationError, validate_processed, validate_raw


def test_valid_raw_and_processed_pass(tiny_raw, tiny_processed):
    validate_raw(tiny_raw)
    validate_processed(tiny_processed)


def test_negative_sales_rejected(tiny_raw):
    bad = tiny_raw.copy()
    bad.loc[0, "sales"] = -1
    with pytest.raises(DataValidationError, match="non-negative"):
        validate_raw(bad)


def test_duplicate_keys_rejected(tiny_processed):
    bad = pd.concat([tiny_processed, tiny_processed.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_processed(bad)


def test_bad_schema_rejected(tiny_raw):
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_raw(tiny_raw.drop(columns=["sales"]))


def test_processed_gap_rejected(tiny_processed):
    bad = tiny_processed.drop(index=5).reset_index(drop=True)
    with pytest.raises(DataValidationError, match="non-daily gaps"):
        validate_processed(bad)


def test_promo_must_be_binary(tiny_processed):
    bad = tiny_processed.copy()
    bad["promo"] = 2
    with pytest.raises(DataValidationError, match="promo"):
        validate_raw(bad)
