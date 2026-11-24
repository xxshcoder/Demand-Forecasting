from __future__ import annotations

import copy

import pandas as pd

from src.data.clean import clean_sales
from src.data.load import load_raw
from src.data.prepare import prepare


def test_clean_removes_duplicates_fills_gaps_and_clips(tiny_raw, params):
    cfg = copy.deepcopy(params)
    cfg["data"]["outlier_clip_quantile"] = 0.75
    cleaned = clean_sales(tiny_raw, cfg)

    assert len(cleaned) == 4
    assert not cleaned[["store_id", "item_id", "date"]].duplicated().any()
    assert cleaned["date"].tolist() == pd.date_range("2024-01-01", periods=4).tolist()
    # 2024-01-02 was absent and is forward-filled from 2024-01-01.
    assert cleaned.loc[cleaned["date"] == "2024-01-02", "sales"].iloc[0] == 10
    assert cleaned["sales"].max() < 1000


def test_clean_zero_strategy_and_no_gap_fill(tiny_raw, params):
    cfg = copy.deepcopy(params)
    cfg["data"]["fill_missing_days"] = False
    cfg["data"]["missing_sales_strategy"] = "zero"
    cfg["data"]["outlier_clip_quantile"] = 1.0
    cleaned = clean_sales(tiny_raw, cfg)
    assert len(cleaned) == 3  # duplicate removed, gap not inserted
    assert cleaned["sales"].max() == 1000


def test_clean_is_sorted_by_series_and_date(tiny_raw, params):
    cleaned = clean_sales(tiny_raw, params)
    expected = cleaned.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(cleaned, expected)


def test_load_raw_renames_canonical_columns(fixtures_dir):
    raw = load_raw(fixtures_dir / "tiny_raw.csv")
    assert {"date", "store_id", "item_id", "sales"} <= set(raw.columns)
    assert pd.api.types.is_datetime64_any_dtype(raw["date"])


def test_prepare_writes_parquet(fixtures_dir, params, tmp_path):
    output = prepare(fixtures_dir / "tiny_raw.csv", tmp_path / "sales.parquet", params)
    assert len(output) == 4
    assert (tmp_path / "sales.parquet").exists()
