"""Deterministic cleaning of raw sales panels (unit-tested, notebook-free)."""

from __future__ import annotations

import pandas as pd

from ..config import load_params

BASE_COLUMNS = ["date", "store_id", "item_id", "sales"]
OPTIONAL_COLUMNS = ["price", "promo"]


def _fill_sales(series: pd.Series, strategy: str) -> pd.Series:
    if strategy == "forward_fill":
        filled = series.ffill().bfill()
        return filled.fillna(0)
    if strategy == "zero":
        return series.fillna(0)
    raise ValueError(f"Unknown data.missing_sales_strategy: {strategy!r}")


def clean_sales(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Clean a raw panel into one row per (store, item, calendar day).

    Steps: drop exact duplicate keys, re-insert missing calendar days, fill
    missing sales using the configured strategy, winsorise outliers per
    series and clip sales at zero. The output is sorted by series and date.
    """
    params = params or load_params()
    cfg = params["data"]
    frame = frame.copy()

    required = set(BASE_COLUMNS)
    if not required <= set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")

    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.drop_duplicates(subset=["store_id", "item_id", "date"], keep="first")
    frame = frame.sort_values(["store_id", "item_id", "date"])

    if cfg.get("fill_missing_days", True):
        all_dates = pd.date_range(frame["date"].min(), frame["date"].max(), freq="D")
        index = pd.MultiIndex.from_product(
            [sorted(frame["store_id"].unique()), sorted(frame["item_id"].unique()), all_dates],
            names=["store_id", "item_id", "date"],
        )
        frame = frame.set_index(["store_id", "item_id", "date"]).reindex(index).reset_index()
    else:
        frame = frame.reset_index(drop=True)

    # A missing day has no observed sales. Forward filling keeps the local
    # level of the series; backfill covers gaps at the very start, and zero
    # covers series with no observations at all.
    frame["sales"] = frame.groupby(["store_id", "item_id"], sort=False)["sales"].transform(
        lambda s: _fill_sales(s, cfg.get("missing_sales_strategy", "forward_fill"))
    )
    if "price" in frame.columns:
        frame["price"] = frame.groupby(["store_id", "item_id"], sort=False)["price"].ffill()
    if "promo" in frame.columns:
        frame["promo"] = frame.groupby(["store_id", "item_id"], sort=False)["promo"].transform(
            lambda s: s.fillna(0)
        )

    quantile = float(cfg["outlier_clip_quantile"])
    if 0 < quantile < 1:
        cap = frame.groupby(["store_id", "item_id"], sort=False)["sales"].transform(
            lambda s: s.quantile(quantile)
        )
        frame["sales"] = frame["sales"].clip(upper=cap)
    frame["sales"] = frame["sales"].clip(lower=0).round().astype("int64")
    if "promo" in frame.columns:
        frame["promo"] = frame["promo"].clip(0, 1).astype("int64")

    return frame.sort_values(["store_id", "item_id", "date"], ignore_index=True)
