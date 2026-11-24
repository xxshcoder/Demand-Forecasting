"""One feature builder shared by batch training and single-request serving.

Every feature of day ``t`` uses only observations at ``t-1`` and earlier.
Calendar features for ``t`` are known at the end of ``t-1``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from functools import cache

import pandas as pd

from ..config import load_params, resolve_path

KEY_COLUMNS = ["store_id", "item_id", "date"]
BASE_MODEL_COLUMNS = [
    "store_id",
    "item_id",
    "dow",
    "month",
    "day_of_month",
    "is_weekend",
    "is_holiday",
]


@cache
def _holiday_calendar(country: str, years: tuple[int, ...]) -> frozenset[date]:
    try:
        import holidays  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - documented fallback
        return frozenset()
    calendar = holidays.country_holidays(country, years=years)
    return frozenset(calendar.keys())


def _is_holiday(dates: pd.Series, country: str) -> pd.Series:
    years = tuple(sorted(set(dates.dt.year.tolist())))
    if not years:
        return pd.Series(dtype="int64", index=dates.index)
    holiday_set = _holiday_calendar(country, years)
    return dates.dt.date.isin(holiday_set).astype("int64")


def _calendar_features(frame: pd.DataFrame, country: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["dow"] = frame["date"].dt.day_of_week.astype("int64")
    frame["month"] = frame["date"].dt.month.astype("int64")
    frame["day_of_month"] = frame["date"].dt.day.astype("int64")
    frame["is_weekend"] = (frame["dow"] >= 5).astype("int64")
    frame["is_holiday"] = _is_holiday(frame["date"], country)
    return frame


def _past_features(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add lags and rolling statistics, always derived from shifted sales."""
    frame = frame.copy()
    grouped = frame.groupby(["store_id", "item_id"], sort=False)["sales"]
    for lag in params["features"]["lags"]:
        frame[f"lag_{lag}"] = grouped.shift(int(lag)).astype("float64")
    for window in params["features"]["rolling_windows"]:
        window = int(window)
        shifted = grouped.transform(lambda s: s.shift(1))
        stats = shifted.groupby([frame["store_id"], frame["item_id"]], sort=False)
        frame[f"roll_mean_{window}"] = stats.transform(
            lambda s: s.rolling(window, min_periods=window).mean()
        )
        frame[f"roll_std_{window}"] = stats.transform(
            lambda s: s.rolling(window, min_periods=window).std().fillna(0.0)
        )
    return frame


def lag_and_rolling_columns(params: dict | None = None) -> list[str]:
    params = params or load_params()
    columns = [f"lag_{int(lag)}" for lag in params["features"]["lags"]]
    for window in params["features"]["rolling_windows"]:
        columns.extend([f"roll_mean_{int(window)}", f"roll_std_{int(window)}"])
    return columns


def build_features(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Build all features for a complete, sorted batch panel."""
    params = params or load_params()
    required = set(KEY_COLUMNS + ["sales"])
    if not required <= set(frame.columns):
        raise ValueError(
            f"Missing columns for feature building: {sorted(required - set(frame.columns))}"
        )
    enriched = _calendar_features(frame, params["data"]["country_holidays"])
    enriched = _past_features(enriched, params)
    return enriched


def model_feature_columns(
    frame: pd.DataFrame, params: dict | None = None, feature_list: Sequence[str] | None = None
) -> list[str]:
    """Return the exact model input columns, in a stable order."""
    params = params or load_params()
    columns = BASE_MODEL_COLUMNS + lag_and_rolling_columns(params)
    if feature_list is not None:
        return [column for column in columns if column in feature_list]
    if params["features"].get("use_price", True) and "price" in frame.columns:
        columns.append("price")
    if params["features"].get("use_promo", True) and "promo" in frame.columns:
        columns.append("promo")
    return columns


def build_single_row_features(
    recent_sales: Sequence[float],
    forecast_date: str | date,
    price: float | None,
    promo: int | None,
    store_id: int,
    item_id: int,
    params: dict | None = None,
    feature_list: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build features for one forecast request using the batch code path.

    ``recent_sales`` contains the last ``max(lags)`` daily values ending the
    day before ``forecast_date``, oldest first.
    """
    params = params or load_params()
    expected = max(int(lag) for lag in params["features"]["lags"])
    if len(recent_sales) != expected:
        raise ValueError(f"recent_sales must contain {expected} values, got {len(recent_sales)}")
    forecast_day = pd.Timestamp(forecast_date).normalize()
    history_days = [forecast_day - pd.Timedelta(days=i) for i in range(len(recent_sales), 0, -1)]
    history = pd.DataFrame(
        {
            "date": history_days,
            "store_id": store_id,
            "item_id": item_id,
            "sales": pd.Series(list(recent_sales), dtype="float64"),
        }
    )
    target = pd.DataFrame(
        {
            "date": [forecast_day],
            "store_id": [store_id],
            "item_id": [item_id],
            "sales": [float("nan")],  # the value we are forecasting
        }
    )
    if "price" in (feature_list or []) or (
        feature_list is None and params["features"].get("use_price", True)
    ):
        history["price"] = float("nan")
        target["price"] = price if price is not None else float("nan")
    if "promo" in (feature_list or []) or (
        feature_list is None and params["features"].get("use_promo", True)
    ):
        history["promo"] = 0
        target["promo"] = promo if promo is not None else 0
    combined = pd.concat([history, target], ignore_index=True)
    enriched = build_features(combined, params)
    columns = model_feature_columns(enriched, params, feature_list)
    return enriched.loc[len(enriched) - 1 :, columns].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build batch features from processed sales")
    parser.add_argument("--input", default=None)
    parser.add_argument("--out", default="data/processed/features.parquet")
    parser.add_argument("--reference-out", default="data/reference/train_features.parquet")
    args = parser.parse_args()

    params = load_params()
    source = resolve_path(args.input or params["data"]["processed_path"])
    frame = pd.read_parquet(source)
    enriched = build_features(frame, params)
    warmup_columns = lag_and_rolling_columns(params)
    # Drop warm-up rows where lags are not available yet.
    cleaned = enriched.dropna(subset=warmup_columns).reset_index(drop=True)
    cleaned.to_parquet(resolve_path(args.out), index=False)

    from .split import time_split  # local import keeps build importable on its own

    train, _, _ = time_split(
        cleaned, int(params["split"]["val_days"]), int(params["split"]["test_days"])
    )
    reference_path = resolve_path(args.reference_out)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(reference_path, index=False)
    print(f"Wrote {len(cleaned):,} feature rows and {len(train):,} reference rows")


if __name__ == "__main__":
    main()
