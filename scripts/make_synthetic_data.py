"""Generate a deterministic, realistic stand-in for the Kaggle demand dataset.

Usage:
    python scripts/make_synthetic_data.py --out data/raw/train.csv --seed 42

The generator is intentionally kept outside ``src/``: it creates toy input
data, it is not reusable business logic. All learned pipeline code lives in
``src/``. The output schema is exactly ``date, store, item, sales``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(
    seed: int = 42,
    start: str = "2014-01-01",
    years: int = 5,
    stores: int = 10,
    items: int = 50,
    missing_rate: float = 0.005,
    outlier_rate: float = 0.0002,
) -> pd.DataFrame:
    """Return a seeded daily sales panel with seasonality, gaps and outliers."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=years * 365 + (years // 4), freq="D")
    n_days = len(dates)
    n_series = stores * items

    store_ids = np.repeat(np.arange(1, stores + 1), items)
    item_ids = np.tile(np.arange(1, items + 1), stores)

    # Per-series level, trend and amplitude; store/item effects are multiplicative.
    level = rng.lognormal(mean=np.log(12.0), sigma=0.35, size=n_series)
    store_factor = np.linspace(0.8, 1.3, stores).repeat(items)
    item_factor = np.tile(np.linspace(0.6, 1.5, items), stores)
    level = level * store_factor * item_factor
    trend = rng.uniform(-0.04, 0.07, size=n_series)

    t = np.arange(n_days, dtype=float)
    dow = dates.day_of_week.to_numpy()
    doy = dates.day_of_year.to_numpy()
    month = dates.month.to_numpy()

    # Weekly and yearly seasonality shared across series, with per-item phase.
    weekly = np.array([0.82, 0.90, 0.96, 1.02, 1.12, 1.28, 1.20])[dow]
    yearly = 1.0 + 0.14 * np.cos(2 * np.pi * (doy - 20) / 365.25)
    summer_dip = np.where((month >= 7) & (month <= 8), 0.94, 1.0)
    holiday = np.where((month == 11) & (dates.isocalendar().week.to_numpy() == 47), 1.35, 1.0)
    holiday = np.where((month == 12) & (dates.day.to_numpy() >= 20), 1.30, holiday)
    holiday = np.where((month == 1) & (dates.day.to_numpy() <= 2), 0.75, holiday)
    season = weekly * yearly * summer_dip * holiday

    # Promotions are a hidden effect: sparse, positive, different per series.
    promo = rng.random((n_series, n_days)) < 0.04
    promo_effect = np.where(promo, rng.uniform(1.08, 1.25, size=(n_series, n_days)), 1.0)

    trend_matrix = 1.0 + np.outer(trend, t / 365.25)
    phase = rng.uniform(0, 2 * np.pi, size=n_series)
    item_season = 1.0 + 0.06 * np.sin(2 * np.pi * doy / 365.25 + phase[:, None])
    mean = level[:, None] * trend_matrix * season[None, :] * item_season * promo_effect
    noise = rng.normal(1.0, 0.14, size=mean.shape)
    sales = np.maximum(np.rint(mean * noise), 0).astype(np.int64)

    # Inject rare outliers (e.g. one-off bulk purchases) and missing days.
    outlier_mask = rng.random(sales.shape) < outlier_rate
    sales[outlier_mask] = np.rint(sales[outlier_mask] * rng.uniform(3, 6, size=outlier_mask.sum()))
    missing_mask = rng.random(sales.shape) < missing_rate

    rows_missing, days_missing = np.nonzero(missing_mask)
    series_index = np.repeat(np.arange(n_series), n_days)
    day_index = np.tile(np.arange(n_days), n_series)
    keep = ~missing_mask.ravel()

    frame = pd.DataFrame(
        {
            "date": dates.to_numpy()[day_index[keep]],
            "store": store_ids[series_index[keep]],
            "item": item_ids[series_index[keep]],
            "sales": sales.ravel()[keep],
        }
    )
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    if len(rows_missing) == 0:  # pragma: no cover - only for degenerate seeds
        raise RuntimeError("Synthetic generator produced no rows")
    return frame.sort_values(["store", "item", "date"], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/train.csv", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--years", default=5, type=int)
    parser.add_argument("--stores", default=10, type=int)
    parser.add_argument("--items", default=50, type=int)
    parser.add_argument("--missing-rate", default=0.005, type=float)
    parser.add_argument("--outlier-rate", default=0.0002, type=float)
    args = parser.parse_args()

    frame = generate_dataset(
        seed=args.seed,
        start=args.start,
        years=args.years,
        stores=args.stores,
        items=args.items,
        missing_rate=args.missing_rate,
        outlier_rate=args.outlier_rate,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(
        f"Wrote {len(frame):,} rows to {args.out} ({frame['date'].min()} .. {frame['date'].max()})"
    )


if __name__ == "__main__":
    main()
