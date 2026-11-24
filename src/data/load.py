"""Read the raw CSV and apply the canonical schema contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_COLUMNS = ["date", "store_id", "item_id", "sales"]
OPTIONAL_COLUMNS = ["price", "promo"]
RAW_RENAMES = {"store": "store_id", "item": "item_id"}


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load a raw CSV, renaming ``store``/``item`` to the canonical names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}; run scripts/make_synthetic_data.py")
    frame = pd.read_csv(path)
    frame = frame.rename(columns=RAW_RENAMES)
    missing = set(BASE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Raw data is missing required columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    for column in ["store_id", "item_id", "sales"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    for column in OPTIONAL_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    columns = BASE_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in frame.columns]
    return frame.loc[:, columns]
