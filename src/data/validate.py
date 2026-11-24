"""Schema and business-rule validation with readable failures."""

from __future__ import annotations

import pandas as pd

from .clean import BASE_COLUMNS, OPTIONAL_COLUMNS

KEY_COLUMNS = ["store_id", "item_id", "date"]


class DataValidationError(ValueError):
    """Raised when a dataframe violates the dataset contract."""


def _check(frame: pd.DataFrame, expect_complete: bool) -> None:
    errors: list[str] = []
    columns = set(frame.columns)
    missing = set(BASE_COLUMNS) - columns
    if missing:
        errors.append(f"missing required columns {sorted(missing)}")
    if errors:
        raise DataValidationError("; ".join(errors))

    if frame.empty:
        raise DataValidationError("dataframe is empty")

    for column in ["store_id", "item_id", "sales"]:
        if frame[column].isna().any():
            errors.append(f"column {column} contains nulls")
        if not pd.api.types.is_integer_dtype(frame[column]):
            errors.append(f"column {column} must be integer, got {frame[column].dtype}")
    if not pd.api.types.is_datetime64_any_dtype(frame["date"]):
        errors.append("column date must be datetime64")

    if (frame["sales"] < 0).any():
        errors.append("sales must be non-negative")

    if "price" in columns and frame["price"].notna().any() and (frame["price"] < 0).any():
        errors.append("price must be non-negative when present")
    if "promo" in columns and frame["promo"].notna().any():
        unexpected = set(frame["promo"].dropna().unique()) - {0, 1}
        if unexpected:
            errors.append(f"promo must be 0/1, found {sorted(unexpected)}")

    if expect_complete:
        duplicated = frame.duplicated(subset=KEY_COLUMNS).sum()
        if duplicated:
            errors.append(f"found {duplicated} duplicate (store_id, item_id, date) rows")
        ordered = frame.sort_values(KEY_COLUMNS)
        gaps = ordered.groupby(["store_id", "item_id"], sort=False)["date"].diff().dropna()
        bad_days = (gaps != pd.Timedelta(days=1)).sum()
        if bad_days:
            errors.append(f"found {bad_days} non-daily gaps in series dates")
        if not frame.sort_values(KEY_COLUMNS).index.equals(frame.index):
            errors.append("processed data must be sorted by store_id, item_id, date")

    if errors:
        raise DataValidationError("; ".join(errors))


def validate_raw(frame: pd.DataFrame) -> None:
    """Validate the raw contract; missing calendar days are allowed here."""
    _check(frame, expect_complete=False)


def validate_processed(frame: pd.DataFrame) -> None:
    """Validate the processed contract: complete, sorted daily series."""
    _check(frame, expect_complete=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the optional column names actually present in a dataframe."""
    return [column for column in OPTIONAL_COLUMNS if column in frame.columns]
