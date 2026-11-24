"""Forecast metrics with explicit, tested zero handling."""

from __future__ import annotations

import numpy as np


def _arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {predicted.shape}")
    if actual.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays")
    return actual, predicted


def mae(y_true, y_pred) -> float:
    actual, predicted = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true, y_pred) -> float:
    actual, predicted = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def masked_mape(y_true, y_pred) -> float:
    """MAPE over rows with positive actuals only; NaN if none are positive."""
    actual, predicted = _arrays(y_true, y_pred)
    mask = actual > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error, well-defined with zero actuals."""
    actual, predicted = _arrays(y_true, y_pred)
    denominator = np.abs(actual).sum()
    if denominator == 0:
        return float("nan")
    return float(np.abs(actual - predicted).sum() / denominator * 100)


def all_metrics(y_true, y_pred) -> dict[str, float]:
    """Return the four metrics reported for every model on the same rows."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": masked_mape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }
