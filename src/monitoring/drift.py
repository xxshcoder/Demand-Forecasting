"""Distribution drift detection: Population Stability Index and KS test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EPSILON = 1e-6


def _clean(values) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def psi(expected, actual, bins: int = 10) -> float:
    """PSI using quantile bins derived from the reference distribution."""
    reference, current = _clean(expected), _clean(actual)
    if reference.size == 0 or current.size == 0:
        return 0.0
    if np.all(reference == reference[0]):
        return 0.0 if np.all(current == reference[0]) else 1.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected_pct = np.histogram(reference, bins=edges)[0] / reference.size + EPSILON
    actual_pct = np.histogram(current, bins=edges)[0] / current.size + EPSILON
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def ks_test(expected, actual) -> tuple[float, float]:
    """Return the two-sample Kolmogorov-Smirnov statistic and p-value."""
    reference, current = _clean(expected), _clean(actual)
    if reference.size == 0 or current.size == 0:
        return 0.0, 1.0
    result = stats.ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


def check_drift(
    reference_df: pd.DataFrame, recent_df: pd.DataFrame, features: list[str], params: dict
) -> dict[str, dict[str, float | bool]]:
    """Return PSI, KS statistic/p-value and a drift flag for each feature."""
    cfg = params["monitoring"]
    results: dict[str, dict[str, float | bool]] = {}
    for column in features:
        if column not in reference_df.columns or column not in recent_df.columns:
            continue
        value = psi(reference_df[column], recent_df[column], int(cfg["psi_bins"]))
        statistic, pvalue = ks_test(reference_df[column], recent_df[column])
        results[column] = {
            "psi": value,
            "ks_statistic": statistic,
            "ks_pvalue": pvalue,
            "drift": bool(
                value > float(cfg["psi_threshold"]) or pvalue < float(cfg["ks_pvalue_threshold"])
            ),
        }
    return results
