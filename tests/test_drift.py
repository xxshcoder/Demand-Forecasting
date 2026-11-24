from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring.drift import check_drift, ks_test, psi


def test_identical_distributions_have_no_drift(params):
    reference = pd.DataFrame({"x": np.arange(1000) % 17})
    recent = reference.copy()
    result = check_drift(reference, recent, ["x"], params)
    assert result["x"]["psi"] == 0.0 or result["x"]["psi"] < 1e-6
    assert result["x"]["ks_pvalue"] > 0.01
    assert not result["x"]["drift"]


def test_shifted_distribution_is_flagged(params):
    reference = pd.DataFrame({"x": np.arange(1000) % 17})
    recent = pd.DataFrame({"x": np.arange(1000) % 17 + 50})
    result = check_drift(reference, recent, ["x"], params)
    assert result["x"]["psi"] > params["monitoring"]["psi_threshold"]
    assert result["x"]["drift"]


def test_constant_and_empty_inputs(params):
    assert psi([5] * 100, [5] * 100) == 0.0
    assert psi([5] * 100, [8] * 100) == 1.0
    assert psi([], []) == 0.0
    assert ks_test([], []) == (0.0, 1.0)


def test_ks_detects_shift():
    statistic, pvalue = ks_test(np.arange(100), np.arange(100) + 1000)
    assert statistic > 0.9
    assert pvalue < 0.01
