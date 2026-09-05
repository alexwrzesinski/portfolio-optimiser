"""
test_data.py

Tests for data.py — uses small, hand-built DataFrames instead of real
yfinance calls, so these run fast and don't depend on network access.
"""

import pandas as pd
import numpy as np
import pytest
from sources.data import compute_returns, analyse_returns


# ---------- compute_returns ----------

def test_compute_returns_basic():
    """Check returns are calculated correctly against hand-computed values."""
    fake_prices = pd.DataFrame({
        "AAPL": [100, 102, 101],
    })

    result = compute_returns(fake_prices)

    expected_day2 = 0.02          # (102 - 100) / 100
    expected_day3 = (101 - 102) / 102

    assert abs(result["AAPL"].iloc[0] - expected_day2) < 1e-6
    assert abs(result["AAPL"].iloc[1] - expected_day3) < 1e-6


def test_compute_returns_drops_first_row():
    """First row (NaN from pct_change) should be dropped, not present."""
    fake_prices = pd.DataFrame({
        "AAPL": [100, 102, 101],
    })

    result = compute_returns(fake_prices)

    # Original had 3 rows; after dropping the first NaN row, expect 2.
    assert len(result) == 2
    assert not result.isna().any().any()  # no NaNs anywhere in the result


def test_compute_returns_multiple_tickers():
    """Each column's returns should be computed independently."""
    fake_prices = pd.DataFrame({
        "AAPL": [100, 102, 101],
        "MSFT": [200, 198, 202],
    })

    result = compute_returns(fake_prices)

    expected_msft_day2 = (198 - 200) / 200
    assert abs(result["MSFT"].iloc[0] - expected_msft_day2) < 1e-6


# ---------- analyse (annualize_stats) ----------

def test_analyse_shapes():
    """Mean should be one value per ticker; cov should be square (tickers x tickers)."""
    fake_returns = pd.DataFrame({
        "AAPL": [0.01, -0.02, 0.015, 0.005],
        "MSFT": [0.02, -0.01, 0.010, 0.000],
    })

    avg, cov = analyse_returns(fake_returns)

    assert len(avg) == 2                     # one mean per ticker
    assert cov.shape == (2, 2)                # square matrix, tickers x tickers


def test_analyse_cov_symmetric():
    """Covariance matrix must be symmetric: cov[A][B] == cov[B][A]."""
    fake_returns = pd.DataFrame({
        "AAPL": [0.01, -0.02, 0.015, 0.005],
        "MSFT": [0.02, -0.01, 0.010, 0.000],
    })

    _, cov = analyse_returns(fake_returns)

    assert abs(cov.loc["AAPL", "MSFT"] - cov.loc["MSFT", "AAPL"]) < 1e-10


def test_analyse_annualization_scaling():
    """Mean should be daily mean * 252, not just the raw daily mean."""
    fake_returns = pd.DataFrame({
        "AAPL": [0.01, -0.02, 0.015, 0.005],
    })

    avg, _ = analyse_returns(fake_returns)

    expected = fake_returns["AAPL"].mean() * 252
    assert abs(avg["AAPL"] - expected) < 1e-10


# ---------- edge cases ----------

def test_compute_returns_empty_raises():
    """Empty prices DataFrame should raise, not silently return garbage."""
    empty_prices = pd.DataFrame()

    with pytest.raises(ValueError):
        compute_returns(empty_prices)