#This script is used to test the optimiser with hand written weights and annualised return values

import numpy as np
from sources.optimiser import portfolio_return, portfolio_variance, portfolio_volatility

weights = np.array([0.5, 0.5])
avg = np.array([0.10, 0.20])
cov = np.array([[0.04, 0.00],
                [0.00, 0.09]])


def test_portfolio_return():
    assert abs(portfolio_return(weights, avg) - 0.15) < 1e-6


def test_portfolio_variance():
    assert abs(portfolio_variance(weights, cov) - 0.0325) < 1e-6


def test_portfolio_volatility():
    assert abs(portfolio_volatility(weights, cov) - np.sqrt(0.0325)) < 1e-6

def test_all_in_one_stock_matches_that_stock():
    # 100% in stock 1: return and variance should equal stock 1's own numbers
    w = np.array([1.0, 0.0])
    assert abs(portfolio_return(w, avg) - 0.10) < 1e-6
    assert abs(portfolio_variance(w, cov) - 0.04) < 1e-6


def test_diversification_lowers_risk():
    # two uncorrelated stocks mixed 50/50 should be less risky than the riskier one alone
    assert portfolio_volatility(weights, cov) < np.sqrt(0.09)