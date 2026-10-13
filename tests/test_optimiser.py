#This script is used to test the optimiser with hand written weights and annualised return values

import numpy as np
from sources.optimiser import portfolio_return, portfolio_variance, portfolio_volatility, min_variance_portfolio, neg_sharpe, max_sharpe_portfolio

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

def test_neg_sharpe_known_answer():
    # 50/50 mix: return 0.15, volatility ~0.1803, Sharpe ~0.610, so neg_sharpe ~ -0.610
    w = np.array([0.5, 0.5])
    assert abs(neg_sharpe(w, avg, cov, 0.04) - (-0.610)) < 1e-3


def test_max_sharpe_known_answer():
    # Uncorrelated stocks: weights proportional to (return - rf) / variance
    w = max_sharpe_portfolio(avg, cov, 0.04)
    assert abs(w[0] - 0.4576) < 1e-3
    assert abs(w[1] - 0.5424) < 1e-3


def test_max_sharpe_weights_valid():
    # Weights sum to 1 and none are negative (long-only)
    w = max_sharpe_portfolio(avg, cov, 0.04)
    assert abs(np.sum(w) - 1) < 1e-6
    assert np.all(w >= -1e-8)


def test_max_sharpe_beats_equal_weights():
    # Optimiser's neg_sharpe must be at least as low (as good) as a naive 50/50 split
    w = max_sharpe_portfolio(avg, cov, 0.04)
    equal = np.array([0.5, 0.5])
    assert neg_sharpe(w, avg, cov, 0.04) <= neg_sharpe(equal, avg, cov, 0.04) + 1e-9