#A portfolio is a list of weights, one per stock, adding up to 1, optimiser goal is to find the best weights
#Portfolio variance equation variance = w^T x cov x w, w is weights, cov is annualised covariance matrix and variance indicates our risk, using a minimisation problem we can find the lowest=risk of stocks

import numpy as np
from scipy.optimize import minimize

#Takes weights and mean returns and returns expected annual return
def portfolio_return(weights,avg):
    annual_return=np.dot(weights,avg)
    return annual_return

#Takes weights and annulaised covaraince matric and computes portfolio variance equation to return the portflio variance
def portfolio_variance(weights,cov):
    port_var= weights @ cov @ weights
    return port_var


#Gets square root of portfolio variance 
def portfolio_volatility(weights,cov):
    return np.sqrt(portfolio_variance(weights,cov))

#Constraints must sum up to 1
def weights_sum_to_one(w): 
    return np.sum(w) - 1


def min_variance_portfolio(avg,cov):
    avg= np.asarray(avg) #Converting avg and cov from data.py into numpy arraus for easy calcualtions
    cov=np.asarray(cov)
    n=len(avg)      #Numbner of stocks

    start=np.full(n,1/n) #A starting guess of weights where it starts off with them being equal
    bounds = [(0,1)]*n  #No negative weights
    constraints = {"type": "eq", "fun": weights_sum_to_one}

    #The actual weights minimisation calculations using scipy, For what is the least risky risk?
    result = minimize(
        portfolio_variance,
        start,
        args=(cov,),
        method="SLSQP", #Sequential Least Squates Quadratic Programming, algorithm: a gradient-based method that solces a sequence of quadratic approximations, with supporiting bounds nad constraints
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise ValueError(f"Optimisaiton failed: {result.message}")

    return result.x #optimal weights

#need this for future max_sharpe calculation, this answers "What gives the most return for each unit of risk?"
def neg_sharpe(weights, avg, cov, risk_free_rate):
    sharpe = (portfolio_return(weights,avg)-risk_free_rate)/(portfolio_volatility(weights,cov))
    return -sharpe

#Max sharpe porfolio optimiser calculation
def max_sharpe_portfolio(avg,cov,risk_free_rate=0.04):
    avg= np.asarray(avg) #Converting avg and cov from data.py into numpy arraus for easy calcualtions
    cov=np.asarray(cov)
    n=len(avg)      #Numbner of stocks

    start=np.full(n,1/n) #A starting guess of weights where it starts off with them being equal
    bounds = [(0,1)]*n  #No negative weights
    constraints = {"type": "eq", "fun": weights_sum_to_one}

    #The actual weights minimisation calculations using scipy, For what is the least risky risk?
    result = minimize(
        neg_sharpe,
        start,
        args=(avg,cov,risk_free_rate),
        method="SLSQP", #Sequential Least Squates Quadratic Programming, algorithm: a gradient-based method that solces a sequence of quadratic approximations, with supporiting bounds nad constraints
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise ValueError(f"Optimisaiton failed: {result.message}")

    return result.x #optimal weights