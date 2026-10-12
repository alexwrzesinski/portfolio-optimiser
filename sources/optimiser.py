#A portfolio is a list of weights, one per stock, adding up to 1, optimiser goal is to find the best weights
#Portfolio variance equation variance = w^T x cov x w, w is weights, cov is annualised covariance matrix and variance indicates our risk, using a minimisation problem we can find the lowest=risk of stocks

import numpy as np
import scipy as sp

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

weights = np.array([0.5, 0.5])
avg = np.array([0.10, 0.20])
cov = np.array([[0.04, 0.00],
                [0.00, 0.09]])

print(portfolio_return(weights, avg))      # 0.5*0.10 + 0.5*0.20 = 0.15
print(portfolio_variance(weights, cov))    # 0.25*0.04 + 0.25*0.09 = 0.0325
print(portfolio_volatility(weights, cov))  # sqrt(0.0325) ≈ 0.1803