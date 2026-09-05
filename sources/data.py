"""data.py

Fetches price data from yfinance and converts them into coviariance 
and returns inputs needed for Markowitz optimiser"""

import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt

def price_fetcher(tickers, start, end):

    data= yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker"
    )

   
    prices = pd.DataFrame(
        {t: data[t]["Close"] for t in tickers
                }
    )
    return prices

def compute_returns(prices):
    if prices.empty:
        raise ValueError("Prices are empty cannot compute returns")
    returns = prices.pct_change()
    returns=returns.dropna(how="all") #First row will always be NAN after pct_change()
    return returns

def analyse_returns(returns):
    avg= returns.mean()*252
    cov= returns.cov()*252
    return avg, cov
