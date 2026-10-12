# portfolio-optimiser
Mean-Variance (Markowitz efficient frontier) with real market data via yfinance, constrained optimisation via spicy and full test coverage

Design Decisions
-Missing tickets raise an error. If a ticket returns no data, then price_fetcher raises a ValueError  and drops the ticker from the portfolio while letting the user know
-Stocks are trimmed to a shared date window: If one stock starts trading a day later than the others, every row before it existed is dropped so all the same dates are used
-Adjusted prices: yfinance already adjust for splits and dividends
-Simple returns, annualised by 252. Daily returns are percentage changes, and mean and covariance are multiplied by 252 to give annual figures