# portfolio-optimiser
Mean-Variance (Markowitz efficient frontier) with real market data via yfinance, constrained optimisation via spicy and full test coverage

Design Decisions
-Missing tickets raise an error. If a ticket returns no data, then price_fetcher raises a ValueError  and drops the ticker from the portfolio while letting the user know
-Stocks are trimmed to a shared date window: If one stock starts trading a day later than the others, every row before it existed is dropped so all the same dates are used
-Adjusted prices: yfinance already adjust for splits and dividends
-Simple returns, annualised by 252. Daily returns are percentage changes, and mean and covariance are multiplied by 252 to give annual figures

Minimum Variance Portfolio:

Ran optimiser on 3 differnet set of stocks, using daily data from 2023-01-01, to 2025-01-01
Resulted: Set1 [AAPL, MSFT, JPM, XOM]-> Weights 25% AAPL, 23% MSFT, 27% JPM, 26% XOM as very similar volatility across stocks so no reason to favour any more than otehrs
          Set2 [KO, JNJ, TSLA, NVDA]->About 60% KO, 32% JNJ, 2% TSLA, 6% NVDA
                                    - Ko and JNJ are low volatilyy so optimiser favoured them much more, but sitll get some of the slice becuase diversifaication lowers overall risk

Takeaway:
- Minimum variance only looks at risk, not returns. NVDA had a very strong 2023, but this portfolio ignores that.
- Results depend on the stocks chosen and the date range, so these weights would change with different data.