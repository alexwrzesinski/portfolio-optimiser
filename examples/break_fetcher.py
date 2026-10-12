from sources.data import price_fetcher

#bad ticker, that doesnt exist
print("--- Test 1: bad ticker ---")
try:
    df = price_fetcher(["AAPL", "ZZZZINVALID"], "2023-01-01", "2023-02-01")
    print(df.head())
    print(df.isna().sum())
except Exception as e:
    print("ERROR:", type(e).__name__, e)

#ABNB only started training in Dec 2020 so this should return empties
print("--- Test 2: late IPO ---")
try:
    df = price_fetcher(["AAPL", "ABNB"], "2019-01-01", "2022-01-01")
    print(df.head())
    print(df.isna().sum())
except Exception as e:
    print("ERROR:", type(e).__name__, e)