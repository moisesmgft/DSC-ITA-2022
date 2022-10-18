import pandas_datareader as pdr
from datetime import datetime
import json

start_date = datetime(1970, 1, 1)
end_date = datetime.now().date()

with open('data/tickers/sp500.json', 'r') as f:
    sp500_tickers = json.load(f)

with open('data/tickers/nasdaq.json', 'r') as f:
    nasdaq_tickers = json.load(f)

with open('data/tickers/nyse.json', 'r') as f:
    nyse_tickers = json.load(f)

other = list(set(nasdaq_tickers).union(set(nyse_tickers)) - set(sp500_tickers))
other_tickers = json.dumps(other)

with open('data/tickers/other.json', 'w') as f:
    json.dump(other_tickers, f)