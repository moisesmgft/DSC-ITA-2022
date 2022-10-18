import pandas_datareader as pdr
from datetime import datetime
import json

start_date = datetime(1970, 1, 1)
end_date = datetime.now().date()

filename = 'data/tickers/sp500.json'
dst_path = 'data/yahoo/sp500/'

# filename = 'data/tickers/other.json'
# dst_path = 'data/yahoo/other/'


with open(filename, 'r') as f:
    sp500_tickers = json.load(f)

sp500_success = []
sp500_failed = []

for ticker in sp500_tickers:
    
    if (len(sp500_failed) >= 100):
        break
    ticker = ticker.replace(".", "-")
    
    try:
        df = pdr.get_data_yahoo(symbols=ticker, start=start_date, end=end_date)
        df.drop(df.columns[[0,1,2,4,5]], axis=1, inplace=True)
        filename = dst_path + ticker + '.csv'
        df.to_csv(filename, index=True)

        sp500_success.append(ticker)
    except:
        sp500_failed.append(ticker)

failed_json = json.dumps(sp500_failed)
with open('log/sp500_download_fail.json', 'w') as f:
    json.dump(failed_json, f)

print(sp500_failed)