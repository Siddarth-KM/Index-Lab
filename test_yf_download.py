import yfinance as yf
import pandas as pd
from datetime import datetime

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM']
start_date = '2024-01-01'
end_date = '2025-07-16'

print('yfinance version:', yf.__version__)
print('pandas version:', pd.__version__)

print('\n--- Batch download ---')
df = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=False)
print(df)
print('Columns:', df.columns)
print('Index:', df.index)

if df.empty or (len(tickers) > 1 and not isinstance(df.columns, pd.MultiIndex)):
    print('\nBatch download failed or returned empty. Trying per-ticker download:')
    for ticker in tickers:
        try:
            tdf = yf.download(ticker, start=start_date, end=end_date, progress=False)
            print(f'\nTicker: {ticker}')
            print(tdf)
            if tdf.empty:
                print(f'  [EMPTY]')
            else:
                print(f'  [ROWS: {len(tdf)}]')
        except Exception as e:
            print(f'  [ERROR for {ticker}]: {e}') 