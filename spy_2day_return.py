import yfinance as yf


# Set start and end dates for a 2-day window
end_date = '2025-08-22'
start_date = '2025-08-18' 

spy = yf.download('SPY', start=start_date, end=end_date)

if len(spy) >= 2:
    open_first = spy.iloc[0]['Open']
    close_second = spy.iloc[1]['Close']
    pct_change = (close_second - open_first) / open_first * 100
    print(pct_change)
else:
    print("Not enough data to compute percent change.")
