import yfinance as yf
start_date = '2025-09-15' 
end_date = '2025-09-19'
spy = yf.download('spy', start=start_date, end=end_date)
open_first = spy.iloc[0]['Open']
close_second = spy.iloc[-1]['Close']
pct_change = (close_second - open_first) / open_first * 100
print(str(pct_change) + '%')
