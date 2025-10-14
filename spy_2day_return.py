import yfinance as yf
start_date = '2025-8-05' 
end_date = '2025-10-11'
spy = yf.download('spy', start=start_date, end=end_date)
open_first = spy.iloc[0]['Open']
close_second = spy.iloc[-1]['Close']
pct_change = (close_second - open_first) / open_first
start = 1637.55
end = start * (1 + pct_change)
print(str(pct_change) + '%' + " " + str(end))
