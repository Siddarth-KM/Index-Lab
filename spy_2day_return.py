import yfinance as yf
start_date = '2025-09-02' 
end_date = '2025-09-05'
spx = yf.download('spy', start=start_date, end=end_date)
open_first = spx.iloc[0]['Open']
close_second = spx.iloc[1]['Close']
pct_change = (close_second - open_first) / open_first * 100
print(str(pct_change) + '%')
