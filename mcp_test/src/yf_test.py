import os
import yfinance as yf

data = yf.Ticker("MSFT")
#print(data.info)
#print(data.calendar)
# data.analyst_price_targets
# data.quarterly_income_stmt
#print(data.history(period='1mo'))
print(data.option_chain(data.options[0]).puts)