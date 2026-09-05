import yfinance as yf

ticker = yf.Ticker("INFY.NS")
dividend_data = ticker.quarterly_balance_sheet
print("INFY.NS Dividend Data:")
print(dividend_data)

