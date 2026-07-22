import yfinance as yf
import matplotlib.pyplot as plt


ticker = "NVDA"


data = yf.download(
    ticker,
    start="2020-01-01",
    auto_adjust=True,
    progress=False,
)
closing_prices = data["Close"][ticker]
daily_returns = closing_prices.pct_change().dropna()

annualised_volatility = daily_returns.std() * (252 ** 0.5)
number_of_years = (data.index[-1] - data.index[0]).days / 365.25
annulised_return = (closing_prices.iloc[-1] / closing_prices.iloc[0]) ** (1 / number_of_years) - 1
Monthly_returns = closing_prices.resample('ME').ffill().pct_change().dropna()

print("First 5 rows of the data:")
print(data.head())
print("\nLast 5 rows of the data:")
print(data.tail())

print("\nSummary Statistics:")
print(data.describe())


starting_price = closing_prices.iloc[0]
ending_price = closing_prices.iloc[-1]

total_return = (ending_price/starting_price) - 1

print(f"Ticker: {ticker}")
print(f"Starting Price: ${starting_price:.2f}")
print(f"Ending Price: ${ending_price:.2f}")
print(f"Total Return: {total_return:.2%}")
print(daily_returns.describe())
print(f"Largest Gain: {daily_returns.max():.2%}", f"occurred on {daily_returns.idxmax().strftime('%Y-%m-%d')}")
print(f"Largest Loss: {daily_returns.min():.2%}", f"occurred on {daily_returns.idxmin().strftime('%Y-%m-%d')}")
print(f"Annualised Volatility: {annualised_volatility:.2%}")
print("\nAnnualised Return: {:.2%}".format(annulised_return))
print("\nWorst Month: {:.2%}".format(Monthly_returns.min()), f"occurred on {Monthly_returns.idxmin().strftime('%Y-%m')}")
print("Best Month: {:.2%}".format(Monthly_returns.max()), f"occurred on {Monthly_returns.idxmax().strftime('%Y-%m')}")

closing_prices.plot(figsize=(12, 6))

plt.title("Nvidia Share Price")
plt.xlabel("Date")
plt.ylabel("Price (USD)")

plt.show()
