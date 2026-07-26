import yfinance as yf
import matplotlib.pyplot as plt


# Stock settings
ticker = "NVDA"


# Download the historical data
data = yf.download(
    ticker,
    start="2020-01-01",
    auto_adjust=True,
    progress=False,
)


# Extract the closing prices and remove any missing values
closing_prices = data["Close"][ticker].dropna()


# Calculate daily and monthly returns
daily_returns = closing_prices.pct_change().dropna()

Monthly_returns = (
    closing_prices
    .resample("ME")
    .ffill()
    .pct_change()
    .dropna()
)


# Calculate the length of the analysis period
number_of_years = (
    closing_prices.index[-1] - closing_prices.index[0]
).days / 365.25


# Calculate return and volatility
annualised_volatility = daily_returns.std() * (252 ** 0.5)

annulised_return = (
    closing_prices.iloc[-1] / closing_prices.iloc[0]
) ** (1 / number_of_years) - 1


# Display the downloaded data
print("First 5 rows of the data:")
print(data.head())

print("\nLast 5 rows of the data:")
print(data.tail())

print("\nSummary Statistics:")
print(data.describe())


# Starting and ending values
starting_price = closing_prices.iloc[0]
ending_price = closing_prices.iloc[-1]

total_return = (ending_price / starting_price) - 1


# Display the stock results
print(f"\nTicker: {ticker}")
print(f"Starting Price: ${starting_price:.2f}")
print(f"Ending Price: ${ending_price:.2f}")
print(f"Total Return: {total_return:.2%}")

print("\nDaily Return Statistics:")
print(daily_returns.describe())

print(
    f"Largest Gain: {daily_returns.max():.2%}",
    f"occurred on {daily_returns.idxmax().strftime('%Y-%m-%d')}",
)

print(
    f"Largest Loss: {daily_returns.min():.2%}",
    f"occurred on {daily_returns.idxmin().strftime('%Y-%m-%d')}",
)

print(f"Annualised Volatility: {annualised_volatility:.2%}")
print("\nAnnualised Return: {:.2%}".format(annulised_return))

print(
    "\nWorst Month: {:.2%}".format(Monthly_returns.min()),
    f"occurred on {Monthly_returns.idxmin().strftime('%Y-%m')}",
)

print(
    "Best Month: {:.2%}".format(Monthly_returns.max()),
    f"occurred on {Monthly_returns.idxmax().strftime('%Y-%m')}",
)


# Plot the historical closing price
closing_prices.plot(figsize=(12, 6))

plt.title("NVIDIA Share Price")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.show()