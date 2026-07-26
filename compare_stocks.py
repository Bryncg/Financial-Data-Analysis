"""
compare_stocks.py

Downloads historical stock data from Yahoo Finance and compares the
performance of multiple companies by calculating:

- Starting and ending prices
- Total return
- Annualised return (CAGR)
- Annualised volatility
- Best and worst trading day
- Best and worst calendar month

The script also generates a comparison table and plots the adjusted
share prices over time.
"""


import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# Download historical stock price data
# --------------------------------------------------

tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]

data = yf.download(
    tickers,
    start="2020-01-01",
    auto_adjust=True,
    progress=False,
)

# Extract closing prices and remove rows with missing values
closing_prices = data["Close"][tickers].dropna()


# --------------------------------------------------
# Calculate performance metrics
# --------------------------------------------------

starting_prices = closing_prices.iloc[0]
ending_prices = closing_prices.iloc[-1]

# Total return over the entire period
total_returns = (ending_prices / starting_prices) - 1

# Length of the analysis period in years
number_of_years = (
    closing_prices.index[-1] - closing_prices.index[0]
).days / 365.25

# Compound Annual Growth Rate
annualised_returns = (
    (ending_prices / starting_prices)
    ** (1 / number_of_years)
) - 1

# Daily percentage returns and annualised volatility
daily_returns = closing_prices.pct_change().dropna()
volatility = daily_returns.std() * (252 ** 0.5)


# --------------------------------------------------
# Find best and worst trading days
# --------------------------------------------------

worst_day = daily_returns.min()
worst_day_date = daily_returns.idxmin().dt.strftime("%Y-%m-%d")

best_day = daily_returns.max()
best_day_date = daily_returns.idxmax().dt.strftime("%Y-%m-%d")


# --------------------------------------------------
# Find best and worst calendar months
# --------------------------------------------------

# Use the final trading day of each month
monthly_prices = closing_prices.resample("ME").last()

# Remove the final month if it is incomplete
if (
    closing_prices.index[-1].to_period("M")
    == monthly_prices.index[-1].to_period("M")
):
    monthly_prices = monthly_prices.iloc[:-1]

monthly_returns = monthly_prices.pct_change().dropna()

worst_month = monthly_returns.min()
worst_month_date = monthly_returns.idxmin().dt.strftime("%Y-%m")

best_month = monthly_returns.max()
best_month_date = monthly_returns.idxmax().dt.strftime("%Y-%m")


# --------------------------------------------------
# Create a formatted comparison table
# --------------------------------------------------

comparison = pd.DataFrame({
    "Starting Price": starting_prices.map(
        lambda value: f"${value:,.2f}"
    ),

    "Ending Price": ending_prices.map(
        lambda value: f"${value:,.2f}"
    ),

    "Total Return": total_returns.map(
        lambda value: f"{value:.2%}"
    ),

    "Annualised Return": annualised_returns.map(
        lambda value: f"{value:.2%}"
    ),

    "Volatility": volatility.map(
        lambda value: f"{value:.2%}"
    ),

    "Worst Day": (
        worst_day.map(lambda value: f"{value:.2%}")
        + " | "
        + worst_day_date
    ),

    "Best Day": (
        best_day.map(lambda value: f"{value:.2%}")
        + " | "
        + best_day_date
    ),

    "Worst Month": (
        worst_month.map(lambda value: f"{value:.2%}")
        + " | "
        + worst_month_date
    ),

    "Best Month": (
        best_month.map(lambda value: f"{value:.2%}")
        + " | "
        + best_month_date
    ),
})

# Display metrics as rows and tickers as columns
comparison = comparison.T
comparison = comparison[tickers]


# --------------------------------------------------
# Display comparison table
# --------------------------------------------------

print("\nStock Comparison:\n")
print(comparison.to_string())


# --------------------------------------------------
# Plot adjusted closing prices
# --------------------------------------------------

closing_prices.plot(figsize=(12, 6))

plt.title("Stock Price Comparison")
plt.xlabel("Date")
plt.ylabel("Adjusted Share Price (USD)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()