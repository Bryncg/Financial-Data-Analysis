import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


# Portfolio settings
tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]
start_date = "2020-01-01"
initial_portfolio_value = 100_000

portfolio_weights = {
    "NVDA": 0.20,
    "AAPL": 0.20,
    "MSFT": 0.20,
    "GOOGL": 0.20,
    "AMZN": 0.20,
}


# Download adjusted historical prices
data = yf.download(
    tickers,
    start=start_date,
    auto_adjust=True,
    progress=False,
)

closing_prices = data["Close"][tickers].copy()


# Convert the weights into a labelled Pandas Series
weights = pd.Series(portfolio_weights).reindex(closing_prices.columns)

# Check that all of the available capital has been allocated
if abs(weights.sum() - 1.0) > 1e-9:
    raise ValueError("Portfolio weights must sum to 1.0.")


# Calculate the daily return of each stock
daily_returns = closing_prices.pct_change().dropna()

# Apply the target weights each day to model a daily-rebalanced portfolio
portfolio_daily_returns = (daily_returns * weights).sum(axis=1)
portfolio_daily_returns.name = "Portfolio Return"


# Annualise the standard deviation of daily portfolio returns
portfolio_volatility = portfolio_daily_returns.std() * (252 ** 0.5)


# Compound the daily returns to produce the portfolio growth path
portfolio_growth = (1 + portfolio_daily_returns).cumprod()
portfolio_values = initial_portfolio_value * portfolio_growth

final_portfolio_value = portfolio_values.iloc[-1]
portfolio_profit = final_portfolio_value - initial_portfolio_value
portfolio_total_return = portfolio_growth.iloc[-1] - 1


# Display the main results
print("Portfolio Weights:")
print(weights.map(lambda value: f"{value:.2%}"))

print("\nPortfolio Daily Returns:")
print(portfolio_daily_returns.head())

print(f"\nAnnualised Portfolio Volatility: {portfolio_volatility:.2%}")
print(f"Starting Portfolio Value: ${initial_portfolio_value:,.2f}")
print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
print(f"Portfolio Profit: ${portfolio_profit:,.2f}")
print(f"Portfolio Total Return: {portfolio_total_return:.2%}")


# Plot the value of the portfolio over time
plt.figure(figsize=(12, 6))
plt.plot(
    portfolio_values,
    label="Equal-Weight Portfolio",
)

plt.title("Equal-Weight Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()