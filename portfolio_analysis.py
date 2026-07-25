import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Portfolio setup
# ============================================================

tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]
benchmark_ticker = "^GSPC"

start_date = "2020-01-01"
initial_portfolio_value = 100_000
benchmark_initial_value = 100_000

portfolio_weights = {
    "NVDA": 0.20,
    "AAPL": 0.20,
    "MSFT": 0.20,
    "GOOGL": 0.20,
    "AMZN": 0.20,
}


# ============================================================
# Download and prepare the price data
# ============================================================

# Download adjusted closing prices for the portfolio and S&P 500
data = yf.download(
    tickers + [benchmark_ticker],
    start=start_date,
    auto_adjust=True,
    progress=False,
)

closing_prices = data["Close"][tickers].copy()
benchmark_prices = data["Close"][benchmark_ticker].copy()

# Find the length of the analysis period in years
years = (
    closing_prices.index[-1] - closing_prices.index[0]
).days / 365.25


# ============================================================
# Portfolio weights and daily returns
# ============================================================

# Use a labelled Series so each weight stays matched to its ticker
weights = pd.Series(portfolio_weights).reindex(closing_prices.columns)

# Make sure the full portfolio has been allocated
if abs(weights.sum() - 1.0) > 1e-9:
    raise ValueError("Portfolio weights must sum to 1.0.")

# Calculate the daily percentage return of each stock
daily_returns = closing_prices.pct_change().dropna()

# Apply the target weights each day
# This models a portfolio that is rebalanced daily
portfolio_daily_returns = (daily_returns * weights).sum(axis=1)
portfolio_daily_returns.name = "Portfolio Return"


# ============================================================
# Portfolio performance and risk metrics
# ============================================================

# Annualised volatility based on 252 trading days
portfolio_volatility = (
    portfolio_daily_returns.std() * (252 ** 0.5)
)

# Compound the daily returns to create the portfolio value series
portfolio_growth = (1 + portfolio_daily_returns).cumprod()
portfolio_values = initial_portfolio_value * portfolio_growth

final_portfolio_value = portfolio_values.iloc[-1]
portfolio_profit = final_portfolio_value - initial_portfolio_value
portfolio_total_return = portfolio_growth.iloc[-1] - 1

# Annual compound growth rate over the full analysis period
portfolio_cagr = (
    final_portfolio_value / initial_portfolio_value
) ** (1 / years) - 1

# Measure the portfolio's decline from each previous high
drawdown_series = (
    portfolio_values / portfolio_values.cummax() - 1
)

portfolio_max_drawdown = drawdown_series.min()

# Sharpe ratio assuming a risk-free rate of zero
portfolio_sharpe_ratio = (
    portfolio_daily_returns.mean()
    / portfolio_daily_returns.std()
    * (252 ** 0.5)
)

# Calculate maximum drawdown separately for each stock
maximum_drawdown_stock = (
    closing_prices / closing_prices.cummax() - 1
).min()

# Match the benchmark dates to the portfolio value series
benchmark_prices = benchmark_prices.reindex(
    portfolio_values.index
).dropna()


# ============================================================
# Portfolio output
# ============================================================

print("Portfolio Weights:")
print(weights.map(lambda value: f"{value:.2%}"))

print("\nPortfolio Daily Returns:")
print(portfolio_daily_returns.head())


# Build a compact summary of the portfolio results
portfolio_summary = pd.DataFrame(
    [
        [
            "Starting Portfolio Value",
            f"${initial_portfolio_value:,.2f}",
        ],
        [
            "Final Portfolio Value",
            f"${final_portfolio_value:,.2f}",
        ],
        [
            "Portfolio Profit",
            f"${portfolio_profit:,.2f}",
        ],
        [
            "Portfolio Total Return",
            f"{portfolio_total_return:.2%}",
        ],
        [
            "Compound Annual Growth Rate (CAGR)",
            f"{portfolio_cagr:.2%}",
        ],
        [
            "Annualised Portfolio Volatility",
            f"{portfolio_volatility:.2%}",
        ],
        [
            "Maximum Drawdown",
            f"{portfolio_max_drawdown:.2%}",
        ],
        [
            "Sharpe Ratio",
            f"{portfolio_sharpe_ratio:.2f}",
        ],
    ],
    columns=["Metric", "Value"],
)

print("\nPortfolio Summary:")
print("Metric                              Value")
print("---------------------------------------------")

for metric, value in portfolio_summary.values:
    print(f"{metric:<35} {value}")


# ============================================================
# Individual stock comparison
# ============================================================

comparison = pd.DataFrame(
    {
        "Starting Price": closing_prices.iloc[0].map(
            lambda value: f"${value:,.2f}"
        ),
        "Ending Price": closing_prices.iloc[-1].map(
            lambda value: f"${value:,.2f}"
        ),
        "Total Return": (
            closing_prices.iloc[-1]
            / closing_prices.iloc[0]
            - 1
        ).map(
            lambda value: f"{value:.2%}"
        ),
        "CAGR": (
            (
                closing_prices.iloc[-1]
                / closing_prices.iloc[0]
            )
            ** (1 / years)
            - 1
        ).map(
            lambda value: f"{value:.2%}"
        ),
        "Annualised Volatility": (
            daily_returns.std() * (252 ** 0.5)
        ).map(
            lambda value: f"{value:.2%}"
        ),
        "Sharpe Ratio": (
            daily_returns.mean()
            / daily_returns.std()
            * (252 ** 0.5)
        ).map(
            lambda value: f"{value:.2f}"
        ),
        "Maximum Drawdown": maximum_drawdown_stock.map(
            lambda value: f"{value:.2%}"
        ),
    }
)

comparison = comparison.T
comparison = comparison[tickers]

print("\nIndividual Stock Performance:")
print(comparison.to_string())


# ============================================================
# Portfolio and benchmark plot
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    portfolio_values,
    label="Equal-Weight Portfolio",
)

plt.plot(
    benchmark_initial_value
    * (benchmark_prices / benchmark_prices.iloc[0]),
    label="S&P 500 (Benchmark)",
)

plt.title("Equal-Weight Portfolio vs S&P 500")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Portfolio drawdown plot
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    drawdown_series,
    label="Portfolio Drawdown",
)

plt.title("Portfolio Drawdown Over Time")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.grid(alpha=0.3)

percentage_formatter = plt.FuncFormatter(
    lambda value, _: f"{value:.2%}"
)

plt.gca().yaxis.set_major_formatter(percentage_formatter)
plt.legend()
plt.tight_layout()
plt.show()