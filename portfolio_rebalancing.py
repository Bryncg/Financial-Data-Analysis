# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Used later to show the graph values as full dollar amounts
from matplotlib.ticker import StrMethodFormatter


# =============================================================================
# PORTFOLIO SETTINGS
# =============================================================================

# Five stocks I want to compare in the portfolio
tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]

# S&P 500 used as the benchmark
benchmark_ticker = "^GSPC"

start_date = "2020-01-01"

# Both the portfolio and benchmark start at $100,000
initial_portfolio_value = 100_000
benchmark_initial_value = 100_000

# Equal 20% starting weight in each stock
portfolio_weights = {
    "NVDA": 0.20,
    "AAPL": 0.20,
    "MSFT": 0.20,
    "GOOGL": 0.20,
    "AMZN": 0.20,
}


# =============================================================================
# DOWNLOAD MARKET DATA
# =============================================================================

# Download adjusted prices for the stocks and S&P 500
# progress=False stops yfinance printing the loading bar
data = yf.download(
    tickers + [benchmark_ticker],
    start=start_date,
    auto_adjust=True,
    progress=False,
)


# =============================================================================
# PREPARE PRICE DATA
# =============================================================================

# Keep the closing prices and remove any dates with missing stock data
closing_prices = data["Close"][tickers].copy().dropna()

# Separate the benchmark prices from the stock prices
benchmark_prices = data["Close"][benchmark_ticker].copy().dropna()

# Turn the weights into a Series and make sure they match the stock column order
weights = pd.Series(portfolio_weights).reindex(closing_prices.columns)


# =============================================================================
# CHECK PORTFOLIO WEIGHTS
# =============================================================================

# Check that the weights total 100%, allowing for a tiny rounding difference
if abs(weights.sum() - 1.0) > 1e-9:
    raise ValueError("Portfolio weights must sum to 1.0.")

# Check that I have given every stock a weight
if weights.isna().any():
    raise ValueError("A portfolio weight is missing.")


# =============================================================================
# SHARED CALCULATIONS
# =============================================================================

# Work out how many years the data covers
# 365.25 is used to account for leap years
years = (
    closing_prices.index[-1] - closing_prices.index[0]
).days / 365.25

# Daily percentage return for each stock
daily_returns = closing_prices.pct_change().dropna()

# =============================================================================
# BUY AND HOLD STRATEGY
# =============================================================================

# Invest using the starting weights and then leave the shares alone.
# The weights will change over time as each stock performs differently.
buy_and_hold_portfolio_value = (
    initial_portfolio_value
    * (closing_prices / closing_prices.iloc[0]).dot(weights)
)

# Daily returns based on the portfolio value
buy_and_hold_portfolio_daily_returns = (
    buy_and_hold_portfolio_value
    .pct_change()
    .dropna()
)

buy_and_hold_portfolio_daily_returns.name = (
    "Buy and Hold Portfolio Returns"
)

# Annual growth rate across the full time period
buy_and_hold_portfolio_cagr = (
    buy_and_hold_portfolio_value.iloc[-1] / initial_portfolio_value
) ** (1 / years) - 1

# Compare the portfolio value with its previous highest value
buy_and_hold_drawdown_series = (
    buy_and_hold_portfolio_value
    / buy_and_hold_portfolio_value.cummax()
    - 1
)

# Worst fall from a previous peak
buy_and_hold_portfolio_max_drawdown = (
    buy_and_hold_drawdown_series.min()
)

# Annualised Sharpe ratio.
# This currently assumes a 0% risk-free rate.
buy_and_hold_portfolio_sharpe_ratio = (
    buy_and_hold_portfolio_daily_returns.mean()
    / buy_and_hold_portfolio_daily_returns.std()
    * (252 ** 0.5)
)


# =============================================================================
# DAILY REBALANCING STRATEGY
# =============================================================================

# Reset the portfolio back to the original weights every trading day
daily_rebalancing_portfolio_daily_returns = (
    daily_returns * weights
).sum(axis=1)

daily_rebalancing_portfolio_daily_returns.name = (
    "Daily Rebalancing Portfolio Returns"
)

# Compound the daily returns to build the portfolio value
daily_rebalancing_portfolio_value = (
    initial_portfolio_value
    * (daily_rebalancing_portfolio_daily_returns + 1).cumprod()
)

# Performance metrics
daily_rebalancing_portfolio_cagr = (
    daily_rebalancing_portfolio_value.iloc[-1]
    / initial_portfolio_value
) ** (1 / years) - 1

daily_rebalancing_drawdown_series = (
    daily_rebalancing_portfolio_value
    / daily_rebalancing_portfolio_value.cummax()
    - 1
)

daily_rebalancing_portfolio_max_drawdown = (
    daily_rebalancing_drawdown_series.min()
)

daily_rebalancing_portfolio_sharpe_ratio = (
    daily_rebalancing_portfolio_daily_returns.mean()
    / daily_rebalancing_portfolio_daily_returns.std()
    * (252 ** 0.5)
)

# =============================================================================
# MONTHLY REBALANCING STRATEGY
# =============================================================================

# Find the last trading day in each month
monthly_rebalancing_dates = (
    closing_prices
    .groupby(closing_prices.index.to_period("M"))
    .tail(1)
    .index
)

# Starting number of shares based on the original 20% weights
monthly_holdings = (
    initial_portfolio_value
    * weights
    / closing_prices.iloc[0]
)

monthly_portfolio_values = {}

# Go through each trading day and track the portfolio value
for date in closing_prices.index:

    # Current value of all shares held
    portfolio_worth_today = (
        monthly_holdings
        * closing_prices.loc[date]
    ).sum()

    monthly_portfolio_values[date] = portfolio_worth_today

    # Reset each stock back to its target weight at the end of the month
    if date in monthly_rebalancing_dates:
        monthly_holdings = (
            portfolio_worth_today
            * weights
            / closing_prices.loc[date]
        )

# Turn the stored daily values into a Series
monthly_rebalancing_portfolio_value = pd.Series(
    monthly_portfolio_values,
    name="Monthly Rebalancing Portfolio Value"
)

monthly_rebalancing_portfolio_daily_returns = (
    monthly_rebalancing_portfolio_value
    .pct_change()
    .dropna()
)

monthly_rebalancing_portfolio_daily_returns.name = (
    "Monthly Rebalancing Portfolio Returns"
)

# Performance metrics
monthly_rebalancing_portfolio_cagr = (
    monthly_rebalancing_portfolio_value.iloc[-1]
    / initial_portfolio_value
) ** (1 / years) - 1

monthly_rebalancing_drawdown_series = (
    monthly_rebalancing_portfolio_value
    / monthly_rebalancing_portfolio_value.cummax()
    - 1
)

monthly_rebalancing_portfolio_max_drawdown = (
    monthly_rebalancing_drawdown_series.min()
)

monthly_rebalancing_portfolio_sharpe_ratio = (
    monthly_rebalancing_portfolio_daily_returns.mean()
    / monthly_rebalancing_portfolio_daily_returns.std()
    * (252 ** 0.5)
)


# =============================================================================
# QUARTERLY REBALANCING STRATEGY
# =============================================================================

# Find the last trading day in each quarter
quarterly_rebalancing_dates = (
    closing_prices
    .groupby(closing_prices.index.to_period("Q"))
    .tail(1)
    .index
)

# Starting number of shares based on the original weights
quarterly_holdings = (
    initial_portfolio_value
    * weights
    / closing_prices.iloc[0]
)

quarterly_portfolio_values = {}

# Go through each trading day and track the portfolio value
for date in closing_prices.index:

    # Current value of all shares held
    portfolio_worth_today = (
        quarterly_holdings
        * closing_prices.loc[date]
    ).sum()

    quarterly_portfolio_values[date] = portfolio_worth_today

    # Reset each stock back to its target weight at the end of the quarter
    if date in quarterly_rebalancing_dates:
        quarterly_holdings = (
            portfolio_worth_today
            * weights
            / closing_prices.loc[date]
        )

# Turn the stored daily values into a Series
quarterly_rebalancing_portfolio_value = pd.Series(
    quarterly_portfolio_values,
    name="Quarterly Rebalancing Portfolio Value"
)

quarterly_rebalancing_portfolio_daily_returns = (
    quarterly_rebalancing_portfolio_value
    .pct_change()
    .dropna()
)

quarterly_rebalancing_portfolio_daily_returns.name = (
    "Quarterly Rebalancing Portfolio Returns"
)

# Performance metrics
quarterly_rebalancing_portfolio_cagr = (
    quarterly_rebalancing_portfolio_value.iloc[-1]
    / initial_portfolio_value
) ** (1 / years) - 1

quarterly_rebalancing_drawdown_series = (
    quarterly_rebalancing_portfolio_value
    / quarterly_rebalancing_portfolio_value.cummax()
    - 1
)

quarterly_rebalancing_portfolio_max_drawdown = (
    quarterly_rebalancing_drawdown_series.min()
)

quarterly_rebalancing_portfolio_sharpe_ratio = (
    quarterly_rebalancing_portfolio_daily_returns.mean()
    / quarterly_rebalancing_portfolio_daily_returns.std()
    * (252 ** 0.5)
)

# =============================================================================
# ANNUAL REBALANCING STRATEGY
# =============================================================================

# Find the last trading day in each year
annual_rebalancing_dates = (
    closing_prices
    .groupby(closing_prices.index.to_period("Y"))
    .tail(1)
    .index
)

# Starting number of shares based on the original weights
annual_holdings = (
    initial_portfolio_value
    * weights
    / closing_prices.iloc[0]
)

annual_portfolio_values = {}

# Go through each trading day and track the portfolio value
for date in closing_prices.index:

    # Current value of all shares held
    portfolio_worth_today = (
        annual_holdings
        * closing_prices.loc[date]
    ).sum()

    annual_portfolio_values[date] = portfolio_worth_today

    # Reset each stock back to its target weight at the end of the year
    if date in annual_rebalancing_dates:
        annual_holdings = (
            portfolio_worth_today
            * weights
            / closing_prices.loc[date]
        )

# Turn the stored daily values into a Series
annual_rebalancing_portfolio_value = pd.Series(
    annual_portfolio_values,
    name="Annual Rebalancing Portfolio Value"
)

annual_rebalancing_portfolio_daily_returns = (
    annual_rebalancing_portfolio_value
    .pct_change()
    .dropna()
)

annual_rebalancing_portfolio_daily_returns.name = (
    "Annual Rebalancing Portfolio Returns"
)

# Performance metrics
annual_rebalancing_portfolio_cagr = (
    annual_rebalancing_portfolio_value.iloc[-1]
    / initial_portfolio_value
) ** (1 / years) - 1

annual_rebalancing_drawdown_series = (
    annual_rebalancing_portfolio_value
    / annual_rebalancing_portfolio_value.cummax()
    - 1
)

annual_rebalancing_portfolio_max_drawdown = (
    annual_rebalancing_drawdown_series.min()
)

annual_rebalancing_portfolio_sharpe_ratio = (
    annual_rebalancing_portfolio_daily_returns.mean()
    / annual_rebalancing_portfolio_daily_returns.std()
    * (252 ** 0.5)
)


# =============================================================================
# S&P 500 BENCHMARK
# =============================================================================

# Scale the benchmark so it also starts at $100,000
benchmark_value = (
    benchmark_initial_value
    * benchmark_prices
    / benchmark_prices.iloc[0]
)

# Daily returns for the benchmark
benchmark_daily_returns = (
    benchmark_value
    .pct_change()
    .dropna()
)

# Number of years covered by the benchmark data
benchmark_years = (
    benchmark_value.index[-1] - benchmark_value.index[0]
).days / 365.25

# Performance metrics
benchmark_cagr = (
    benchmark_value.iloc[-1]
    / benchmark_value.iloc[0]
) ** (1 / benchmark_years) - 1

benchmark_volatility = (
    benchmark_daily_returns.std()
    * (252 ** 0.5)
)

# Annualised Sharpe ratio (0% risk-free rate)
benchmark_sharpe_ratio = (
    benchmark_daily_returns.mean()
    / benchmark_daily_returns.std()
    * (252 ** 0.5)
)

# Worst fall from a previous peak
benchmark_max_drawdown = (
    benchmark_value
    / benchmark_value.cummax()
    - 1
).min()

# =============================================================================
# STRATEGY COMPARISON TABLE
# =============================================================================

# Put the main results for each strategy into one table
strategy_comparison = pd.DataFrame({
    "strategy": [
        "Buy and Hold",
        "Daily Rebalancing",
        "Monthly Rebalancing",
        "Quarterly Rebalancing",
        "Annual Rebalancing",
        "Benchmark (S&P 500)",
    ],

    # Value at the end of the data
    "Final Value": [
        f"${buy_and_hold_portfolio_value.iloc[-1]:,.2f}",
        f"${daily_rebalancing_portfolio_value.iloc[-1]:,.2f}",
        f"${monthly_rebalancing_portfolio_value.iloc[-1]:,.2f}",
        f"${quarterly_rebalancing_portfolio_value.iloc[-1]:,.2f}",
        f"${annual_rebalancing_portfolio_value.iloc[-1]:,.2f}",
        f"${benchmark_value.iloc[-1]:,.2f}",
    ],

    # Return across the full period
    "Total Return": [
        buy_and_hold_portfolio_value.iloc[-1]
        / initial_portfolio_value
        - 1,

        daily_rebalancing_portfolio_value.iloc[-1]
        / initial_portfolio_value
        - 1,

        monthly_rebalancing_portfolio_value.iloc[-1]
        / initial_portfolio_value
        - 1,

        quarterly_rebalancing_portfolio_value.iloc[-1]
        / initial_portfolio_value
        - 1,

        annual_rebalancing_portfolio_value.iloc[-1]
        / initial_portfolio_value
        - 1,

        benchmark_value.iloc[-1]
        / benchmark_value.iloc[0]
        - 1,
    ],

    "CAGR": [
        buy_and_hold_portfolio_cagr,
        daily_rebalancing_portfolio_cagr,
        monthly_rebalancing_portfolio_cagr,
        quarterly_rebalancing_portfolio_cagr,
        annual_rebalancing_portfolio_cagr,
        benchmark_cagr,
    ],

    "Volatility": [
        buy_and_hold_portfolio_daily_returns.std() * (252 ** 0.5),
        daily_rebalancing_portfolio_daily_returns.std() * (252 ** 0.5),
        monthly_rebalancing_portfolio_daily_returns.std() * (252 ** 0.5),
        quarterly_rebalancing_portfolio_daily_returns.std() * (252 ** 0.5),
        annual_rebalancing_portfolio_daily_returns.std() * (252 ** 0.5),
        benchmark_volatility,
    ],

    "Sharpe": [
        buy_and_hold_portfolio_sharpe_ratio,
        daily_rebalancing_portfolio_sharpe_ratio,
        monthly_rebalancing_portfolio_sharpe_ratio,
        quarterly_rebalancing_portfolio_sharpe_ratio,
        annual_rebalancing_portfolio_sharpe_ratio,
        benchmark_sharpe_ratio,
    ],

    "Max Drawdown": [
        buy_and_hold_portfolio_max_drawdown,
        daily_rebalancing_portfolio_max_drawdown,
        monthly_rebalancing_portfolio_max_drawdown,
        quarterly_rebalancing_portfolio_max_drawdown,
        annual_rebalancing_portfolio_max_drawdown,
        benchmark_max_drawdown,
    ]
})

# Use the strategy names as the row labels
strategy_comparison = strategy_comparison.set_index("strategy")

# Keep a separate copy for formatting so the original values stay numeric
formatted_strategy_comparison = strategy_comparison.copy()

# Show the main return and risk figures as percentages
for column in ["Total Return", "CAGR", "Volatility", "Max Drawdown"]:
    formatted_strategy_comparison[column] = (
        formatted_strategy_comparison[column]
        .map(lambda value: f"{value:.2%}")
    )

# Keep the Sharpe ratio to two decimal places
formatted_strategy_comparison["Sharpe"] = (
    formatted_strategy_comparison["Sharpe"]
    .map(lambda value: f"{value:.2f}")
)

print("\nStrategy Comparison")
print(formatted_strategy_comparison)


# =============================================================================
# PLOT PORTFOLIO STRATEGY VALUES
# =============================================================================

# Plot every strategy on the same graph
plt.figure(figsize=(12, 6))

plt.plot(
    buy_and_hold_portfolio_value,
    label="Buy and Hold Portfolio",
)

plt.plot(
    daily_rebalancing_portfolio_value,
    label="Daily Rebalancing Portfolio",
)

plt.plot(
    monthly_rebalancing_portfolio_value,
    label="Monthly Rebalancing Portfolio",
)

plt.plot(
    quarterly_rebalancing_portfolio_value,
    label="Quarterly Rebalancing Portfolio",
)

plt.plot(
    annual_rebalancing_portfolio_value,
    label="Annual Rebalancing Portfolio",
)

# Dashed line makes the benchmark easier to separate from the portfolio strategies
plt.plot(
    benchmark_value,
    label="S&P 500 Benchmark",
    linestyle="--"
)

plt.title("Portfolio Strategy Value Comparison")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")

plt.grid(alpha=0.3)
plt.legend()

# Show full dollar values instead of scientific notation
plt.gca().yaxis.set_major_formatter(
    StrMethodFormatter('${x:,.0f}')
)

plt.tight_layout()
plt.show()

# =============================================================================
# FUTURE IDEA - COMBINE THE REBALANCING SECTIONS
# =============================================================================

# The Monthly, Quarterly and Annual sections all do nearly the same thing.
# The main difference is just whether I use "M", "Q" or "Y".
#
# I have left them separate for now because it is easier for me to follow
# how each strategy works.
#
# Later on I could shorten the code by running them all through one loop.


# rebalancing_periods = {
#     "Monthly": "M",
#     "Quarterly": "Q",
#     "Annual": "Y",
# }

# periodic_results = {}

# # Run the same process for each rebalancing period
# for strategy_name, period_code in rebalancing_periods.items():

#     # Find the last trading day for the chosen period
#     rebalancing_dates = (
#         closing_prices
#         .groupby(closing_prices.index.to_period(period_code))
#         .tail(1)
#         .index
#     )

#     # Starting number of shares based on the original weights
#     holdings = (
#         initial_portfolio_value
#         * weights
#         / closing_prices.iloc[0]
#     )

#     portfolio_values = {}

#     # Go through each trading day and track the portfolio value
#     for date in closing_prices.index:

#         portfolio_worth_today = (
#             holdings
#             * closing_prices.loc[date]
#         ).sum()

#         portfolio_values[date] = portfolio_worth_today

#         # Reset the holdings when a rebalance date is reached
#         if date in rebalancing_dates:
#             holdings = (
#                 portfolio_worth_today
#                 * weights
#                 / closing_prices.loc[date]
#             )

#     # Turn the stored values into a Series
#     portfolio_value = pd.Series(
#         portfolio_values,
#         name=f"{strategy_name} Portfolio Value"
#     )

#     portfolio_daily_returns = (
#         portfolio_value
#         .pct_change()
#         .dropna()
#     )

#     # Performance metrics
#     portfolio_cagr = (
#         portfolio_value.iloc[-1]
#         / initial_portfolio_value
#     ) ** (1 / years) - 1

#     drawdown_series = (
#         portfolio_value
#         / portfolio_value.cummax()
#         - 1
#     )

#     portfolio_max_drawdown = drawdown_series.min()

#     portfolio_sharpe_ratio = (
#         portfolio_daily_returns.mean()
#         / portfolio_daily_returns.std()
#         * (252 ** 0.5)
#     )

#     # Store the results for each strategy
#     periodic_results[strategy_name] = {
#         "Portfolio Value": portfolio_value,
#         "Daily Returns": portfolio_daily_returns,
#         "CAGR": portfolio_cagr,
#         "Max Drawdown": portfolio_max_drawdown,
#         "Sharpe Ratio": portfolio_sharpe_ratio,
#     }


# I could then get the results for each strategy like this:
#
# periodic_results["Monthly"]["Portfolio Value"]
# periodic_results["Quarterly"]["CAGR"]
# periodic_results["Annual"]["Sharpe Ratio"]