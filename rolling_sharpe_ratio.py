import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Download market data
# --------------------------------------------------

# Download daily stock data.
data = yf.download(
    [
        "NVDA",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
    ],
    start="2020-01-01",
    end=None,
    auto_adjust=True,
    progress=False,
)

# Download the 13-week Treasury bill yield.
treasury_yield_data = yf.download(
    "^IRX",
    start="2020-01-01",
    end=None,
    auto_adjust=True,
    progress=False,
)

# Uncomment to inspect the Treasury data.
# print("\nTreasury Yield Data:")
# print(treasury_yield_data.head())
# print(treasury_yield_data.tail())


# --------------------------------------------------
# Rename stock tickers
# --------------------------------------------------

stock_labels = {
    "NVDA": "NVIDIA",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
}


# --------------------------------------------------
# Closing prices and daily returns
# --------------------------------------------------

# Keep only the adjusted closing prices.
closing_prices = (
    data["Close"]
    .rename(columns=stock_labels)
    .dropna()
)

# Convert closing prices into daily returns.
daily_returns = closing_prices.pct_change().dropna()

daily_returns.index.name = None
daily_returns.columns.name = None


# --------------------------------------------------
# Portfolio weights
# --------------------------------------------------

# Equal-weight portfolio.
portfolio_weights = {
    "NVIDIA": 0.20,
    "Apple": 0.20,
    "Microsoft": 0.20,
    "Alphabet": 0.20,
    "Amazon": 0.20,
}

# Match the weights to the order of the return columns.
weights = (
    pd.Series(portfolio_weights)
    .reindex(daily_returns.columns)
)

# Check that every stock has a matching weight.
if weights.isnull().any():
    raise ValueError("A portfolio weight is missing.")

# Check that the portfolio is fully invested.
if abs(weights.sum() - 1.0) > 1e-6:
    raise ValueError("Portfolio weights must sum to 1.")


# --------------------------------------------------
# Portfolio daily returns
# --------------------------------------------------

portfolio_daily_returns = (
    daily_returns
    @ weights
)


# --------------------------------------------------
# Risk-free rate
# --------------------------------------------------

# Use the 13-week Treasury bill yield as the
# risk-free rate proxy.
annual_risk_free_rate = (
    treasury_yield_data["Close"]
    .squeeze()
    / 100
)

# Match Treasury rates to portfolio trading dates.
annual_risk_free_rate = (
    annual_risk_free_rate
    .reindex(portfolio_daily_returns.index)
    .ffill()
)

# Check that every portfolio date has a risk-free rate.
if annual_risk_free_rate.isnull().any():
    raise ValueError(
        "Risk-free rate data is missing for some portfolio dates."
    )

# Convert the annual Treasury yield into a daily rate.
daily_risk_free_rate = (
    annual_risk_free_rate / 252
)

# Calculate daily portfolio excess returns.
daily_excess_returns = (
    portfolio_daily_returns
    - daily_risk_free_rate
)


# --------------------------------------------------
# Rolling window settings
# --------------------------------------------------

rolling_window = 60


# --------------------------------------------------
# Rolling covariance
# --------------------------------------------------

# Calculate a covariance matrix using the previous
# 60 trading days for every available date.
rolling_covariance = (
    daily_returns
    .rolling(rolling_window)
    .cov()
    .dropna()
)

# Keep one copy of each date from the MultiIndex.
rolling_dates = (
    rolling_covariance
    .index
    .get_level_values(0)
    .unique()
)

# Convert daily covariance into annualised covariance.
annualised_rolling_covariance = (
    rolling_covariance * 252
)


# --------------------------------------------------
# Rolling portfolio risk
# --------------------------------------------------

rolling_portfolio_variance = []
rolling_portfolio_volatility = []

# Calculate portfolio variance and volatility
# for every rolling date.
for date in rolling_dates:
    rolling_cov = (
        annualised_rolling_covariance.loc[date]
    )

    portfolio_variance = (
        weights.T
        @ rolling_cov
        @ weights
    )

    portfolio_volatility = np.sqrt(
        portfolio_variance
    )

    rolling_portfolio_variance.append(
        portfolio_variance
    )

    rolling_portfolio_volatility.append(
        portfolio_volatility
    )


# --------------------------------------------------
# Convert rolling risk results to Series
# --------------------------------------------------

rolling_portfolio_variance = pd.Series(
    rolling_portfolio_variance,
    index=rolling_dates,
    name="Rolling Portfolio Variance",
)

rolling_portfolio_volatility = pd.Series(
    rolling_portfolio_volatility,
    index=rolling_dates,
    name="Rolling Portfolio Volatility",
)


# --------------------------------------------------
# Rolling portfolio returns
# --------------------------------------------------

# Calculate the average portfolio return over each
# 60-day rolling window.
rolling_portfolio_returns = (
    portfolio_daily_returns
    .rolling(rolling_window)
    .mean()
    .dropna()
)

# Annualise the rolling average portfolio return.
annualised_rolling_portfolio_returns = (
    rolling_portfolio_returns * 252
)

# Check that rolling return and volatility dates match.
if not annualised_rolling_portfolio_returns.index.equals(
    rolling_portfolio_volatility.index
):
    raise ValueError(
        "Data alignment issue: Returns and volatility indices do not match."
    )


# --------------------------------------------------
# Volatility cross-check
# --------------------------------------------------

# Calculate rolling volatility directly from the
# portfolio return series as a second method.
direct_rolling_volatility = (
    portfolio_daily_returns
    .rolling(rolling_window)
    .std()
    * np.sqrt(252)
).dropna()

if not direct_rolling_volatility.index.equals(
    rolling_portfolio_volatility.index
):
    raise ValueError(
        "Volatility validation indices do not match."
    )

volatility_difference = (
    direct_rolling_volatility
    - rolling_portfolio_volatility
)

# Uncomment to verify that both methods give
# effectively the same volatility.
# print(
#     "\nMaximum Volatility Difference:"
# )
# print(
#     volatility_difference.abs().max()
# )


# --------------------------------------------------
# Rolling excess returns
# --------------------------------------------------

# Calculate the rolling average of daily excess returns.
rolling_excess_returns = (
    daily_excess_returns
    .rolling(rolling_window)
    .mean()
    .dropna()
)

# Annualise the rolling excess return.
annualised_rolling_excess_returns = (
    rolling_excess_returns * 252
)

# Check that excess returns and volatility use
# the same rolling dates.
if not annualised_rolling_excess_returns.index.equals(
    rolling_portfolio_volatility.index
):
    raise ValueError(
        "Excess returns and volatility indices do not match."
    )


# --------------------------------------------------
# Rolling Sharpe Ratio
# --------------------------------------------------

rolling_sharpe_ratio = (
    annualised_rolling_excess_returns
    / rolling_portfolio_volatility
)

# Uncomment to inspect the rolling Sharpe values.
# print("\nRolling Sharpe Ratio:")
# print(rolling_sharpe_ratio.head())
# print(rolling_sharpe_ratio.tail())


# --------------------------------------------------
# Sharpe Ratio summary statistics
# --------------------------------------------------

current_sharpe_ratio = (
    rolling_sharpe_ratio.iloc[-1]
)

average_sharpe_ratio = (
    rolling_sharpe_ratio.mean()
)

max_sharpe_ratio = (
    rolling_sharpe_ratio.max()
)

date_of_max_sharpe_ratio = (
    rolling_sharpe_ratio.idxmax()
)

min_sharpe_ratio = (
    rolling_sharpe_ratio.min()
)

date_of_min_sharpe_ratio = (
    rolling_sharpe_ratio.idxmin()
)

latest_annual_risk_free_rate = (
    annual_risk_free_rate.iloc[-1]
)


# --------------------------------------------------
# Display summary statistics
# --------------------------------------------------

print("\nRolling Sharpe Ratio Summary Statistics:")

print(
    f"Current Sharpe Ratio: "
    f"{current_sharpe_ratio:.4f}"
)

print(
    f"Average Sharpe Ratio: "
    f"{average_sharpe_ratio:.4f}"
)

print(
    f"Maximum Sharpe Ratio: "
    f"{max_sharpe_ratio:.4f} "
    f"on {date_of_max_sharpe_ratio.date()}"
)

print(
    f"Minimum Sharpe Ratio: "
    f"{min_sharpe_ratio:.4f} "
    f"on {date_of_min_sharpe_ratio.date()}"
)

print(
    f"Current 13-Week Treasury Yield: "
    f"{latest_annual_risk_free_rate:.2%}"
)


# --------------------------------------------------
# Historical market events
# --------------------------------------------------

# Manually selected dates used to give historical
# context to changes in risk-adjusted performance.
market_events = {
    "COVID-19 Crash": "2020-03-16",
    "COVID-19 Recovery": "2020-08-18",
    "2022 Market Correction": "2022-06-16",
    "2022 Inflation and Rate Hikes": "2022-10-13",
    "Regional Banking Crisis": "2023-03-10",
    "2024 Market Recovery": "2024-06-14",
    "DeepSeek Tech Sell-off": "2025-01-27",
    "Tariff Market Crash": "2025-04-03",
}


# --------------------------------------------------
# Plot rolling Sharpe Ratio
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(15, 8)
)

ax.plot(
    rolling_sharpe_ratio.index,
    rolling_sharpe_ratio,
    label="Rolling Sharpe Ratio",
    color="blue",
    linewidth=2,
)

ax.set_title(
    "60-Day Rolling Portfolio Sharpe Ratio",
    fontsize=16,
)

ax.set_xlabel("Date")

ax.set_ylabel("Sharpe Ratio")


# --------------------------------------------------
# Reference lines
# --------------------------------------------------

# Show the average Sharpe Ratio across the full period.
ax.axhline(
    y=average_sharpe_ratio,
    color="red",
    linestyle="--",
    label=(
        f"Average Sharpe Ratio: "
        f"{average_sharpe_ratio:.4f}"
    ),
)

# Separate positive and negative excess performance.
ax.axhline(
    y=0,
    color="black",
    linestyle="--",
    linewidth=1,
    label="Zero Sharpe",
)


# --------------------------------------------------
# Mark important Sharpe Ratio values
# --------------------------------------------------

# Mark maximum rolling Sharpe Ratio.
ax.scatter(
    date_of_max_sharpe_ratio,
    max_sharpe_ratio,
    color="green",
    zorder=5,
    label=(
        f"Max Sharpe Ratio: "
        f"{max_sharpe_ratio:.4f}"
    ),
)

ax.annotate(
    (
        f"Max Sharpe Ratio: {max_sharpe_ratio:.4f}\n"
        f"{date_of_max_sharpe_ratio.date()}"
    ),
    xy=(
        date_of_max_sharpe_ratio,
        max_sharpe_ratio,
    ),
    xytext=(30, -25),
    textcoords="offset points",
    ha="left",
    bbox={
        "boxstyle": "round",
        "fc": "white",
        "ec": "green",
        "alpha": 0.9,
    },
    arrowprops={
        "arrowstyle": "->",
        "color": "black",
    },
)

# Mark minimum rolling Sharpe Ratio.
ax.scatter(
    date_of_min_sharpe_ratio,
    min_sharpe_ratio,
    color="orange",
    zorder=5,
    label=(
        f"Min Sharpe Ratio: "
        f"{min_sharpe_ratio:.4f}"
    ),
)

ax.annotate(
    (
        f"Min Sharpe Ratio: {min_sharpe_ratio:.4f}\n"
        f"{date_of_min_sharpe_ratio.date()}"
    ),
    xy=(
        date_of_min_sharpe_ratio,
        min_sharpe_ratio,
    ),
    xytext=(65, 30),
    textcoords="offset points",
    bbox={
        "boxstyle": "round",
        "fc": "white",
        "ec": "orange",
        "alpha": 0.9,
    },
    arrowprops={
        "arrowstyle": "->",
        "color": "black",
    },
)

# Mark current rolling Sharpe Ratio.
current_sharpe_ratio_date = (
    rolling_sharpe_ratio.index[-1]
)

ax.scatter(
    current_sharpe_ratio_date,
    current_sharpe_ratio,
    color="purple",
    zorder=5,
    label=(
        f"Current Sharpe Ratio: "
        f"{current_sharpe_ratio:.4f}"
    ),
)

ax.annotate(
    (
        f"Current Sharpe Ratio: {current_sharpe_ratio:.4f}\n"
        f"{current_sharpe_ratio_date.date()}"
    ),
    xy=(
        current_sharpe_ratio_date,
        current_sharpe_ratio,
    ),
    xytext=(-120, 195),
    textcoords="offset points",
    ha="left",
    bbox={
        "boxstyle": "round",
        "fc": "white",
        "ec": "purple",
        "alpha": 0.9,
    },
    arrowprops={
        "arrowstyle": "->",
        "color": "black",
        "relpos": (1.0, 0.0),
    },
)


# --------------------------------------------------
# Add historical market events
# --------------------------------------------------

for event_name, event_date in market_events.items():
    event_date = pd.Timestamp(event_date)

    ax.axvline(
        event_date,
        linestyle=":",
        linewidth=1,
        alpha=0.25,
    )

    # Keep each event label near the top of the chart.
    ax.text(
        event_date,
        0.92,
        event_name,
        transform=ax.get_xaxis_transform(),
        rotation=90,
        ha="right",
        va="top",
        fontsize=8,
    )


# --------------------------------------------------
# Format the chart
# --------------------------------------------------

# Add a small amount of space around the data so
# the annotations stay inside the chart.
ax.margins(
    x=0.04,
    y=0.08,
)

ax.grid(
    alpha=0.25
)

# Keep the legend outside the plot so it does not
# cover the Sharpe Ratio data or annotations.
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
)

plt.tight_layout()

plt.show()