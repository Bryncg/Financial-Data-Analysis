import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Download stock data
# --------------------------------------------------

# Download daily data for the selected stocks.
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


# --------------------------------------------------
# Rename stock tickers
# --------------------------------------------------

# Rename the tickers so they are easier to read.
stock_labels = {
    "NVDA": "NVIDIA",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
}


# --------------------------------------------------
# Closing prices
# --------------------------------------------------

# Keep only the adjusted closing prices.
closing_prices = (
    data["Close"]
    .rename(columns=stock_labels)
    .dropna()
)

# Uncomment to inspect the downloaded prices.
# print(closing_prices)


# --------------------------------------------------
# Daily returns
# --------------------------------------------------

# Convert closing prices into daily percentage returns.
daily_returns = closing_prices.pct_change().dropna()

# Remove the default DataFrame axis names.
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
if weights.isna().any():
    raise ValueError("A portfolio weight is missing.")

# Check that the portfolio is fully invested.
if abs(weights.sum() - 1.0) > 1e-6:
    raise ValueError("Portfolio weights must sum to 1.")


# --------------------------------------------------
# Rolling window settings
# --------------------------------------------------

# Number of trading days included in each rolling window.
rolling_window = 60


# --------------------------------------------------
# Rolling covariance calculation
# --------------------------------------------------

# Calculate a new covariance matrix for every date using
# the previous 60 trading days.
rolling_covariance = (
    daily_returns
    .rolling(window=rolling_window)
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


# --------------------------------------------------
# Example rolling covariance matrices
# --------------------------------------------------

# These checks were used to confirm that complete covariance
# matrices could be extracted from the rolling MultiIndex.

# first_date = rolling_dates[0]
# middle_date = rolling_dates[len(rolling_dates) // 2]
# last_date = rolling_dates[-1]

# first_rolling_covariance = (
#     rolling_covariance.loc[first_date]
# )

# middle_rolling_covariance = (
#     rolling_covariance.loc[middle_date]
# )

# last_rolling_covariance = (
#     rolling_covariance.loc[last_date]
# )

# print(
#     f"\nFirst {rolling_window}-Day "
#     f"Rolling Covariance Matrix"
# )

# print(
#     f"Window Ending: "
#     f"{first_date.strftime('%Y-%m-%d')}"
# )

# print(first_rolling_covariance.round(6))

# print(
#     f"\nMiddle {rolling_window}-Day "
#     f"Rolling Covariance Matrix"
# )

# print(
#     f"Window Ending: "
#     f"{middle_date.strftime('%Y-%m-%d')}"
# )

# print(middle_rolling_covariance.round(6))

# print(
#     f"\nLast {rolling_window}-Day "
#     f"Rolling Covariance Matrix"
# )

# print(
#     f"Window Ending: "
#     f"{last_date.strftime('%Y-%m-%d')}"
# )

# print(last_rolling_covariance.round(6))


# --------------------------------------------------
# Latest-window verification
# --------------------------------------------------

# This calculation was used to test the portfolio variance
# formula on one rolling covariance matrix before building
# the complete rolling time series.

# annualised_last_covariance = (
#     last_rolling_covariance * 252
# )

# last_portfolio_variance = (
#     weights.T
#     @ annualised_last_covariance
#     @ weights
# )

# last_volatility = np.sqrt(
#     last_portfolio_variance
# )

# print("\nLatest Portfolio Variance")
# print(last_portfolio_variance)

# print("\nLatest Portfolio Volatility")
# print(f"{last_volatility:.2%}")


# --------------------------------------------------
# Annualised rolling covariance
# --------------------------------------------------

# Convert each daily rolling covariance matrix into
# annualised covariance.
annualised_rolling_covariance = (
    rolling_covariance * 252
)


# --------------------------------------------------
# Rolling portfolio risk
# --------------------------------------------------

rolling_portfolio_variance = []
rolling_portfolio_volatility = []

# Calculate portfolio variance and volatility for every
# available rolling date.
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
# Convert rolling results into dated Series
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

# These checks were used to confirm that the first and last
# rolling values were stored correctly.

# print("\nFirst Rolling Variance Values")
# print(rolling_portfolio_variance.head())

# print("\nLast Rolling Variance Values")
# print(rolling_portfolio_variance.tail())

# print("\nFirst Rolling Volatility Values")
# print(
#     rolling_portfolio_volatility
#     .head()
#     .apply(lambda value: f"{value:.2%}")
# )

# print("\nLast Rolling Volatility Values")
# print(
#     rolling_portfolio_volatility
#     .tail()
#     .apply(lambda value: f"{value:.2%}")
# )


# --------------------------------------------------
# Rolling volatility statistics
# --------------------------------------------------

current_rolling_volatility = (
    rolling_portfolio_volatility.iloc[-1]
)

average_rolling_volatility = (
    rolling_portfolio_volatility.mean()
)

max_rolling_volatility = (
    rolling_portfolio_volatility.max()
)

max_rolling_volatility_date = (
    rolling_portfolio_volatility.idxmax()
)

min_rolling_volatility = (
    rolling_portfolio_volatility.min()
)

min_rolling_volatility_date = (
    rolling_portfolio_volatility.idxmin()
)


# --------------------------------------------------
# Rolling variance statistics
# --------------------------------------------------

current_rolling_variance = (
    rolling_portfolio_variance.iloc[-1]
)

average_rolling_variance = (
    rolling_portfolio_variance.mean()
)

max_rolling_variance = (
    rolling_portfolio_variance.max()
)

max_rolling_variance_date = (
    rolling_portfolio_variance.idxmax()
)

min_rolling_variance = (
    rolling_portfolio_variance.min()
)

min_rolling_variance_date = (
    rolling_portfolio_variance.idxmin()
)


# --------------------------------------------------
# Display summary statistics
# --------------------------------------------------

print("\nRolling Portfolio Volatility Summary")

print(
    f"Current Rolling Volatility: "
    f"{current_rolling_volatility:.2%}"
)

print(
    f"Average Rolling Volatility: "
    f"{average_rolling_volatility:.2%}"
)

print(
    f"Maximum Rolling Volatility: "
    f"{max_rolling_volatility:.2%}"
)

print(
    f"Date of Maximum Volatility: "
    f"{max_rolling_volatility_date:%Y-%m-%d}"
)

print(
    f"Minimum Rolling Volatility: "
    f"{min_rolling_volatility:.2%}"
)

print(
    f"Date of Minimum Volatility: "
    f"{min_rolling_volatility_date:%Y-%m-%d}"
)


print("\nRolling Portfolio Variance Summary")

print(
    f"Current Rolling Variance: "
    f"{current_rolling_variance:.6f}"
)

print(
    f"Average Rolling Variance: "
    f"{average_rolling_variance:.6f}"
)

print(
    f"Maximum Rolling Variance: "
    f"{max_rolling_variance:.6f}"
)

print(
    f"Date of Maximum Variance: "
    f"{max_rolling_variance_date:%Y-%m-%d}"
)

print(
    f"Minimum Rolling Variance: "
    f"{min_rolling_variance:.6f}"
)

print(
    f"Date of Minimum Variance: "
    f"{min_rolling_variance_date:%Y-%m-%d}"
)


# --------------------------------------------------
# Historical market events
# --------------------------------------------------

# These dates are manually selected reference points used
# to give historical context to major changes in volatility.
market_events = {
    "COVID-19 Crash": "2020-03-16",
    "COVID-19 Recovery": "2020-08-18",
    "2022 Market Correction": "2022-06-16",
    "2022 Inflation and Rate Hikes": "2022-10-13",
    "2024 Market Recovery": "2024-06-14",
    "Regional Banking Crisis": "2023-03-10",
    "DeepSeek Tech Sell-off": "2025-01-27",
    "Tariff Market Crash": "2025-04-03",
}


# --------------------------------------------------
# Plot rolling portfolio volatility
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(14, 7)
)

# Plot the rolling annualised portfolio volatility.
ax.plot(
    rolling_portfolio_volatility.index,
    rolling_portfolio_volatility,
    label="Rolling Portfolio Volatility",
    color="blue",
)

ax.set_title(
    f"{rolling_window}-Day Rolling Portfolio Volatility"
)

ax.set_xlabel("Date")

ax.set_ylabel(
    "Annualised Rolling Portfolio Volatility"
)

# Show average volatility across the full period.
ax.axhline(
    y=average_rolling_volatility,
    color="orange",
    linestyle="--",
    label=(
        f"Average Volatility: "
        f"{average_rolling_volatility:.2%}"
    ),
)


# --------------------------------------------------
# Mark important volatility values
# --------------------------------------------------

# Mark maximum rolling volatility.
ax.scatter(
    max_rolling_volatility_date,
    max_rolling_volatility,
    color="red",
    zorder=5,
    label=(
        f"Max Volatility: "
        f"{max_rolling_volatility:.2%} on "
        f"{max_rolling_volatility_date:%Y-%m-%d}"
    ),
)

ax.annotate(
    (
        f"Maximum: {max_rolling_volatility:.2%}\n"
        f"{max_rolling_volatility_date:%Y-%m-%d}"
    ),
    xy=(
        max_rolling_volatility_date,
        max_rolling_volatility,
    ),
    xytext=(80, -25),
    textcoords="offset points",
    bbox={
        "boxstyle": "round",
        "fc": "white",
        "ec": "red",
        "alpha": 0.9,
    },
    arrowprops={
        "arrowstyle": "->",
        "color": "black",
    },
)

# Mark minimum rolling volatility.
ax.scatter(
    min_rolling_volatility_date,
    min_rolling_volatility,
    color="green",
    zorder=5,
    label=(
        f"Min Volatility: "
        f"{min_rolling_volatility:.2%} on "
        f"{min_rolling_volatility_date:%Y-%m-%d}"
    ),
)

ax.annotate(
    (
        f"Minimum: {min_rolling_volatility:.2%}\n"
        f"{min_rolling_volatility_date:%Y-%m-%d}"
    ),
    xy=(
        min_rolling_volatility_date,
        min_rolling_volatility,
    ),
    xytext=(40, 15),
    textcoords="offset points",
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

# Mark current rolling volatility.
current_rolling_volatility_date = (
    rolling_portfolio_volatility.index[-1]
)

ax.scatter(
    current_rolling_volatility_date,
    current_rolling_volatility,
    color="purple",
    zorder=5,
    label=(
        f"Current Volatility: "
        f"{current_rolling_volatility:.2%} on "
        f"{current_rolling_volatility_date:%Y-%m-%d}"
    ),
)

ax.annotate(
    (
        f"Current: {current_rolling_volatility:.2%}\n"
        f"{current_rolling_volatility_date:%Y-%m-%d}"
    ),
    xy=(
        current_rolling_volatility_date,
        current_rolling_volatility,
    ),
    xytext=(-85, 40),
    textcoords="offset points",
    bbox={
        "boxstyle": "round",
        "fc": "white",
        "ec": "purple",
        "alpha": 0.9,
    },
    arrowprops={
        "arrowstyle": "->",
        "color": "black",
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
        0.94,
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

# Display volatility values as percentages.
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda value, position: f"{value:.0%}"
    )
)

ax.grid(alpha=0.25)

ax.legend(
    loc="upper right"
)

plt.tight_layout()

plt.show()