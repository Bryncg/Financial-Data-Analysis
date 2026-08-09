import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import data as web


# --------------------------------------------------
# Download stock data
# --------------------------------------------------

# Download historical prices for the selected stocks.
data = yf.download(
    [
        "NVDA",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
    ],
    start="2015-01-01",
    end=None,
    auto_adjust=True,
    progress=False,
)


# --------------------------------------------------
# Treasury data
# --------------------------------------------------

# Download the 3-month Treasury yield from FRED.
treasury_data = web.DataReader(
    "DGS3MO",
    "fred",
    start="2015-01-01",
    end=None,
).dropna()

# Convert the Treasury yield from percentage to decimal.
treasury_data = treasury_data / 100

# Use the historical average as the annual risk-free rate.
historical_risk_free_rate = (
    treasury_data.mean().values[0]
)


# --------------------------------------------------
# Rename stock columns
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

# Keep the closing prices and apply the readable stock names.
closing_prices = (
    data["Close"]
    .rename(columns=stock_labels)
    .dropna()
)

# Calculate daily percentage returns.
daily_returns = (
    closing_prices
    .pct_change()
    .dropna()
)

daily_returns.index.name = None
daily_returns.columns.name = None


# --------------------------------------------------
# Annualised returns and covariance
# --------------------------------------------------

# Calculate the average daily return for each stock.
mean_daily_returns = (
    daily_returns.mean()
)

# Annualise the average daily returns.
annualized_returns = (
    mean_daily_returns * 252
)

# Calculate the daily covariance matrix.
cov_matrix_daily = (
    daily_returns.cov()
)

# Annualise the covariance matrix.
cov_matrix_annual = (
    cov_matrix_daily * 252
)


# --------------------------------------------------
# Monte Carlo settings
# --------------------------------------------------

# Number of simulated portfolios.
num_portfolios = 100000

# Fixed seed so the simulation can be reproduced.
np.random.seed(42)


# --------------------------------------------------
# Generate portfolio weights
# --------------------------------------------------

# Generate random long-only portfolio weights.
# The Dirichlet distribution makes each row sum to 1.
weights = np.random.dirichlet(
    np.ones(len(closing_prices.columns)),
    num_portfolios,
)

# Original method used before switching to Dirichlet.
# Kept here for reference.
#
# weights = np.random.random(
#     (num_portfolios, len(closing_prices.columns))
# )
#
# weights /= np.sum(
#     weights,
#     axis=1
# )[:, np.newaxis]


# --------------------------------------------------
# Verify generated weights
# --------------------------------------------------

weights_sum_to_one = np.allclose(
    np.sum(weights, axis=1),
    1,
)

no_negative_weights = np.all(
    weights >= 0
)

no_weights_above_one = np.all(
    weights <= 1
)


# --------------------------------------------------
# Portfolio return and volatility
# --------------------------------------------------

# Calculate the expected annual return for every portfolio.
portfolio_returns = np.dot(
    weights,
    annualized_returns,
)

# Calculate annualised volatility for every portfolio.
portfolio_volatility = np.sqrt(
    np.einsum(
        "ij,jk,ik->i",
        weights,
        cov_matrix_annual,
        weights,
    )
)


# --------------------------------------------------
# Sharpe Ratios
# --------------------------------------------------

# Calculate the Sharpe Ratio for every simulated portfolio.
sharpe_ratios = (
    portfolio_returns
    - historical_risk_free_rate
) / portfolio_volatility

# Check the Sharpe Ratio results.
sharpe_nan_count = (
    np.isnan(sharpe_ratios).sum()
)

sharpe_infinite_count = (
    np.isinf(sharpe_ratios).sum()
)


# --------------------------------------------------
# Simulation results DataFrame
# --------------------------------------------------

# Keep the simulation results numeric so they can
# still be used for calculations and plotting.
dataframe = pd.DataFrame({
    "Returns": portfolio_returns,
    "Volatility": portfolio_volatility,
    "Sharpe Ratio": sharpe_ratios,
    "Apple Weight": weights[:, 0],
    "Amazon Weight": weights[:, 1],
    "Alphabet Weight": weights[:, 2],
    "Microsoft Weight": weights[:, 3],
    "NVIDIA Weight": weights[:, 4],
})


# --------------------------------------------------
# Maximum Sharpe portfolio
# --------------------------------------------------

# Find the simulated portfolio with the highest Sharpe Ratio.
max_sharpe_idx = (
    sharpe_ratios.argmax()
)

max_sharpe_portfolio = (
    dataframe.iloc[max_sharpe_idx]
)


# --------------------------------------------------
# Minimum volatility portfolio
# --------------------------------------------------

# Find the simulated portfolio with the lowest volatility.
min_volatility_idx = (
    portfolio_volatility.argmin()
)

min_volatility_portfolio = (
    dataframe.iloc[min_volatility_idx]
)


# --------------------------------------------------
# Equal-weight benchmark
# --------------------------------------------------

# Create the original 20% equal-weight portfolio.
equal_weights = np.array(
    [
        1 / len(closing_prices.columns)
    ]
    * len(closing_prices.columns)
)

# Calculate its expected annual return.
equal_portfolio_return = np.dot(
    equal_weights,
    annualized_returns,
)

# Calculate its annualised volatility.
equal_portfolio_volatility = np.sqrt(
    np.dot(
        equal_weights.T,
        np.dot(
            cov_matrix_annual,
            equal_weights,
        ),
    )
)

# Calculate its Sharpe Ratio.
equal_sharpe_ratio = (
    equal_portfolio_return
    - historical_risk_free_rate
) / equal_portfolio_volatility


# --------------------------------------------------
# Portfolio comparison table
# --------------------------------------------------

# Compare the maximum Sharpe, minimum volatility
# and equal-weight portfolios.
comparison_table = pd.DataFrame({
    "Portfolio": [
        "Maximum Sharpe Ratio",
        "Minimum Volatility",
        "Equal Weight Portfolio",
    ],

    "Returns": [
        f"{max_sharpe_portfolio['Returns']:.2%}",
        f"{min_volatility_portfolio['Returns']:.2%}",
        f"{equal_portfolio_return:.2%}",
    ],

    "Volatility": [
        f"{max_sharpe_portfolio['Volatility']:.2%}",
        f"{min_volatility_portfolio['Volatility']:.2%}",
        f"{equal_portfolio_volatility:.2%}",
    ],

    "Sharpe Ratio": [
        f"{max_sharpe_portfolio['Sharpe Ratio']:.2f}",
        f"{min_volatility_portfolio['Sharpe Ratio']:.2f}",
        f"{equal_sharpe_ratio:.2f}",
    ],

    "Apple Weight": [
        f"{max_sharpe_portfolio['Apple Weight']:.2%}",
        f"{min_volatility_portfolio['Apple Weight']:.2%}",
        f"{equal_weights[0]:.2%}",
    ],

    "Amazon Weight": [
        f"{max_sharpe_portfolio['Amazon Weight']:.2%}",
        f"{min_volatility_portfolio['Amazon Weight']:.2%}",
        f"{equal_weights[1]:.2%}",
    ],

    "Alphabet Weight": [
        f"{max_sharpe_portfolio['Alphabet Weight']:.2%}",
        f"{min_volatility_portfolio['Alphabet Weight']:.2%}",
        f"{equal_weights[2]:.2%}",
    ],

    "Microsoft Weight": [
        f"{max_sharpe_portfolio['Microsoft Weight']:.2%}",
        f"{min_volatility_portfolio['Microsoft Weight']:.2%}",
        f"{equal_weights[3]:.2%}",
    ],

    "NVIDIA Weight": [
        f"{max_sharpe_portfolio['NVIDIA Weight']:.2%}",
        f"{min_volatility_portfolio['NVIDIA Weight']:.2%}",
        f"{equal_weights[4]:.2%}",
    ],
})


# --------------------------------------------------
# Verify selected portfolios
# --------------------------------------------------

# Rebuild the maximum Sharpe portfolio from its weights.
verified_max_weights = np.array([
    max_sharpe_portfolio["Apple Weight"],
    max_sharpe_portfolio["Amazon Weight"],
    max_sharpe_portfolio["Alphabet Weight"],
    max_sharpe_portfolio["Microsoft Weight"],
    max_sharpe_portfolio["NVIDIA Weight"],
])

# Rebuild the minimum volatility portfolio from its weights.
verified_min_weights = np.array([
    min_volatility_portfolio["Apple Weight"],
    min_volatility_portfolio["Amazon Weight"],
    min_volatility_portfolio["Alphabet Weight"],
    min_volatility_portfolio["Microsoft Weight"],
    min_volatility_portfolio["NVIDIA Weight"],
])


# --------------------------------------------------
# Validate maximum Sharpe portfolio
# --------------------------------------------------

# Recalculate its return from the saved weights.
verified_max_return = np.dot(
    verified_max_weights,
    annualized_returns,
)

# Recalculate its volatility from the covariance matrix.
verified_max_volatility = np.sqrt(
    np.dot(
        verified_max_weights.T,
        np.dot(
            cov_matrix_annual,
            verified_max_weights,
        ),
    )
)

# Recalculate its Sharpe Ratio.
verified_max_sharpe = (
    verified_max_return
    - historical_risk_free_rate
) / verified_max_volatility

max_return_match = np.isclose(
    verified_max_return,
    max_sharpe_portfolio["Returns"],
)

max_volatility_match = np.isclose(
    verified_max_volatility,
    max_sharpe_portfolio["Volatility"],
)

max_sharpe_match = np.isclose(
    verified_max_sharpe,
    max_sharpe_portfolio["Sharpe Ratio"],
)


# --------------------------------------------------
# Validate minimum volatility portfolio
# --------------------------------------------------

# Recalculate its return from the saved weights.
verified_min_return = np.dot(
    verified_min_weights,
    annualized_returns,
)

# Recalculate its volatility from the covariance matrix.
verified_min_volatility = np.sqrt(
    np.dot(
        verified_min_weights.T,
        np.dot(
            cov_matrix_annual,
            verified_min_weights,
        ),
    )
)

# Recalculate its Sharpe Ratio.
verified_min_sharpe = (
    verified_min_return
    - historical_risk_free_rate
) / verified_min_volatility

min_return_match = np.isclose(
    verified_min_return,
    min_volatility_portfolio["Returns"],
)

min_volatility_match = np.isclose(
    verified_min_volatility,
    min_volatility_portfolio["Volatility"],
)

min_sharpe_match = np.isclose(
    verified_min_sharpe,
    min_volatility_portfolio["Sharpe Ratio"],
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\nMonte Carlo Portfolio Simulation")
print("--------------------------------")

print(
    f"Simulated Portfolios: "
    f"{num_portfolios:,}"
)

print(
    f"Historical Average Risk-Free Rate: "
    f"{historical_risk_free_rate:.2%}"
)

print("\nSimulation Checks")

print(
    f"Weights Sum to 1: "
    f"{weights_sum_to_one}"
)

print(
    f"No Negative Weights: "
    f"{no_negative_weights}"
)

print(
    f"No Weights Above 100%: "
    f"{no_weights_above_one}"
)

print(
    f"Sharpe Ratio NaN Values: "
    f"{sharpe_nan_count}"
)

print(
    f"Sharpe Ratio Infinite Values: "
    f"{sharpe_infinite_count}"
)

print("\nPortfolio Comparison")

# index=False removes the 0, 1 and 2 from the left side.
print(
    comparison_table.to_string(
        index=False
    )
)

print("\nPortfolio Validation")

print(
    f"Maximum Sharpe Return Match: "
    f"{max_return_match}"
)

print(
    f"Maximum Sharpe Volatility Match: "
    f"{max_volatility_match}"
)

print(
    f"Maximum Sharpe Ratio Match: "
    f"{max_sharpe_match}"
)

print(
    f"Minimum Volatility Return Match: "
    f"{min_return_match}"
)

print(
    f"Minimum Volatility Match: "
    f"{min_volatility_match}"
)

print(
    f"Minimum Volatility Sharpe Match: "
    f"{min_sharpe_match}"
)


# --------------------------------------------------
# Plot Monte Carlo portfolio simulation
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(14, 8),
    constrained_layout=True,
)

# Plot all simulated portfolios.
scatter = ax.scatter(
    portfolio_volatility,
    portfolio_returns,
    c=sharpe_ratios,
    cmap="viridis",
    s=12,
    alpha=0.45,
)

# Show the Sharpe Ratio colour scale.
colorbar = fig.colorbar(
    scatter,
    ax=ax,
    pad=0.02,
)

colorbar.set_label(
    "Sharpe Ratio"
)


# --------------------------------------------------
# Highlight important portfolios
# --------------------------------------------------

# Maximum Sharpe Ratio portfolio.
ax.scatter(
    max_sharpe_portfolio["Volatility"],
    max_sharpe_portfolio["Returns"],
    marker="*",
    color="red",
    s=300,
    label="Maximum Sharpe Ratio",
    zorder=5,
)

# Minimum volatility portfolio.
ax.scatter(
    min_volatility_portfolio["Volatility"],
    min_volatility_portfolio["Returns"],
    marker="*",
    color="green",
    s=300,
    label="Minimum Volatility",
    zorder=5,
)

# Equal-weight benchmark portfolio.
ax.scatter(
    equal_portfolio_volatility,
    equal_portfolio_return,
    marker="*",
    color="blue",
    s=300,
    label="Equal Weight Portfolio",
    zorder=5,
)


# --------------------------------------------------
# Annotate important portfolios
# --------------------------------------------------

ax.annotate(
    (
        f"Maximum Sharpe\n"
        f"Return: {max_sharpe_portfolio['Returns']:.2%}\n"
        f"Volatility: {max_sharpe_portfolio['Volatility']:.2%}\n"
        f"Sharpe: {max_sharpe_portfolio['Sharpe Ratio']:.2f}"
    ),
    xy=(
        max_sharpe_portfolio["Volatility"],
        max_sharpe_portfolio["Returns"],
    ),
    xytext=(-40, 40),
    textcoords="offset points",
    bbox={
        "boxstyle": "round",
        "fc": "white",
        "alpha": 0.9,
    },
    arrowprops={
        "arrowstyle": "->",
    },
)

ax.annotate(
    (
        f"Minimum Volatility\n"
        f"Return: {min_volatility_portfolio['Returns']:.2%}\n"
        f"Volatility: {min_volatility_portfolio['Volatility']:.2%}\n"
        f"Sharpe: {min_volatility_portfolio['Sharpe Ratio']:.2f}"
    ),
    xy=(
        min_volatility_portfolio["Volatility"],
        min_volatility_portfolio["Returns"],
    ),
    xytext=(-35, 90),
    textcoords="offset points",
    bbox={
        "boxstyle": "round",
        "fc": "white",
        "alpha": 0.9,
    },
    arrowprops={
        "arrowstyle": "->",
    },
)

ax.annotate(
    (
        f"Equal Weight\n"
        f"Return: {equal_portfolio_return:.2%}\n"
        f"Volatility: {equal_portfolio_volatility:.2%}\n"
        f"Sharpe: {equal_sharpe_ratio:.2f}"
    ),
    xy=(
        equal_portfolio_volatility,
        equal_portfolio_return,
    ),
    xytext=(-55, 70),
    textcoords="offset points",
    bbox={
        "boxstyle": "round",
        "fc": "white",
        "alpha": 0.9,
    },
    arrowprops={
        "arrowstyle": "->",
    },
)


# --------------------------------------------------
# Format chart
# --------------------------------------------------

ax.set_title(
    "Monte Carlo Portfolio Simulation: Risk vs Expected Return",
    fontsize=16,
)

ax.set_xlabel(
    "Annualised Portfolio Volatility"
)

ax.set_ylabel(
    "Annualised Expected Return"
)

# Display return and volatility as percentages.
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda value, position: f"{value:.0%}"
    )
)

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda value, position: f"{value:.0%}"
    )
)

ax.grid(
    alpha=0.25
)

ax.legend(
    loc="upper left",
)

plt.show()