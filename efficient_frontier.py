import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
from pandas_datareader import data as web


# --------------------------------------------------
# Download stock data
# --------------------------------------------------

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

treasury_data = web.DataReader(
    "DGS3MO",
    "fred",
    start="2015-01-01",
    end=None,
).dropna()

treasury_data = treasury_data / 100

historical_risk_free_rate = (
    treasury_data.mean().values[0]
)


# --------------------------------------------------
# Prepare stock data
# --------------------------------------------------

stock_labels = {
    "NVDA": "NVIDIA",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
}

closing_prices = (
    data["Close"]
    .rename(columns=stock_labels)
    .dropna()
)

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

mean_daily_returns = daily_returns.mean()

annualised_returns = (
    mean_daily_returns * 252
)

cov_daily_returns = daily_returns.cov()

annualised_covariance = (
    cov_daily_returns * 252
)


# --------------------------------------------------
# Equal-weight benchmark
# --------------------------------------------------

# Keep the same 20% equal-weight portfolio used in
# earlier projects as a benchmark for the optimisation.
equal_weights = np.array(
    [
        1 / len(closing_prices.columns)
    ]
    * len(closing_prices.columns)
)

equal_portfolio_return = np.dot(
    equal_weights,
    annualised_returns,
)

equal_portfolio_volatility = np.sqrt(
    np.dot(
        equal_weights.T,
        np.dot(
            annualised_covariance,
            equal_weights,
        ),
    )
)

equal_portfolio_sharpe_ratio = (
    equal_portfolio_return
    - historical_risk_free_rate
) / equal_portfolio_volatility


# --------------------------------------------------
# Portfolio functions
# --------------------------------------------------

# These functions are used by the optimiser.
# SciPy changes the weights and repeatedly passes them
# through these calculations while searching for a solution.
def portfolio_return(weights):
    return np.dot(
        weights,
        annualised_returns
    )


def portfolio_volatility(weights):
    return np.sqrt(
        np.dot(
            weights.T,
            np.dot(
                annualised_covariance,
                weights,
            ),
        )
    )


# Use the equal-weight portfolio to check that the
# reusable functions reproduce the calculations above.
assert np.isclose(
    portfolio_return(equal_weights),
    equal_portfolio_return
), "Portfolio return function test failed"

assert np.isclose(
    portfolio_volatility(equal_weights),
    equal_portfolio_volatility
), "Portfolio volatility function test failed"


# --------------------------------------------------
# Starting weights for optimisation
# --------------------------------------------------

# SciPy needs a valid starting point before it begins
# changing the portfolio weights.
initial_weights = equal_weights.copy()


# --------------------------------------------------
# Minimum-volatility optimisation function
# --------------------------------------------------

# This is the main function used to build the efficient frontier.
#
# Without a target return, it finds the lowest-volatility portfolio.
#
# With a target return, it finds the lowest-volatility portfolio
# that is also required to achieve that return.
def get_optimal_portfolio(
    initial_weights,
    target_return=None
):

    # The portfolio must remain fully invested,
    # so all weights must add to 100%.
    constraints = (
        {
            "type": "eq",
            "fun": lambda x:
                np.sum(x) - 1
        },
    )

    # When a target return is supplied, add another
    # equality constraint requiring that exact return.
    if target_return is not None:
        constraints += (
            {
                "type": "eq",
                "fun": lambda x:
                    portfolio_return(x)
                    - target_return
            },
        )

    # Long-only portfolio:
    # each stock can range from 0% to 100%.
    bounds = tuple(
        (0, 1)
        for _ in range(len(initial_weights))
    )

    # SLSQP is used because the problem has both
    # weight bounds and equality constraints.
    optimal_portfolio = sco.minimize(
        portfolio_volatility,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return optimal_portfolio


# --------------------------------------------------
# Minimum-volatility portfolio
# --------------------------------------------------

# No target return is supplied here.
# SciPy is therefore free to search purely for the
# lowest possible portfolio volatility.
min_vol_result = get_optimal_portfolio(
    initial_weights
)

if not min_vol_result.success:
    raise ValueError(
        "Minimum-volatility optimisation failed: "
        f"{min_vol_result.message}"
    )

min_vol_weights = min_vol_result.x

min_vol_return = (
    portfolio_return(min_vol_weights)
)

min_vol_volatility = (
    portfolio_volatility(min_vol_weights)
)

min_vol_sharpe_ratio = (
    min_vol_return
    - historical_risk_free_rate
) / min_vol_volatility


# --------------------------------------------------
# Validate minimum-volatility portfolio
# --------------------------------------------------

min_vol_weights_valid = (
    np.all(min_vol_weights >= 0)
    and np.all(min_vol_weights <= 1)
    and np.isclose(
        np.sum(min_vol_weights),
        1
    )
)

# .fun stores the value that SciPy minimised.
# For this optimisation it should match the volatility
# recalculated directly from the selected weights.
min_volatility_match = np.isclose(
    min_vol_result.fun,
    portfolio_volatility(min_vol_weights)
)


# --------------------------------------------------
# Efficient frontier target returns
# --------------------------------------------------

# The efficient branch begins at the global
# minimum-volatility portfolio.
#
# From there, gradually demand higher returns and find
# the lowest volatility available at each return level.
target_returns = np.linspace(
    min_vol_return,
    annualised_returns.max(),
    50
)

efficient_portfolio_returns = []
efficient_portfolio_volatilities = []


# --------------------------------------------------
# Build the efficient frontier
# --------------------------------------------------

# Each loop solves a new constrained optimisation problem.
# The return is fixed and SciPy finds the lowest volatility
# portfolio capable of reaching it.
for target_return in target_returns:

    result = get_optimal_portfolio(
        initial_weights,
        target_return=target_return
    )

    if result.success:

        efficient_portfolio_returns.append(
            portfolio_return(result.x)
        )

        efficient_portfolio_volatilities.append(
            portfolio_volatility(result.x)
        )

    else:

        print(
            f"Optimisation failed for target return "
            f"{target_return:.2%}: "
            f"{result.message}"
        )


# --------------------------------------------------
# Store efficient frontier results
# --------------------------------------------------

efficient_frontier_results = pd.DataFrame(
    {
        "Return": efficient_portfolio_returns,
        "Volatility": efficient_portfolio_volatilities,
    }
)

efficient_frontier_results = (
    efficient_frontier_results
    .sort_values(by="Return")
    .reset_index(drop=True)
)


# --------------------------------------------------
# Maximum-Sharpe portfolio
# --------------------------------------------------

# SciPy only minimises objective functions.
# Returning the negative Sharpe Ratio turns a
# maximum-Sharpe problem into a minimisation problem.
def negative_sharpe_ratio(weights):

    port_return = (
        portfolio_return(weights)
    )

    port_volatility = (
        portfolio_volatility(weights)
    )

    sharpe_ratio = (
        port_return
        - historical_risk_free_rate
    ) / port_volatility

    return -sharpe_ratio


max_sharpe_constraints = (
    {
        "type": "eq",
        "fun": lambda x:
            np.sum(x) - 1
    },
)

max_sharpe_bounds = tuple(
    (0, 1)
    for _ in range(len(initial_weights))
)

max_sharpe_result = sco.minimize(
    negative_sharpe_ratio,
    initial_weights,
    method="SLSQP",
    bounds=max_sharpe_bounds,
    constraints=max_sharpe_constraints,
)

if not max_sharpe_result.success:
    raise ValueError(
        "Maximum-Sharpe optimisation failed: "
        f"{max_sharpe_result.message}"
    )

max_sharpe_weights = max_sharpe_result.x

max_sharpe_return = (
    portfolio_return(max_sharpe_weights)
)

max_sharpe_volatility = (
    portfolio_volatility(max_sharpe_weights)
)

max_sharpe_ratio = (
    max_sharpe_return
    - historical_risk_free_rate
) / max_sharpe_volatility


# --------------------------------------------------
# Validate maximum-Sharpe portfolio
# --------------------------------------------------

max_sharpe_weights_valid = (
    np.all(max_sharpe_weights >= 0)
    and np.all(max_sharpe_weights <= 1)
    and np.isclose(
        np.sum(max_sharpe_weights),
        1
    )
)

# Because SciPy minimised negative Sharpe,
# .fun should match the negative value of the
# Sharpe Ratio recalculated from the selected weights.
max_sharpe_match = np.isclose(
    max_sharpe_result.fun,
    -max_sharpe_ratio
)


# --------------------------------------------------
# Monte Carlo comparison
# --------------------------------------------------

# Repeat the same 100,000-portfolio simulation from
# Project 10 using the same data and risk-free rate.
#
# This allows the sampled Monte Carlo solutions to be
# compared directly with the SciPy optimised solutions.
num_portfolios = 100_000

np.random.seed(42)

monte_carlo_weights = np.random.dirichlet(
    np.ones(len(closing_prices.columns)),
    num_portfolios
)

monte_carlo_returns = np.dot(
    monte_carlo_weights,
    annualised_returns,
)

monte_carlo_volatilities = np.sqrt(
    np.einsum(
        "ij,jk,ik->i",
        monte_carlo_weights,
        annualised_covariance,
        monte_carlo_weights
    )
)

monte_carlo_sharpe_ratios = (
    monte_carlo_returns
    - historical_risk_free_rate
) / monte_carlo_volatilities


# --------------------------------------------------
# Select Monte Carlo benchmark portfolios
# --------------------------------------------------

monte_carlo_max_sharpe_idx = np.argmax(
    monte_carlo_sharpe_ratios
)

monte_carlo_min_vol_idx = np.argmin(
    monte_carlo_volatilities
)


# Maximum-Sharpe Monte Carlo portfolio.
monte_carlo_max_sharpe_weights = (
    monte_carlo_weights[
        monte_carlo_max_sharpe_idx
    ]
)

monte_carlo_max_sharpe_return = (
    monte_carlo_returns[
        monte_carlo_max_sharpe_idx
    ]
)

monte_carlo_max_sharpe_volatility = (
    monte_carlo_volatilities[
        monte_carlo_max_sharpe_idx
    ]
)

monte_carlo_max_sharpe_ratio = (
    monte_carlo_sharpe_ratios[
        monte_carlo_max_sharpe_idx
    ]
)


# Minimum-volatility Monte Carlo portfolio.
monte_carlo_min_vol_weights = (
    monte_carlo_weights[
        monte_carlo_min_vol_idx
    ]
)

monte_carlo_min_vol_return = (
    monte_carlo_returns[
        monte_carlo_min_vol_idx
    ]
)

monte_carlo_min_vol_volatility = (
    monte_carlo_volatilities[
        monte_carlo_min_vol_idx
    ]
)

monte_carlo_min_vol_sharpe_ratio = (
    monte_carlo_sharpe_ratios[
        monte_carlo_min_vol_idx
    ]
)


# --------------------------------------------------
# Monte Carlo vs optimisation comparison
# --------------------------------------------------

comparison_table = pd.DataFrame({
    "Portfolio": [
        "Monte Carlo Maximum-Sharpe",
        "Optimised Maximum-Sharpe",
        "Monte Carlo Minimum-Volatility",
        "Optimised Minimum-Volatility",
        "Equal Weight",
    ],
    "Return": [
        monte_carlo_max_sharpe_return,
        max_sharpe_return,
        monte_carlo_min_vol_return,
        min_vol_return,
        equal_portfolio_return,
    ],
    "Volatility": [
        monte_carlo_max_sharpe_volatility,
        max_sharpe_volatility,
        monte_carlo_min_vol_volatility,
        min_vol_volatility,
        equal_portfolio_volatility,
    ],
    "Sharpe Ratio": [
        monte_carlo_max_sharpe_ratio,
        max_sharpe_ratio,
        monte_carlo_min_vol_sharpe_ratio,
        min_vol_sharpe_ratio,
        equal_portfolio_sharpe_ratio,
    ],
})


# Add the stock weights so the difference between
# the sampled and optimised portfolios can also be compared.
comparison_weights = np.vstack([
    monte_carlo_max_sharpe_weights,
    max_sharpe_weights,
    monte_carlo_min_vol_weights,
    min_vol_weights,
    equal_weights,
])

weight_columns = pd.DataFrame(
    comparison_weights,
    columns=[
        "Apple Weight",
        "Amazon Weight",
        "Alphabet Weight",
        "Microsoft Weight",
        "NVIDIA Weight",
    ]
)

comparison_table = pd.concat(
    [
        comparison_table,
        weight_columns,
    ],
    axis=1,
)


# Compare the actual optimisation objectives rather
# than simply comparing portfolio returns.
max_sharpe_improvement = (
    max_sharpe_ratio
    - monte_carlo_max_sharpe_ratio
)

min_vol_improvement = (
    monte_carlo_min_vol_volatility
    - min_vol_volatility
)


# --------------------------------------------------
# Format comparison table for terminal output
# --------------------------------------------------

# Keep comparison_table numerical for calculations.
# Only convert a copy into percentage strings for printing.
comparison_table_display = comparison_table.copy()

for column in [
    "Return",
    "Volatility",
    "Apple Weight",
    "Amazon Weight",
    "Alphabet Weight",
    "Microsoft Weight",
    "NVIDIA Weight",
]:
    comparison_table_display[column] = (
        comparison_table_display[column]
        .map(lambda x: f"{x:.2%}")
    )

comparison_table_display["Sharpe Ratio"] = (
    comparison_table_display["Sharpe Ratio"]
    .map(lambda x: f"{x:.4f}")
)


# --------------------------------------------------
# Final efficient frontier validation
# --------------------------------------------------

# Final checks on the frontier itself rather than on
# one individual optimised portfolio.
frontier_has_nan = (
    efficient_frontier_results
    .isna()
    .any()
    .any()
)

frontier_has_infinite = np.isinf(
    efficient_frontier_results[
        ["Return", "Volatility"]
    ].to_numpy()
).any()

frontier_returns_increasing = (
    efficient_frontier_results[
        "Return"
    ].is_monotonic_increasing
)

frontier_volatilities_increasing = (
    efficient_frontier_results[
        "Volatility"
    ].is_monotonic_increasing
)

frontier_point_count_valid = (
    len(efficient_frontier_results)
    == len(target_returns)
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\nPortfolio Optimisation Summary")

print("\nEqual-Weight Portfolio")
print(
    f"Return: {equal_portfolio_return:.2%}"
)
print(
    f"Volatility: {equal_portfolio_volatility:.2%}"
)
print(
    f"Sharpe Ratio: "
    f"{equal_portfolio_sharpe_ratio:.4f}"
)


print("\nMinimum-Volatility Portfolio")
print(
    f"Return: {min_vol_return:.2%}"
)
print(
    f"Volatility: {min_vol_volatility:.2%}"
)
print(
    f"Sharpe Ratio: {min_vol_sharpe_ratio:.4f}"
)

print("\nMinimum-Volatility Weights")

for stock, weight in zip(
    annualised_returns.index,
    min_vol_weights
):
    print(
        f"{stock}: {weight:.2%}"
    )


print("\nMaximum-Sharpe Portfolio")
print(
    f"Return: {max_sharpe_return:.2%}"
)
print(
    f"Volatility: {max_sharpe_volatility:.2%}"
)
print(
    f"Sharpe Ratio: {max_sharpe_ratio:.4f}"
)

print("\nMaximum-Sharpe Weights")

for stock, weight in zip(
    annualised_returns.index,
    max_sharpe_weights
):
    print(
        f"{stock}: {weight:.2%}"
    )


print("\nMonte Carlo vs Optimisation Comparison")
print(
    comparison_table_display.to_string(
        index=False
    )
)

print(
    "\nMaximum-Sharpe Improvement:",
    f"{max_sharpe_improvement:.6f}"
)

print(
    "Minimum-Volatility Improvement:",
    f"{min_vol_improvement:.6%}"
)


print("\nOptimisation Validation")

print(
    "Minimum-Volatility Optimisation:",
    min_vol_result.success
)

print(
    "Minimum-Volatility Weights Valid:",
    min_vol_weights_valid
)

print(
    "Minimum-Volatility Recalculation:",
    min_volatility_match
)

print(
    "Maximum-Sharpe Optimisation:",
    max_sharpe_result.success
)

print(
    "Maximum-Sharpe Weights Valid:",
    max_sharpe_weights_valid
)

print(
    "Maximum-Sharpe Recalculation:",
    max_sharpe_match
)


print("\nEfficient Frontier Validation")

print(
    "No NaN Values:",
    not frontier_has_nan
)

print(
    "No Infinite Values:",
    not frontier_has_infinite
)

print(
    "Returns Increasing:",
    frontier_returns_increasing
)

print(
    "Volatility Increasing:",
    frontier_volatilities_increasing
)

print(
    "All Frontier Points Generated:",
    frontier_point_count_valid
)


# --------------------------------------------------
# Efficient frontier plot
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(14, 8),
    constrained_layout=True,
)


# Plot the Monte Carlo portfolios in the background.
# The cloud shows the portfolio combinations sampled
# randomly, while the efficient frontier shows the
# optimised boundary of those possible portfolios.
scatter = ax.scatter(
    monte_carlo_volatilities,
    monte_carlo_returns,
    c=monte_carlo_sharpe_ratios,
    cmap="viridis",
    s=8,
    alpha=0.20,
    zorder=1,
)

colorbar = fig.colorbar(
    scatter,
    ax=ax,
    pad=0.02,
)

colorbar.set_label(
    "Sharpe Ratio"
)


# --------------------------------------------------
# Efficient frontier
# --------------------------------------------------

ax.plot(
    efficient_frontier_results["Volatility"],
    efficient_frontier_results["Return"],
    "b--",
    label="Efficient Frontier",
    zorder=4,
)


# --------------------------------------------------
# Highlight optimised portfolios
# --------------------------------------------------

ax.scatter(
    max_sharpe_volatility,
    max_sharpe_return,
    marker="*",
    color="red",
    s=300,
    label="Maximum Sharpe Ratio",
    zorder=6,
)

ax.scatter(
    min_vol_volatility,
    min_vol_return,
    marker="*",
    color="green",
    s=300,
    label="Minimum Volatility",
    zorder=6,
)

ax.scatter(
    equal_portfolio_volatility,
    equal_portfolio_return,
    marker="*",
    color="blue",
    s=300,
    label="Equal Weight Portfolio",
    zorder=6,
)


# --------------------------------------------------
# Highlight Monte Carlo comparison portfolios
# --------------------------------------------------

# Circles are used for Monte Carlo results and stars
# for direct optimisation so the two methods are easy to compare.
ax.scatter(
    monte_carlo_max_sharpe_volatility,
    monte_carlo_max_sharpe_return,
    marker="o",
    color="red",
    s=100,
    label="Monte Carlo Max Sharpe",
    zorder=5,
)

ax.scatter(
    monte_carlo_min_vol_volatility,
    monte_carlo_min_vol_return,
    marker="o",
    color="green",
    s=100,
    label="Monte Carlo Min Volatility",
    zorder=5,
)


# --------------------------------------------------
# Annotate important portfolios
# --------------------------------------------------

ax.annotate(
    (
        f"Maximum Sharpe\n"
        f"Return: {max_sharpe_return:.2%}\n"
        f"Volatility: {max_sharpe_volatility:.2%}\n"
        f"Sharpe: {max_sharpe_ratio:.2f}"
    ),
    xy=(
        max_sharpe_volatility,
        max_sharpe_return,
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
        f"Return: {min_vol_return:.2%}\n"
        f"Volatility: {min_vol_volatility:.2%}\n"
        f"Sharpe: {min_vol_sharpe_ratio:.2f}"
    ),
    xy=(
        min_vol_volatility,
        min_vol_return,
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
        f"Sharpe: {equal_portfolio_sharpe_ratio:.2f}"
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

ax.annotate(
    (
        f"Monte Carlo Max Sharpe\n"
        f"Return: {monte_carlo_max_sharpe_return:.2%}\n"
        f"Volatility: {monte_carlo_max_sharpe_volatility:.2%}\n"
        f"Sharpe: {monte_carlo_max_sharpe_ratio:.2f}"
    ),
    xy=(
        monte_carlo_max_sharpe_volatility,
        monte_carlo_max_sharpe_return,
    ),
    xytext=(50, -30),
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
        f"Monte Carlo Min Volatility\n"
        f"Return: {monte_carlo_min_vol_return:.2%}\n"
        f"Volatility: {monte_carlo_min_vol_volatility:.2%}\n"
        f"Sharpe: {monte_carlo_min_vol_sharpe_ratio:.2f}"
    ),
    xy=(
        monte_carlo_min_vol_volatility,
        monte_carlo_min_vol_return,
    ),
    xytext=(50, -30),
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
# Plot formatting
# --------------------------------------------------

ax.set_title(
    "Efficient Frontier: Portfolio Risk vs Expected Return"
)

ax.set_xlabel(
    "Volatility (Standard Deviation)"
)

ax.set_ylabel(
    "Expected Return"
)

ax.xaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda x, _: f"{x:.0%}"
    )
)

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda y, _: f"{y:.0%}"
    )
)

ax.grid(
    alpha=0.25
)

ax.legend()

plt.show()