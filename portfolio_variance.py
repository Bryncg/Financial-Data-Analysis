import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
daily_returns.columns.name = None
daily_returns.index.name = None


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
if abs(weights.sum() - 1.0) > 1e-9:
    raise ValueError("Portfolio weights must sum to 1.0.")


# --------------------------------------------------
# Portfolio returns
# --------------------------------------------------

# Apply the equal weights to each day's stock returns.
portfolio_daily_returns = (
    daily_returns * weights
).sum(axis=1)

portfolio_daily_returns.name = (
    "Portfolio Daily Returns"
)

# Length of the analysis period in years.
years = (
    closing_prices.index[-1]
    - closing_prices.index[0]
).days / 365.25

# Compound the daily portfolio returns across the full period.
portfolio_growth = (
    1 + portfolio_daily_returns
).prod()

# Convert the total growth into an annualised return.
annualized_return = (
    portfolio_growth ** (1 / years)
) - 1


# --------------------------------------------------
# Covariance matrix
# --------------------------------------------------

# Calculate the covariance matrix from daily returns.
covariance_matrix = daily_returns.cov()

# Convert daily covariance into annualised covariance.
annualized_covariance_matrix = (
    covariance_matrix * 252
)

print("Daily Covariance Matrix:")
print(covariance_matrix.round(6))

print("\nAnnualised Covariance Matrix:")
print(annualized_covariance_matrix.round(4))


# --------------------------------------------------
# Portfolio variance
# --------------------------------------------------

# Method 1:
# Weight every row and column of the covariance matrix.
weighted_covariance_matrix = (
    annualized_covariance_matrix
    .mul(weights, axis=0)
    .mul(weights, axis=1)
)

portfolio_variance_1 = (
    weighted_covariance_matrix
    .sum()
    .sum()
)

# Method 2:
# Use the standard matrix equation wᵀΣw.
portfolio_variance_2 = (
    weights.T
    @ annualized_covariance_matrix
    @ weights
)

print("\nPortfolio Variance (Method 1):")
print(portfolio_variance_1)

print("\nPortfolio Variance (Method 2):")
print(portfolio_variance_2)


# --------------------------------------------------
# Portfolio volatility
# --------------------------------------------------

# Portfolio volatility is the square root of variance.
portfolio_volatility = np.sqrt(
    portfolio_variance_1
)

print("\nPortfolio Volatility:")
print(f"{portfolio_volatility:.2%}")


# --------------------------------------------------
# Marginal risk contribution
# --------------------------------------------------

# Estimate how portfolio volatility responds to a
# small increase in each stock's weight.
marginal_risk = (
    annualized_covariance_matrix
    @ weights
    / portfolio_volatility
)


# --------------------------------------------------
# Component risk contribution
# --------------------------------------------------

# Multiply marginal risk by each stock's portfolio weight.
component_risk_contributions = (
    weights * marginal_risk
)

print("\nMarginal Risk Contributions:")
print(marginal_risk.round(4))

print("\nComponent Risk Contributions:")
print(component_risk_contributions.round(4))

print(
    "\nComponent Risk Total:"
)

print(
    component_risk_contributions.sum()
)


# --------------------------------------------------
# Percentage risk contribution
# --------------------------------------------------

# Convert each component into a share of total
# portfolio volatility.
percentage_risk_contributions = (
    component_risk_contributions
    / portfolio_volatility
)

print("\nPercentage Risk Contributions:")
print(
    percentage_risk_contributions
    .map(lambda value: f"{value:.2%}")
)

print(
    "\nPercentage Risk Total:"
)

print(
    percentage_risk_contributions.sum()
)


# --------------------------------------------------
# Risk contribution table
# --------------------------------------------------

# Keep a numeric table for later calculations and plotting.
risk_contribution_numeric = pd.concat(
    [
        weights.rename("Portfolio Weight"),
        marginal_risk.rename(
            "Marginal Risk Contribution"
        ),
        component_risk_contributions.rename(
            "Component Risk Contribution"
        ),
        percentage_risk_contributions.rename(
            "Percentage Risk Contribution"
        ),
    ],
    axis=1,
)

# Create a formatted copy for terminal output.
risk_contribution_df = (
    risk_contribution_numeric.copy()
)

risk_contribution_df[
    "Portfolio Weight"
] = risk_contribution_df[
    "Portfolio Weight"
].map(lambda value: f"{value:.2%}")

risk_contribution_df[
    "Marginal Risk Contribution"
] = risk_contribution_df[
    "Marginal Risk Contribution"
].map(lambda value: f"{value:.4f}")

risk_contribution_df[
    "Component Risk Contribution"
] = risk_contribution_df[
    "Component Risk Contribution"
].map(lambda value: f"{value:.4f}")

risk_contribution_df[
    "Percentage Risk Contribution"
] = risk_contribution_df[
    "Percentage Risk Contribution"
].map(lambda value: f"{value:.2%}")

print("\nRisk Contribution Table:")
print(risk_contribution_df)


# --------------------------------------------------
# Portfolio risk summary
# --------------------------------------------------

highest_risk_stock = (
    percentage_risk_contributions.idxmax()
)

lowest_risk_stock = (
    percentage_risk_contributions.idxmin()
)

highest_risk_contribution = (
    percentage_risk_contributions.max()
)

lowest_risk_contribution = (
    percentage_risk_contributions.min()
)

equal_weight = 0.20

largest_overweight_risk = (
    highest_risk_contribution - equal_weight
)

largest_underweight_risk = (
    equal_weight - lowest_risk_contribution
)

summary_stats = {
    "Portfolio Variance": portfolio_variance_1,
    "Portfolio Volatility": portfolio_volatility,
    "Annualised Return": annualized_return,
    "Highest Risk Contributor": highest_risk_stock,
    "Highest Risk Contribution": highest_risk_contribution,
    "Lowest Risk Contributor": lowest_risk_stock,
    "Lowest Risk Contribution": lowest_risk_contribution,
    "Equal Weight": equal_weight,
    "Largest Overweight Risk": largest_overweight_risk,
    "Largest Underweight Risk": largest_underweight_risk,
}

print("\nPortfolio Risk Summary")

print(
    f"Portfolio Variance: "
    f"{summary_stats['Portfolio Variance']:.4f}"
)

print(
    f"Portfolio Volatility: "
    f"{summary_stats['Portfolio Volatility']:.2%}"
)

print(
    f"Annualised Return: "
    f"{summary_stats['Annualised Return']:.2%}"
)

print(
    f"Highest Risk Contributor: "
    f"{summary_stats['Highest Risk Contributor']} "
    f"({summary_stats['Highest Risk Contribution']:.2%})"
)

print(
    f"Lowest Risk Contributor: "
    f"{summary_stats['Lowest Risk Contributor']} "
    f"({summary_stats['Lowest Risk Contribution']:.2%})"
)

print(
    f"Equal Portfolio Weight: "
    f"{summary_stats['Equal Weight']:.2%}"
)

print(
    f"Largest Overweight Risk: "
    f"{summary_stats['Largest Overweight Risk']:.2%}"
)

print(
    f"Largest Underweight Risk: "
    f"{summary_stats['Largest Underweight Risk']:.2%}"
)


# --------------------------------------------------
# Sort risk contributions
# --------------------------------------------------

# Sort the stocks from highest to lowest risk contribution.
sorted_risk_contributions = (
    percentage_risk_contributions
    .sort_values(ascending=False)
)


# --------------------------------------------------
# Plot percentage risk contributions
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 6)
)

# Change the bar colour depending on whether risk
# contribution is above or below the 20% weight.
bar_colours = [
    "green" if value > equal_weight else "red"
    for value in sorted_risk_contributions
]

bars = ax.bar(
    sorted_risk_contributions.index,
    sorted_risk_contributions.values,
    color=bar_colours,
)

ax.set_title(
    "Percentage Risk Contributions by Stock"
)

ax.set_ylabel(
    "Percentage Risk Contribution"
)

ax.set_xlabel("Stock")

# Show the equal portfolio weight for comparison.
ax.axhline(
    y=equal_weight,
    color="red",
    linestyle="--",
    label="Equal Weight (20%)",
)

# Display each percentage above its bar.
for bar, value in zip(
    bars,
    sorted_risk_contributions.values,
):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.01,
        f"{value:.2%}",
        ha="center",
        va="bottom",
    )

ax.set_ylim(
    0,
    sorted_risk_contributions.max() + 0.08,
)

ax.legend()

plt.tight_layout()
plt.show()