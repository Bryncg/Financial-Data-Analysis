import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
from pandas_datareader import data as web
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import FuncFormatter

#download stock data
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

#rename stock columns
stock_labels = {
    "NVDA": "NVIDIA",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
}

#download treasury data
treasury_data = web.DataReader(
    "DGS3MO",
    "fred",
    start="2015-01-01",
    end=None,
).dropna()

#estimation and out of sample split date
split_date = "2024-01-01"

#convert to decimal
treasury_data = treasury_data / 100

#estimation and test risk free rates
estimation_treasury = treasury_data[
    treasury_data.index < split_date
]
out_of_sample_treasury = treasury_data[
    treasury_data.index >= split_date
]

estimation_risk_free_rate = estimation_treasury.mean().values[0]

test_risk_free_rate = out_of_sample_treasury.mean().values[0]



#closing stock prices
closing_prices = (
    data["Close"]
    .rename(columns=stock_labels)
    .dropna()
)

#daily percentage returns
daily_returns = (
    closing_prices
    .pct_change()
    .dropna()
)

daily_returns.index.name = None
daily_returns.columns.name = None


estimation_returns = daily_returns[
    daily_returns.index < split_date
]

out_of_sample_returns = daily_returns[
    daily_returns.index >= split_date
]


#mean daily returns
mean_daily_returns = estimation_returns.mean()
#annualised expected returns
annualised_returns = mean_daily_returns * 252

#daily covariance matrix
daily_covariance = estimation_returns.cov()
#annualised covariance matrix
annualised_covariance = daily_covariance * 252

#portfolio calculations

#portfolio return and volatility functions

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
                weights
            )
        )
    )

#equal weight portfolio
equal_weights = np.array([1 / len(closing_prices.columns)] * len(closing_prices.columns))

equal_portfolio_return = np.dot(equal_weights, annualised_returns)
equal_portfolio_volatility = np.sqrt(
    np.dot(
        equal_weights.T,
        np.dot(
            annualised_covariance,
            equal_weights
        )
    )
)

equal_portfolio_sharpe_ratio = (
    (equal_portfolio_return - estimation_risk_free_rate)
    / equal_portfolio_volatility
)

#use equal weights to check the functions
assert np.isclose(
    portfolio_return(equal_weights),
    equal_portfolio_return
), "Portfolio return function test failed."

assert np.isclose(
    portfolio_volatility(equal_weights),
    equal_portfolio_volatility
), "Portfolio volatility function test failed."



#minimum volatility portfolio
initial_weights = equal_weights.copy()

def get_optimal_portfolio(
        initial_weights,
):
    constraints = (
        {
            "type": "eq",
            "fun": lambda x:
                np.sum(x) - 1,
        },
    )

    bounds = tuple(
        (0, 1) for _ in range(len(initial_weights))
    )

    optimal_portfolio = sco.minimize(
        portfolio_volatility,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return optimal_portfolio

min_vol_result = get_optimal_portfolio(
    initial_weights
)

if not min_vol_result.success:
    raise ValueError(
        "Minimum volatility portfolio optimisation failed."
    )

min_vol_weights = min_vol_result.x

min_vol_return = portfolio_return(min_vol_weights)
min_vol_volatility = portfolio_volatility(min_vol_weights)

min_vol_sharpe_ratio = (
    (min_vol_return - estimation_risk_free_rate)
    / min_vol_volatility
)


#validate minimum volatility portfolio

min_vol_weights_valid = (
    np.all(min_vol_weights >= 0)
    and np.all(min_vol_weights <= 1)
    and np.isclose(
        np.sum(min_vol_weights),
        1
    )
)

min_volatility_match = np.isclose(
    min_vol_result.fun,
    portfolio_volatility(min_vol_weights)
)

assert min_vol_weights_valid, "Minimum volatility portfolio weights are invalid."
assert min_volatility_match, "Minimum volatility portfolio volatility does not match the calculated value."





#max sharpe ratio portfolio
def negative_sharpe_ratio(weights):

    port_return = (
        portfolio_return(weights)
    )

    port_volatility = (
        portfolio_volatility(weights)
    )

    sharpe_ratio = (
        port_return
        - estimation_risk_free_rate
    ) / port_volatility

    return -sharpe_ratio

max_sharpe_constraints = (
    {
        "type": "eq",
        "fun": lambda x:
            np.sum(x) - 1,
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
        "Maximum Sharpe optimisation failed. "
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
    - estimation_risk_free_rate
) / max_sharpe_volatility

#validate maximum sharpe portfolio

max_sharpe_weights_valid = (
    np.all(max_sharpe_weights >= 0)
    and np.all(max_sharpe_weights <= 1)
    and np.isclose(
        np.sum(max_sharpe_weights),
        1
    )
)

max_sharpe_match = np.isclose(
    max_sharpe_result.fun,
    -max_sharpe_ratio
)

assert max_sharpe_weights_valid, "Maximum Sharpe ratio portfolio weights are invalid."
assert max_sharpe_match, "Maximum Sharpe ratio portfolio value does not match the calculated value."

#print portfolio weights with stock names
print("Minimum Volatility Portfolio:")
for stock, weight in zip(closing_prices.columns, min_vol_weights):
    print(f"{stock}: {weight:.2%}")

print("\nMaximum Sharpe Ratio Portfolio:")
for stock, weight in zip(closing_prices.columns, max_sharpe_weights):
    print(f"{stock}: {weight:.2%}")

print("\nEqual Weight Portfolio:")
for stock, weight in zip(closing_prices.columns, equal_weights):
    print(f"{stock}: {weight:.2%}")

#estimation period performance table
estimation_performance = pd.DataFrame({
    "portfolio": ["Minimum Volatility", "Maximum Sharpe Ratio", "Equal Weight"],
    "return": [min_vol_return, max_sharpe_return, equal_portfolio_return],
    "volatility": [min_vol_volatility, max_sharpe_volatility, equal_portfolio_volatility],
    "sharpe_ratio": [min_vol_sharpe_ratio, max_sharpe_ratio, equal_portfolio_sharpe_ratio],
})

#print estimation period performance
print("\nPortfolio Performance Comparison:")
print(estimation_performance.to_string(index=False))

#out of sample performance
out_of_sample_mean_daily_returns = out_of_sample_returns.mean()
out_of_sample_annualised_returns = out_of_sample_mean_daily_returns * 252

out_of_sample_daily_covariance = out_of_sample_returns.cov()
out_of_sample_annualised_covariance = out_of_sample_daily_covariance * 252

#reusable function for out of sample performance
def calculate_out_of_sample_performance(weights):
    port_return = np.dot(weights, out_of_sample_annualised_returns)
    port_volatility = np.sqrt(
        np.dot(
            weights.T,
            np.dot(
                out_of_sample_annualised_covariance,
                weights
            )
        )
    )
    port_sharpe_ratio = (
        (port_return - test_risk_free_rate)
        / port_volatility
    )

    return port_return, port_volatility, port_sharpe_ratio

#minimum volatility portfolio out of sample performance
(
    min_vol_out_of_sample_return,
    min_vol_out_of_sample_volatility,
    min_vol_out_of_sample_sharpe_ratio,
) = calculate_out_of_sample_performance(min_vol_weights)

#maximum sharpe ratio portfolio out of sample performance
(
    max_sharpe_out_of_sample_return,
    max_sharpe_out_of_sample_volatility,
    max_sharpe_out_of_sample_sharpe_ratio,
) = calculate_out_of_sample_performance(max_sharpe_weights)

#equal weight portfolio out of sample performance
(
    equal_out_of_sample_return,
    equal_out_of_sample_volatility,
    equal_out_of_sample_sharpe_ratio,
) = calculate_out_of_sample_performance(equal_weights)

#print out of sample performance
print("\nOut of Sample Portfolio Performance Comparison:")
out_of_sample_performance = pd.DataFrame({
    "portfolio": [
        "Minimum Volatility",
        "Maximum Sharpe Ratio",
        "Equal Weight",
    ],
    "return": [
        min_vol_out_of_sample_return,
        max_sharpe_out_of_sample_return,
        equal_out_of_sample_return,
    ],
    "volatility": [
        min_vol_out_of_sample_volatility,
        max_sharpe_out_of_sample_volatility,
        equal_out_of_sample_volatility,
    ],
    "sharpe_ratio": [
        min_vol_out_of_sample_sharpe_ratio,
        max_sharpe_out_of_sample_sharpe_ratio,
        equal_out_of_sample_sharpe_ratio,
    ],
})

print(out_of_sample_performance.to_string(index=False))

#estimated vs out of sample performance
performance_comparison = pd.DataFrame({
    "Portfolio": ["Minimum Volatility", "Maximum Sharpe Ratio", "Equal Weight"],
    "Estimated Return": [min_vol_return, max_sharpe_return, equal_portfolio_return],
    "OOS Return": [min_vol_out_of_sample_return, max_sharpe_out_of_sample_return, equal_out_of_sample_return],
    "Return \u0394": [min_vol_out_of_sample_return - min_vol_return, max_sharpe_out_of_sample_return - max_sharpe_return, equal_out_of_sample_return - equal_portfolio_return],
    "Estimated Volatility": [min_vol_volatility, max_sharpe_volatility, equal_portfolio_volatility],
    "OOS Volatility": [min_vol_out_of_sample_volatility, max_sharpe_out_of_sample_volatility, equal_out_of_sample_volatility],
    "Volatility \u0394": [min_vol_out_of_sample_volatility - min_vol_volatility, max_sharpe_out_of_sample_volatility - max_sharpe_volatility, equal_out_of_sample_volatility - equal_portfolio_volatility],
    "Estimated Sharpe Ratio": [min_vol_sharpe_ratio, max_sharpe_ratio, equal_portfolio_sharpe_ratio],
    "OOS Sharpe Ratio": [min_vol_out_of_sample_sharpe_ratio, max_sharpe_out_of_sample_sharpe_ratio, equal_out_of_sample_sharpe_ratio],
    "Sharpe Ratio \u0394": [min_vol_out_of_sample_sharpe_ratio - min_vol_sharpe_ratio, max_sharpe_out_of_sample_sharpe_ratio - max_sharpe_ratio, equal_out_of_sample_sharpe_ratio - equal_portfolio_sharpe_ratio]
})

performance_comparison_display = performance_comparison.copy()
performance_comparison_display["Estimated Return"] = performance_comparison_display["Estimated Return"].map(lambda x: f"{x:.2%}")
performance_comparison_display["OOS Return"] = performance_comparison_display["OOS Return"].map(lambda x: f"{x:.2%}")
performance_comparison_display["Return \u0394"] = performance_comparison_display["Return \u0394"].map(lambda x: f"{x * 100:+.2f} pp")
performance_comparison_display["Estimated Volatility"] = performance_comparison_display["Estimated Volatility"].map(lambda x: f"{x:.2%}")
performance_comparison_display["OOS Volatility"] = performance_comparison_display["OOS Volatility"].map(lambda x: f"{x:.2%}")
performance_comparison_display["Volatility \u0394"] = performance_comparison_display["Volatility \u0394"].map(lambda x: f"{x * 100:+.2f} pp")
performance_comparison_display["Estimated Sharpe Ratio"] = performance_comparison_display["Estimated Sharpe Ratio"].map(lambda x: f"{x:.3f}")
performance_comparison_display["OOS Sharpe Ratio"] = performance_comparison_display["OOS Sharpe Ratio"].map(lambda x: f"{x:.3f}")
performance_comparison_display["Sharpe Ratio \u0394"] = performance_comparison_display["Sharpe Ratio \u0394"].map(lambda x: f"{x:+.3f}")

print("\nEstimated vs Out of Sample Performance Comparison:")
print(performance_comparison_display.to_string(index=False))


#return contribution using element by element multiplication

def calculate_return_contribution(weights, returns):
    return weights * returns

min_vol_return_contribution = calculate_return_contribution(min_vol_weights, out_of_sample_annualised_returns)
print(min_vol_return_contribution)

assert np.isclose(
    min_vol_return_contribution.sum(),
    min_vol_out_of_sample_return
), "Min volatility return attribution do not sum to portfolio return"

max_sharpe_return_contribution = calculate_return_contribution(
    max_sharpe_weights,
    out_of_sample_annualised_returns
)
print(max_sharpe_return_contribution)

assert np.isclose(
    max_sharpe_return_contribution.sum(),
    max_sharpe_out_of_sample_return
), "Max Sharpe return attribution do not sum to portfolio return"

equal_return_contribution = calculate_return_contribution(equal_weights, out_of_sample_annualised_returns)
print(equal_return_contribution)

assert np.isclose(
    equal_return_contribution.sum(),
    equal_out_of_sample_return
), "Equal Weight return attribution do not sum to portfolio return"

#risk attribution

def calculate_risk_attribution(
        annualised_cov_matrix,
        weights
):
    portfolio_variance = weights @ annualised_cov_matrix @ weights.T
    portfolio_volatility = np.sqrt(portfolio_variance)
    mrc = annualised_cov_matrix @ weights.T / portfolio_volatility
    crc = weights * mrc
    prc = crc / portfolio_volatility
    return prc, crc, mrc, portfolio_volatility


(
    min_vol_prc,
    min_vol_crc,
    min_vol_mrc,
    min_vol_risk_volatility,
) = calculate_risk_attribution(
    out_of_sample_annualised_covariance,
    min_vol_weights
)

assert np.isclose(
    min_vol_crc.sum(),
    min_vol_risk_volatility
), "Component risk contributions do not sum to portfolio volatility"

assert np.isclose(
    min_vol_prc.sum(),
    1.0
), "Percentage risk contributions do not sum to 1"

assert np.isclose(
    min_vol_risk_volatility,
    min_vol_out_of_sample_volatility
), "Minimum volatility risk attribution volatility mismatch"

(
    max_sharpe_prc,
    max_sharpe_crc,
    max_sharpe_mrc,
    max_sharpe_risk_volatility,
) = calculate_risk_attribution(
    out_of_sample_annualised_covariance,
    max_sharpe_weights
)

assert np.isclose(
    max_sharpe_crc.sum(),
    max_sharpe_risk_volatility
), "Component risk contributions do not sum to portfolio volatility"

assert np.isclose(
    max_sharpe_prc.sum(),
    1.0
), "Percentage risk contributions do not sum to 1"

assert np.isclose(
    max_sharpe_risk_volatility,
    max_sharpe_out_of_sample_volatility
), "Max Sharpe risk attribution volatility mismatch"

(
    equal_prc,
    equal_crc,
    equal_mrc,
    equal_risk_volatility,
) = calculate_risk_attribution(
    out_of_sample_annualised_covariance,
    equal_weights
)

assert np.isclose(
    equal_crc.sum(),
    equal_risk_volatility
), "Component risk contributions do not sum to portfolio volatility"

assert np.isclose(
    equal_prc.sum(),
    1.0
), "Percentage risk contributions do not sum to 1"

assert np.isclose(
    equal_risk_volatility,
    equal_out_of_sample_volatility
), "Equal Weight risk attribution volatility mismatch"

#attribution tables for terminal output

def create_attribution_table(
        weights,
        return_contribution,
        mrc,
        crc,
        prc,
):
    attribution_table = pd.DataFrame({
        "Stock": closing_prices.columns,
        "Portfolio Weights": weights,
        "Return Contribution": return_contribution,
        "Marginal Risk Contribution": mrc,
        "Component Risk Contribution": crc,
        "Percentage Risk Contribution": prc,
    })

    display_table = attribution_table.copy()
    display_table["Portfolio Weights"] = (
        display_table["Portfolio Weights"]
        .map(lambda x: f"{x:.2%}")
    )
    display_table["Return Contribution"] = (
        display_table["Return Contribution"]
        .map(lambda x: f"{x * 100:+.2f} pp")
    )
    display_table["Marginal Risk Contribution"] = (
        display_table["Marginal Risk Contribution"]
        .map(lambda x: f"{x:.2%}")
    )
    display_table["Component Risk Contribution"] = (
        display_table["Component Risk Contribution"]
        .map(lambda x: f"{x * 100:+.2f} pp vol")
    )
    display_table["Percentage Risk Contribution"] = (
        display_table["Percentage Risk Contribution"]
        .map(lambda x: f"{x:.2%}")
    )

    return attribution_table, display_table


min_vol_attribution_table, min_vol_display = (
    create_attribution_table(
        min_vol_weights,
        min_vol_return_contribution,
        min_vol_mrc,
        min_vol_crc,
        min_vol_prc,
    )
)

max_sharpe_attribution_table, max_sharpe_display = (
    create_attribution_table(
        max_sharpe_weights,
        max_sharpe_return_contribution,
        max_sharpe_mrc,
        max_sharpe_crc,
        max_sharpe_prc,
    )
)

equal_attribution_table, equal_display = (
    create_attribution_table(
        equal_weights,
        equal_return_contribution,
        equal_mrc,
        equal_crc,
        equal_prc,
    )
)


print("\nMinimum Volatility Portfolio Attribution Table:")
print(min_vol_display.to_string(index=False))

print("\nMaximum Sharpe Ratio Portfolio Attribution Table:")
print(max_sharpe_display.to_string(index=False))

print("\nEqual Weight Portfolio Attribution Table:")
print(equal_display.to_string(index=False))

#visualisations



#reusable function for performance comparison plots
def plot_performance_comparison(
        estimated_values,
        oos_values,
        portfolio_labels,
        ylabel,
        title,
        percentage=True
):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bar_width = 0.35
    index = np.arange(len(estimated_values))

    ax.bar(
        index,
        estimated_values,
        bar_width,
        label="Estimated"
    )

    ax.bar(
        index + bar_width,
        oos_values,
        bar_width,
        label="OOS"
    )

    for i in range(len(estimated_values)):
        ax.text(index[i], estimated_values.iloc[i], f"{estimated_values.iloc[i]:.2%}" if percentage else f"{estimated_values.iloc[i]:.3f}", ha='center', va='bottom')
        ax.text(index[i] + bar_width, oos_values.iloc[i], f"{oos_values.iloc[i]:.2%}" if percentage else f"{oos_values.iloc[i]:.3f}", ha='center', va='bottom')

    ax.set_xlabel("Portfolio")
    ax.set_ylabel(ylabel)
    if percentage:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(title)
    ax.set_xticks(index + bar_width / 2, portfolio_labels)
    ax.legend()
    plt.show()


#estimated vs OOS return comparison
estimated_returns = performance_comparison["Estimated Return"]
oos_returns = performance_comparison["OOS Return"]

plot_performance_comparison(
    estimated_returns,
    oos_returns,
    performance_comparison["Portfolio"],
    "Annualised Return",
    "Estimated vs OOS Annualised Portfolio Return",
    percentage=True
)

#estimated vs OOS volatility comparison
estimated_volatility = performance_comparison["Estimated Volatility"]
oos_volatility = performance_comparison["OOS Volatility"]

plot_performance_comparison(
    estimated_volatility,
    oos_volatility,
    performance_comparison["Portfolio"],
    "Annualised Volatility",
    "Estimated vs OOS Annualised Portfolio Volatility",
    percentage=True
)

#estimated vs OOS sharpe ratio comparison
estimated_sharpe = performance_comparison["Estimated Sharpe Ratio"]
oos_sharpe = performance_comparison["OOS Sharpe Ratio"]

plot_performance_comparison(
    estimated_sharpe,
    oos_sharpe,
    performance_comparison["Portfolio"],
    "Sharpe Ratio",
    "Estimated vs OOS Portfolio Sharpe Ratio",
    percentage=False
)



#portfolio weight vs percentage risk contribution
#reusable function for weight vs risk plots
def plot_weight_vs_risk(
        weights,
        risk_contributions,
        stock_labels,
        title
):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    index = np.arange(len(weights))
    bar_width = 0.35

    ax.bar(
        index,
        weights,
        bar_width,
        label="Portfolio Weight"
    )

    ax.bar(
        index + bar_width,
        risk_contributions,
        bar_width,
        label="Risk Contribution"
    )

    for i in range(len(weights)):
        ax.text(index[i], weights[i], f"{weights[i]:.2%}", ha='center', va='bottom')
        ax.text(index[i] + bar_width, risk_contributions.iloc[i], f"{risk_contributions.iloc[i]:.2%}", ha='center', va='bottom')

    ax.set_xlabel("Stock")
    ax.set_ylabel("Portfolio Weight / Risk Contribution")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(title)
    ax.set_xticks(index + bar_width / 2, stock_labels)
    ax.legend()
    plt.show()

#minimum volatility weight vs risk

plot_weight_vs_risk(
    min_vol_weights,
    min_vol_prc,
    closing_prices.columns,
    "Minimum Volatility Portfolio: Weight vs Risk Contribution"
)

#maximum sharpe weight vs risk

plot_weight_vs_risk(
    max_sharpe_weights,
    max_sharpe_prc,
    closing_prices.columns,
    "Maximum Sharpe Portfolio: Weight vs Risk Contribution"
)

#equal weight vs risk

plot_weight_vs_risk(
    equal_weights,
    equal_prc,
    closing_prices.columns,
    "Equal Weight Portfolio: Weight vs Risk Contribution"
)



#return contribution by stock
#reusable function for return contribution plot
def plot_return_contribution(
        min_vol_contributions,
        max_sharpe_contributions,
        equal_contributions,
        stock_labels,
        title
):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    index = np.arange(len(min_vol_contributions))
    bar_width = 0.25

    ax.bar(
        index,
        min_vol_contributions,
        bar_width,
        label="Minimum Volatility"
    )

    ax.bar(
        index + bar_width,
        max_sharpe_contributions,
        bar_width,
        label="Maximum Sharpe"
    )

    ax.bar(
        index + 2 * bar_width,
        equal_contributions,
        bar_width,
        label="Equal Weight"
    )

    for i in range(len(min_vol_contributions)):
        ax.text(index[i], min_vol_contributions.iloc[i], f"{min_vol_contributions.iloc[i]*100:+.2f} pp", ha='center', va='bottom')
        ax.text(index[i] + bar_width, max_sharpe_contributions.iloc[i], f"{max_sharpe_contributions.iloc[i]*100:+.2f} pp", ha='center', va='bottom')
        ax.text(index[i] + 2 * bar_width, equal_contributions.iloc[i], f"{equal_contributions.iloc[i]*100:+.2f} pp", ha='center', va='bottom')

    ax.set_xlabel("Stock")
    ax.set_ylabel("Return Contribution (percentage points)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.0f}"))
    ax.set_title(title)
    ax.set_xticks(index + bar_width, stock_labels)
    ax.legend()
    plt.show()

plot_return_contribution(
    min_vol_return_contribution,
    max_sharpe_return_contribution,
    equal_return_contribution,
    closing_prices.columns,
    "Out-of-Sample Return Contribution by Stock"
)

#cumulative OOS performance
#use the same target weights throughout the OOS period
min_vol_daily_portfolio_returns = out_of_sample_returns @ min_vol_weights
max_sharpe_daily_portfolio_returns = out_of_sample_returns @ max_sharpe_weights
equal_daily_portfolio_returns = out_of_sample_returns @ equal_weights

min_vol_cumulative_returns = ((1 + min_vol_daily_portfolio_returns).cumprod() - 1)
max_sharpe_cumulative_returns = ((1 + max_sharpe_daily_portfolio_returns).cumprod() - 1)
equal_cumulative_returns = ((1 + equal_daily_portfolio_returns).cumprod() - 1)

def plot_cumulative_performance(
        min_vol_cumulative_returns,
        max_sharpe_cumulative_returns,
        equal_cumulative_returns,
        title
):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    ax.plot(min_vol_cumulative_returns, label="Minimum Volatility")
    ax.plot(max_sharpe_cumulative_returns, label="Maximum Sharpe")
    ax.plot(equal_cumulative_returns, label="Equal Weight")

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(title)
    ax.axhline(0, linewidth=0.8)
    ax.legend()
    plt.show()

plot_cumulative_performance(
    min_vol_cumulative_returns,
    max_sharpe_cumulative_returns,
    equal_cumulative_returns,
    "Out-of-Sample Cumulative Performance"
)


#final validation checks

#check all portfolio weights add up to 1
assert np.isclose(
    min_vol_weights.sum(), 1
), "Minimum Volatility portfolio weights do not sum to 1"

assert np.isclose(
    max_sharpe_weights.sum(), 1
), "Maximum Sharpe portfolio weights do not sum to 1"

assert np.isclose(
    equal_weights.sum(), 1
), "Equal Weight portfolio weights do not sum to 1"


#return contributions sum to annualised OOS portfolio returns
assert np.isclose(
    min_vol_return_contribution.sum(),
    min_vol_out_of_sample_return
), "Minimum Volatility return contributions do not sum to OOS portfolio return"

assert np.isclose(
    max_sharpe_return_contribution.sum(),
    max_sharpe_out_of_sample_return
), "Maximum Sharpe return contributions do not sum to OOS portfolio return"

assert np.isclose(
    equal_return_contribution.sum(),
    equal_out_of_sample_return
), "Equal Weight return contributions do not sum to OOS portfolio return"


#component risk contributions sum to portfolio volatility
assert np.isclose(
    min_vol_crc.sum(),
    min_vol_risk_volatility
), "Minimum Volatility component risk contributions do not sum to portfolio volatility"

assert np.isclose(
    max_sharpe_crc.sum(),
    max_sharpe_risk_volatility
), "Maximum Sharpe component risk contributions do not sum to portfolio volatility"

assert np.isclose(
    equal_crc.sum(),
    equal_risk_volatility
), "Equal Weight component risk contributions do not sum to portfolio volatility"


#percentage risk contributions sum to 1
assert np.isclose(
    min_vol_prc.sum(), 1
), "Minimum Volatility percentage risk contributions do not sum to 1"

assert np.isclose(
    max_sharpe_prc.sum(), 1
), "Maximum Sharpe percentage risk contributions do not sum to 1"

assert np.isclose(
    equal_prc.sum(), 1
), "Equal Weight percentage risk contributions do not sum to 1"


#risk-attribution volatility matches independently calculated OOS portfolio volatility
assert np.isclose(
    min_vol_risk_volatility,
    min_vol_out_of_sample_volatility
), "Minimum Volatility attribution volatility does not match OOS volatility"

assert np.isclose(
    max_sharpe_risk_volatility,
    max_sharpe_out_of_sample_volatility
), "Maximum Sharpe attribution volatility does not match OOS volatility"

assert np.isclose(
    equal_risk_volatility,
    equal_out_of_sample_volatility
), "Equal Weight attribution volatility does not match OOS volatility"


print("All final validation checks passed.")