import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import norm


# --------------------------------------------------
# stock data
# --------------------------------------------------

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
    end="2026-09-01",
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


# --------------------------------------------------
# equal weighted portfolio
# --------------------------------------------------

#equal portfolio weighting
equal_weights = np.array(
    [1 / len(closing_prices.columns)] * len(closing_prices.columns)
)

#portfolio daily returns
portfolio_daily_returns = daily_returns @ equal_weights

print(portfolio_daily_returns.head())
print(portfolio_daily_returns.describe())


# --------------------------------------------------
# historical VaR and expected shortfall
# --------------------------------------------------

#historical VaR finds the loss threshold from previous portfolio returns
def historical_var(portfolio_returns, confidence_level):
    return -portfolio_returns.quantile(1 - confidence_level)


#expected shortfall finds the average loss beyond the VaR threshold
def expected_shortfall(portfolio_returns, confidence_level):
    var = historical_var(portfolio_returns, confidence_level)

    return -portfolio_returns[
        portfolio_returns <= -var
    ].mean()


historical_var_95 = historical_var(
    portfolio_daily_returns,
    0.95,
)

historical_var_99 = historical_var(
    portfolio_daily_returns,
    0.99,
)

expected_shortfall_95 = expected_shortfall(
    portfolio_daily_returns,
    0.95,
)

expected_shortfall_99 = expected_shortfall(
    portfolio_daily_returns,
    0.99,
)

print("Historical VaR 95%:", f"{historical_var_95:.2%}")
print("Historical VaR 99%:", f"{historical_var_99:.2%}")
print("Historical ES 95%:", f"{expected_shortfall_95:.2%}")
print("Historical ES 99%:", f"{expected_shortfall_99:.2%}")


# --------------------------------------------------
# monetary portfolio risk
# --------------------------------------------------

#initial portfolio value
portfolio_value = 100000


#convert VaR percentage into portfolio value
def portfolio_var(portfolio_value, var_value):
    return portfolio_value * var_value


#convert expected shortfall percentage into portfolio value
def portfolio_expected_shortfall(portfolio_value, es_value):
    return portfolio_value * es_value


portfolio_var_95 = portfolio_var(
    portfolio_value,
    historical_var_95,
)

portfolio_var_99 = portfolio_var(
    portfolio_value,
    historical_var_99,
)

portfolio_es_95 = portfolio_expected_shortfall(
    portfolio_value,
    expected_shortfall_95,
)

portfolio_es_99 = portfolio_expected_shortfall(
    portfolio_value,
    expected_shortfall_99,
)

print("Portfolio VaR 95%:", f"${portfolio_var_95:,.2f}")
print("Portfolio VaR 99%:", f"${portfolio_var_99:,.2f}")
print("Portfolio ES 95%:", f"${portfolio_es_95:,.2f}")
print("Portfolio ES 99%:", f"${portfolio_es_99:,.2f}")


# --------------------------------------------------
# parametric VaR and expected shortfall
# --------------------------------------------------

def parametric_var(portfolio_returns, confidence_level):
    mean = portfolio_returns.mean()
    std = portfolio_returns.std()

    #find the z-score for the lower tail of the normal distribution
    z_score = norm.ppf(1 - confidence_level)

    #calculate the loss threshold at the chosen confidence level
    return -(mean + z_score * std)


def parametric_expected_shortfall(
    portfolio_returns,
    confidence_level,
):
    mean = portfolio_returns.mean()
    std = portfolio_returns.std()

    #lower tail z-score used for the normal distribution
    z_score = norm.ppf(1 - confidence_level)

    #average expected loss beyond the parametric VaR threshold
    return -(
        mean
        - std
        * norm.pdf(z_score)
        / (1 - confidence_level)
    )


parametric_var_95 = parametric_var(
    portfolio_daily_returns,
    0.95,
)

parametric_var_99 = parametric_var(
    portfolio_daily_returns,
    0.99,
)

parametric_es_95 = parametric_expected_shortfall(
    portfolio_daily_returns,
    0.95,
)

parametric_es_99 = parametric_expected_shortfall(
    portfolio_daily_returns,
    0.99,
)

print("Parametric VaR 95%:", f"{parametric_var_95:.2%}")
print("Parametric VaR 99%:", f"{parametric_var_99:.2%}")
print("Parametric ES 95%:", f"{parametric_es_95:.2%}")
print("Parametric ES 99%:", f"{parametric_es_99:.2%}")


# --------------------------------------------------
# monte carlo VaR and expected shortfall
# --------------------------------------------------

#define number of simulations
num_simulations = 100000

#set random seed so simulations can be reproduced
np.random.seed(25)


#simulate portfolio returns using its historical mean and volatility
def simulate_returns(
    portfolio_returns,
    num_simulations=num_simulations,
):
    mean = portfolio_returns.mean()
    std = portfolio_returns.std()

    simulations = np.random.normal(
        mean,
        std,
        num_simulations,
    )

    return simulations


#find VaR from the lower tail of the simulated returns
def monte_carlo_var(
    simulated_returns,
    confidence_level,
):
    return -np.percentile(
        simulated_returns,
        (1 - confidence_level) * 100,
    )


#average simulated loss beyond the Monte Carlo VaR threshold
def monte_carlo_expected_shortfall(
    simulated_returns,
    confidence_level,
):
    var_threshold = np.percentile(
        simulated_returns,
        (1 - confidence_level) * 100,
    )

    return -simulated_returns[
        simulated_returns <= var_threshold
    ].mean()


#generate one set of simulated returns for both VaR and ES
simulated_returns_data = simulate_returns(
    portfolio_daily_returns
)

monte_carlo_var_95 = monte_carlo_var(
    simulated_returns_data,
    0.95,
)

monte_carlo_var_99 = monte_carlo_var(
    simulated_returns_data,
    0.99,
)

monte_carlo_es_95 = monte_carlo_expected_shortfall(
    simulated_returns_data,
    0.95,
)

monte_carlo_es_99 = monte_carlo_expected_shortfall(
    simulated_returns_data,
    0.99,
)

print("Monte Carlo VaR 95%:", f"{monte_carlo_var_95:.2%}")
print("Monte Carlo VaR 99%:", f"{monte_carlo_var_99:.2%}")
print("Monte Carlo ES 95%:", f"{monte_carlo_es_95:.2%}")
print("Monte Carlo ES 99%:", f"{monte_carlo_es_99:.2%}")


# --------------------------------------------------
# method comparison
# --------------------------------------------------

#compare the three VaR and expected shortfall methods
comparison_df = pd.DataFrame({
    "Method": [
        "Historical",
        "Parametric",
        "Monte Carlo",
    ],
    "VaR 95%": [
        historical_var_95,
        parametric_var_95,
        monte_carlo_var_95,
    ],
    "VaR 99%": [
        historical_var_99,
        parametric_var_99,
        monte_carlo_var_99,
    ],
    "ES 95%": [
        expected_shortfall_95,
        parametric_es_95,
        monte_carlo_es_95,
    ],
    "ES 99%": [
        expected_shortfall_99,
        parametric_es_99,
        monte_carlo_es_99,
    ],
})

print("\nVaR and Expected Shortfall Comparison")
print(comparison_df.to_string(index=False))


# --------------------------------------------------
# out of sample VaR backtesting
# --------------------------------------------------

#split data so VaR is estimated without using future returns
split_date = "2024-01-01"

in_sample_data = portfolio_daily_returns[
    portfolio_daily_returns.index < split_date
]

out_of_sample_data = portfolio_daily_returns[
    portfolio_daily_returns.index >= split_date
]


#calculate historical VaR and ES using only the in-sample period
backtest_var_95 = historical_var(
    in_sample_data,
    0.95,
)

backtest_var_99 = historical_var(
    in_sample_data,
    0.99,
)

backtest_es_95 = expected_shortfall(
    in_sample_data,
    0.95,
)

backtest_es_99 = expected_shortfall(
    in_sample_data,
    0.99,
)


#identify days where the actual loss exceeded historical VaR
historical_breaches_95 = (
    out_of_sample_data < -backtest_var_95
)

historical_breaches_99 = (
    out_of_sample_data < -backtest_var_99
)


#calculate parametric VaR and ES using only the in-sample period
parametric_backtest_var_95 = parametric_var(
    in_sample_data,
    0.95,
)

parametric_backtest_var_99 = parametric_var(
    in_sample_data,
    0.99,
)

parametric_backtest_es_95 = (
    parametric_expected_shortfall(
        in_sample_data,
        0.95,
    )
)

parametric_backtest_es_99 = (
    parametric_expected_shortfall(
        in_sample_data,
        0.99,
    )
)


#identify days where the actual loss exceeded parametric VaR
parametric_breaches_95 = (
    out_of_sample_data
    < -parametric_backtest_var_95
)

parametric_breaches_99 = (
    out_of_sample_data
    < -parametric_backtest_var_99
)


# --------------------------------------------------
# breach rates
# --------------------------------------------------

#calculate the percentage of out-of-sample days that breached VaR
historical_breach_rate_95 = (
    historical_breaches_95.sum()
    / len(out_of_sample_data)
)

historical_breach_rate_99 = (
    historical_breaches_99.sum()
    / len(out_of_sample_data)
)

parametric_breach_rate_95 = (
    parametric_breaches_95.sum()
    / len(out_of_sample_data)
)

parametric_breach_rate_99 = (
    parametric_breaches_99.sum()
    / len(out_of_sample_data)
)


#compare observed breach rates against the expected 5% and 1%
historical_diff_95 = (
    historical_breach_rate_95 - 0.05
)

historical_diff_99 = (
    historical_breach_rate_99 - 0.01
)

parametric_diff_95 = (
    parametric_breach_rate_95 - 0.05
)

parametric_diff_99 = (
    parametric_breach_rate_99 - 0.01
)


print(
    "\nNumber of historical breaches for 95% VaR:",
    historical_breaches_95.sum(),
)

print(
    "Number of historical breaches for 99% VaR:",
    historical_breaches_99.sum(),
)

print(
    "Number of parametric breaches for 95% VaR:",
    parametric_breaches_95.sum(),
)

print(
    "Number of parametric breaches for 99% VaR:",
    parametric_breaches_99.sum(),
)


#compare historical and parametric breach rates
breach_comparison_df = pd.DataFrame({
    "Method": [
        "Historical",
        "Parametric",
    ],
    "Breach Rate 95%": [
        historical_breach_rate_95,
        parametric_breach_rate_95,
    ],
    "Breach Rate 99%": [
        historical_breach_rate_99,
        parametric_breach_rate_99,
    ],
    "Difference 95%": [
        historical_diff_95,
        parametric_diff_95,
    ],
    "Difference 99%": [
        historical_diff_99,
        parametric_diff_99,
    ],
})

print("\nVaR Backtesting")
print(
    breach_comparison_df.to_string(
        index=False
    )
)


# --------------------------------------------------
# final validations
# --------------------------------------------------

#portfolio weights should sum to 1 with no negative weights
assert np.isclose(
    equal_weights.sum(),
    1,
), "Portfolio weights do not sum to 1"

assert np.all(
    equal_weights >= 0
), "Portfolio contains negative weights"


#ES should always be at least as large as VaR
assert (
    expected_shortfall_95
    >= historical_var_95
), "Historical ES 95% is less than VaR 95%"

assert (
    expected_shortfall_99
    >= historical_var_99
), "Historical ES 99% is less than VaR 99%"

assert (
    parametric_es_95
    >= parametric_var_95
), "Parametric ES 95% is less than VaR 95%"

assert (
    parametric_es_99
    >= parametric_var_99
), "Parametric ES 99% is less than VaR 99%"

assert (
    monte_carlo_es_95
    >= monte_carlo_var_95
), "Monte Carlo ES 95% is less than VaR 95%"

assert (
    monte_carlo_es_99
    >= monte_carlo_var_99
), "Monte Carlo ES 99% is less than VaR 99%"


#99% risk should be at least as large as 95% risk
assert historical_var_99 >= historical_var_95
assert expected_shortfall_99 >= expected_shortfall_95

assert parametric_var_99 >= parametric_var_95
assert parametric_es_99 >= parametric_es_95

assert monte_carlo_var_99 >= monte_carlo_var_95
assert monte_carlo_es_99 >= monte_carlo_es_95


#breach rates should remain between 0 and 1
assert 0 <= historical_breach_rate_95 <= 1
assert 0 <= historical_breach_rate_99 <= 1

assert 0 <= parametric_breach_rate_95 <= 1
assert 0 <= parametric_breach_rate_99 <= 1


#99% VaR should have fewer or equal breaches than 95% VaR
assert (
    historical_breaches_99.sum()
    <= historical_breaches_95.sum()
)

assert (
    parametric_breaches_99.sum()
    <= parametric_breaches_95.sum()
)


print(
    "\nIn-sample observations:",
    len(in_sample_data),
)

print(
    "Out-of-sample observations:",
    len(out_of_sample_data),
)

print("All validation checks passed.")


# --------------------------------------------------
# visualisations
# --------------------------------------------------

#portfolio return distribution and VaR thresholds
plt.figure(figsize=(10, 6))

plt.hist(
    portfolio_daily_returns,
    bins=50,
    alpha=0.6,
    color="g",
    label="Portfolio Returns",
)

plt.axvline(
    -historical_var_95,
    color="r",
    linestyle="dashed",
    linewidth=2,
    label="Historical VaR 95%",
)

plt.axvline(
    -historical_var_99,
    color="b",
    linestyle="dashed",
    linewidth=2,
    label="Historical VaR 99%",
)

plt.axvline(
    -parametric_var_95,
    color="r",
    linestyle="solid",
    linewidth=2,
    label="Parametric VaR 95%",
)

plt.axvline(
    -parametric_var_99,
    color="b",
    linestyle="solid",
    linewidth=2,
    label="Parametric VaR 99%",
)

plt.title(
    "Portfolio Return Distribution and VaR Thresholds"
)

plt.xlabel("Daily Returns")
plt.ylabel("Frequency")

plt.gca().xaxis.set_major_formatter(
    PercentFormatter(1.0)
)

plt.legend()
plt.tight_layout()
plt.show()


#historical vs parametric vs Monte Carlo VaR and ES
plt.figure(figsize=(10, 6))

labels = [
    "VaR 95%",
    "VaR 99%",
    "ES 95%",
    "ES 99%",
]

historical_values = [
    historical_var_95,
    historical_var_99,
    expected_shortfall_95,
    expected_shortfall_99,
]

parametric_values = [
    parametric_var_95,
    parametric_var_99,
    parametric_es_95,
    parametric_es_99,
]

monte_carlo_values = [
    monte_carlo_var_95,
    monte_carlo_var_99,
    monte_carlo_es_95,
    monte_carlo_es_99,
]

x = np.arange(len(labels))
width = 0.25

plt.bar(
    x - width,
    historical_values,
    width,
    label="Historical",
)

plt.bar(
    x,
    parametric_values,
    width,
    label="Parametric",
)

plt.bar(
    x + width,
    monte_carlo_values,
    width,
    label="Monte Carlo",
)

plt.xlabel("Metrics")
plt.ylabel("Daily Loss")

plt.title(
    "Historical vs Parametric vs Monte Carlo VaR and ES"
)

plt.xticks(x, labels)

plt.gca().yaxis.set_major_formatter(
    PercentFormatter(1.0)
)

plt.legend()
plt.tight_layout()
plt.show()


#out-of-sample VaR backtest highlighting breaches
plt.figure(figsize=(10, 6))

plt.plot(
    out_of_sample_data.index,
    out_of_sample_data,
    label="Portfolio Returns",
)

plt.axhline(
    -backtest_var_95,
    color="r",
    linestyle="dashed",
    linewidth=2,
    label="Historical VaR 95%",
)

plt.axhline(
    -backtest_var_99,
    color="b",
    linestyle="dashed",
    linewidth=2,
    label="Historical VaR 99%",
)

plt.axhline(
    -parametric_backtest_var_95,
    color="r",
    linestyle="solid",
    linewidth=2,
    label="Parametric VaR 95%",
)

plt.axhline(
    -parametric_backtest_var_99,
    color="b",
    linestyle="solid",
    linewidth=2,
    label="Parametric VaR 99%",
)

plt.scatter(
    out_of_sample_data.index[
        historical_breaches_95
    ],
    out_of_sample_data[
        historical_breaches_95
    ],
    color="r",
    marker="x",
    label="Historical Breaches 95%",
)

plt.scatter(
    out_of_sample_data.index[
        historical_breaches_99
    ],
    out_of_sample_data[
        historical_breaches_99
    ],
    color="b",
    marker="x",
    label="Historical Breaches 99%",
)

plt.scatter(
    out_of_sample_data.index[
        parametric_breaches_95
    ],
    out_of_sample_data[
        parametric_breaches_95
    ],
    color="r",
    marker="o",
    label="Parametric Breaches 95%",
)

plt.scatter(
    out_of_sample_data.index[
        parametric_breaches_99
    ],
    out_of_sample_data[
        parametric_breaches_99
    ],
    color="b",
    marker="o",
    label="Parametric Breaches 99%",
)

plt.title("Out of Sample VaR Backtest")
plt.xlabel("Date")
plt.ylabel("Portfolio Returns")

plt.gca().yaxis.set_major_formatter(
    PercentFormatter(1.0)
)

plt.legend()
plt.tight_layout()
plt.show()


#observed vs expected VaR breach rates
plt.figure(figsize=(10, 6))

labels = [
    "VaR 95%",
    "VaR 99%",
]

historical_breach_rates = [
    historical_breach_rate_95,
    historical_breach_rate_99,
]

expected_breach_rates = [
    0.05,
    0.01,
]

parametric_breach_rates = [
    parametric_breach_rate_95,
    parametric_breach_rate_99,
]

x = np.arange(len(labels))
width = 0.25

plt.bar(
    x - width,
    historical_breach_rates,
    width,
    label="Historical",
)

plt.bar(
    x,
    expected_breach_rates,
    width,
    label="Expected",
)

plt.bar(
    x + width,
    parametric_breach_rates,
    width,
    label="Parametric",
)

plt.xlabel("Metrics")
plt.ylabel("Breach Rate")

plt.title(
    "Observed vs Expected VaR Breach Rates"
)

plt.xticks(x, labels)

plt.gca().yaxis.set_major_formatter(
    PercentFormatter(1.0)
)

plt.legend()
plt.tight_layout()
plt.show()