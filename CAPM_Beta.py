import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from sklearn.linear_model import LinearRegression


# --------------------------------------------------
# Download market data
# --------------------------------------------------

# selected stocks
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

# S&P 500 benchmark
benchmark = yf.download(
    "^GSPC",
    start="2015-01-01",
    end=None,
    auto_adjust=True,
    progress=False,
)


# --------------------------------------------------
# Risk-free rate
# --------------------------------------------------

# use the 3-month US Treasury rate as the risk-free rate
treasury_data = web.DataReader(
    "DGS3MO",
    "fred",
    start="2015-01-01",
    end=None,
).dropna()

# FRED provides the yield as a percentage so convert to decimal
treasury_data = treasury_data / 100

# use the historical average as the annual risk-free rate
historical_risk_free_rate = (
    treasury_data.mean().values[0]
)


# --------------------------------------------------
# Prepare stock and benchmark returns
# --------------------------------------------------

stock_labels = {
    "NVDA": "NVIDIA",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
}

# stock closing prices
closing_prices = (
    data["Close"]
    .rename(columns=stock_labels)
    .dropna()
)

# stock daily returns
daily_returns = (
    closing_prices
    .pct_change()
    .dropna()
)

daily_returns.index.name = None
daily_returns.columns.name = None


# S&P 500 closing prices
benchmark_closing_prices = (
    benchmark["Close"]
    .squeeze()
    .rename("S&P 500")
    .dropna()
)

# S&P 500 daily returns
benchmark_daily_returns = (
    benchmark_closing_prices
    .pct_change()
    .dropna()
)

benchmark_daily_returns.index.name = None


# --------------------------------------------------
# Align stock and market returns
# --------------------------------------------------

# CAPM compares each stock against the market.
# Keeping everything in one DataFrame makes sure the
# stock and S&P 500 returns always refer to the same dates.
aligned_data = pd.concat(
    [
        daily_returns,
        benchmark_daily_returns,
    ],
    axis=1,
    join="inner",
)

# keep a separate stock-only version for calculations
# where the S&P 500 column is not required
aligned_stock_returns = (
    aligned_data
    .drop(columns=["S&P 500"])
)


# --------------------------------------------------
# Calculate individual stock beta
# --------------------------------------------------

# Beta measures how sensitive each stock has historically
# been to movements in the wider market.
#
# Beta = Cov(stock return, market return)
#        --------------------------------
#              Var(market return)

market_variance = (
    aligned_data["S&P 500"].var()
)

# take the S&P 500 covariance column and remove
# the market's covariance with itself
covariance_with_market = (
    aligned_data
    .cov()["S&P 500"]
    .drop("S&P 500")
)

beta_values = (
    covariance_with_market
    / market_variance
)


# --------------------------------------------------
# Equal-weight portfolio beta
# --------------------------------------------------

# Every stock has a 20% allocation, so the portfolio beta
# is the average of the five individual stock betas.
equal_weight_portfolio_beta = (
    beta_values.mean()
)


# --------------------------------------------------
# Market return and market risk premium
# --------------------------------------------------

# historical annualised market return from the same
# aligned dates used for the beta calculation
market_annualised_return = (
    aligned_data["S&P 500"].mean()
    * 252
)

# additional return investors historically received
# from taking market risk rather than the risk-free rate
market_risk_premium = (
    market_annualised_return
    - historical_risk_free_rate
)


# --------------------------------------------------
# CAPM expected returns
# --------------------------------------------------

# CAPM Expected Return =
# Risk-Free Rate + Beta * Market Risk Premium
capm_expected_returns = (
    historical_risk_free_rate
    + beta_values * market_risk_premium
)

# apply the same CAPM equation to the equal-weight portfolio
capm_portfolio_expected_return = (
    historical_risk_free_rate
    + equal_weight_portfolio_beta
    * market_risk_premium
)


# --------------------------------------------------
# Historical returns vs CAPM expected returns
# --------------------------------------------------

# annualise the average daily stock returns using the
# same aligned sample used throughout the CAPM analysis
historical_average_returns = (
    aligned_stock_returns.mean()
    * 252
)

# difference between the historical annualised return
# and the return implied by CAPM
capm_returns_difference = (
    historical_average_returns
    - capm_expected_returns
)


# --------------------------------------------------
# Security Market Line values
# --------------------------------------------------

# The Security Market Line shows the return CAPM predicts
# for different levels of systematic market risk (beta).
sml_beta = np.linspace(
    0,
    beta_values.max() + 0.5,
    100,
)

sml_expected_returns = (
    historical_risk_free_rate
    + sml_beta * market_risk_premium
)


# --------------------------------------------------
# CAPM regression
# --------------------------------------------------

# CAPM can also be estimated using linear regression:
#
# Stock Excess Return =
# Alpha + Beta * Market Excess Return + Error
#
# This gives another estimate of beta and also allows
# the regression alpha to be calculated.

# Convert the annual risk-free rate into a daily rate
# because the regression uses daily stock returns.
daily_risk_free_rate = (
    historical_risk_free_rate
    / 252
)

# Excess return means the return above the risk-free rate.
market_excess_return = (
    aligned_data["S&P 500"]
    - daily_risk_free_rate
)

stock_excess_returns = (
    aligned_stock_returns
    - daily_risk_free_rate
)


# --------------------------------------------------
# Prepare regression inputs
# --------------------------------------------------

# X contains the market excess return.
#
# sklearn expects X to be a two-dimensional array,
# so reshape it into one column of daily observations.
X = (
    market_excess_return
    .values
    .reshape(-1, 1)
)

# Y stores each stock's daily excess returns.
# Keeping these separate makes it easier to see exactly
# what is being compared against the market in the regression.
Y = {}

for stock in aligned_stock_returns.columns:

    Y[stock] = (
        stock_excess_returns[stock]
        .values
    )


# --------------------------------------------------
# Estimate regression beta and alpha
# --------------------------------------------------

reg = LinearRegression()

reg_alpha_values = {}
reg_beta_values = {}

# Run one regression for every stock.
#
# X remains the S&P 500 excess return each time.
# Y changes to the excess returns of the current stock.
for stock in aligned_stock_returns.columns:

    reg.fit(
        X,
        Y[stock],
    )

    # Regression intercept = daily alpha
    reg_alpha_values[stock] = (
        reg.intercept_
    )

    # Regression slope = beta
    reg_beta_values[stock] = (
        reg.coef_[0]
    )


# --------------------------------------------------
# Annualise regression alpha
# --------------------------------------------------

# The regression uses daily returns, so its intercept
# is a daily alpha. Multiply by 252 to compare it with
# the annualised return figures used elsewhere.
annual_reg_alpha_values = {
    stock: alpha * 252
    for stock, alpha
    in reg_alpha_values.items()
}


# Convert regression dictionaries into Series so that
# stocks remain labelled and can be compared directly.
reg_beta_series = pd.Series(
    reg_beta_values,
    name="Regression Beta",
)

annual_alpha_series = pd.Series(
    annual_reg_alpha_values,
    name="Annualised Regression Alpha",
)


# --------------------------------------------------
# Results table
# --------------------------------------------------

results_table = pd.DataFrame({
    "Beta": beta_values,
    "Regression Beta": reg_beta_series,
    "Historical Return": historical_average_returns,
    "CAPM Expected Return": capm_expected_returns,
    "CAPM Return Gap": capm_returns_difference,
    "Annualised Alpha": annual_alpha_series,
})


# --------------------------------------------------
# Final validation
# --------------------------------------------------

# The covariance formula and linear regression should
# give almost exactly the same beta values.
beta_validation = np.allclose(
    beta_values,
    reg_beta_series,
)

# The historical CAPM return gap should also agree with
# the annualised regression alpha for this setup.
alpha_validation = np.allclose(
    capm_returns_difference,
    annual_alpha_series,
)

# Check the main numerical results for missing
# or infinite values.
beta_values_valid = (
    not beta_values.isna().any()
    and not np.isinf(beta_values).any()
)

regression_beta_values_valid = (
    not reg_beta_series.isna().any()
    and not np.isinf(reg_beta_series).any()
)

capm_returns_valid = (
    not capm_expected_returns.isna().any()
    and not np.isinf(capm_expected_returns).any()
)

alpha_values_valid = (
    not annual_alpha_series.isna().any()
    and not np.isinf(annual_alpha_series).any()
)

# An equal-weight portfolio beta should lie between the
# lowest and highest individual stock betas.
portfolio_beta_valid = (
    beta_values.min()
    <= equal_weight_portfolio_beta
    <= beta_values.max()
)


# Stop the program if an important validation fails.
if not beta_values_valid:
    raise ValueError(
        "Beta values contain NaN or infinite values."
    )

if not regression_beta_values_valid:
    raise ValueError(
        "Regression beta values contain NaN or infinite values."
    )

if not capm_returns_valid:
    raise ValueError(
        "CAPM expected returns contain NaN or infinite values."
    )

if not alpha_values_valid:
    raise ValueError(
        "Regression alpha values contain NaN or infinite values."
    )

if not beta_validation:
    raise ValueError(
        "Covariance beta and regression beta do not match."
    )

if not alpha_validation:
    raise ValueError(
        "CAPM return gap and regression alpha do not match."
    )

if not portfolio_beta_valid:
    raise ValueError(
        "Equal-weight portfolio beta is outside the "
        "individual stock beta range."
    )


# --------------------------------------------------
# Format results for terminal
# --------------------------------------------------

# Keep results_table numerical for calculations.
# Only format a copy for terminal output.
results_table_display = (
    results_table.copy()
)

results_table_display["Beta"] = (
    results_table_display["Beta"]
    .map(lambda x: f"{x:.4f}")
)

results_table_display["Regression Beta"] = (
    results_table_display["Regression Beta"]
    .map(lambda x: f"{x:.4f}")
)

for column in [
    "Historical Return",
    "CAPM Expected Return",
    "CAPM Return Gap",
    "Annualised Alpha",
]:

    results_table_display[column] = (
        results_table_display[column]
        .map(lambda x: f"{x:.2%}")
    )


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\nCAPM and Portfolio Beta Analysis")

print("\nMarket Summary")

print(
    "Historical Risk-Free Rate:",
    f"{historical_risk_free_rate:.2%}",
)

print(
    "S&P 500 Annualised Return:",
    f"{market_annualised_return:.2%}",
)

print(
    "Market Risk Premium:",
    f"{market_risk_premium:.2%}",
)


print("\nEqual-Weight Portfolio")

print(
    "Portfolio Beta:",
    f"{equal_weight_portfolio_beta:.4f}",
)

print(
    "CAPM Expected Return:",
    f"{capm_portfolio_expected_return:.2%}",
)


print("\nStock Results")

print(
    results_table_display.to_string()
)


print("\nValidation")

print(
    "Covariance Beta vs Regression Beta:",
    beta_validation,
)

print(
    "CAPM Return Gap vs Regression Alpha:",
    alpha_validation,
)

print(
    "Beta Values Valid:",
    beta_values_valid,
)

print(
    "Regression Beta Values Valid:",
    regression_beta_values_valid,
)

print(
    "CAPM Returns Valid:",
    capm_returns_valid,
)

print(
    "Regression Alpha Values Valid:",
    alpha_values_valid,
)

print(
    "Equal-Weight Portfolio Beta Valid:",
    portfolio_beta_valid,
)


# --------------------------------------------------
# Security Market Line plot
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(14, 8),
    constrained_layout=True,
)


# Reverse the company-name dictionary so ticker symbols
# can be used instead of long company names on the plot.
ticker_labels = {
    value: key
    for key, value in stock_labels.items()
}


# --------------------------------------------------
# Plot CAPM and historical returns
# --------------------------------------------------

ax.scatter(
    beta_values,
    capm_expected_returns,
    color="blue",
    s=65,
    label="CAPM Expected Returns",
    zorder=4,
)

ax.scatter(
    beta_values,
    historical_average_returns,
    color="orange",
    marker="*",
    s=145,
    label="Historical Returns",
    zorder=5,
)

ax.scatter(
    equal_weight_portfolio_beta,
    capm_portfolio_expected_return,
    color="red",
    s=70,
    label="Equal-Weight Portfolio",
    zorder=5,
)


# --------------------------------------------------
# Security Market Line
# --------------------------------------------------

ax.plot(
    sml_beta,
    sml_expected_returns,
    color="green",
    linestyle="--",
    linewidth=1.5,
    label="Security Market Line",
)


# --------------------------------------------------
# Ticker labels
# --------------------------------------------------

# Longer labels need slightly different positions because
# several of the stocks have very similar beta values.
ticker_label_settings = {
    "Apple": {
        "offset": (24, 10),
        "ha": "left",
    },
    "Amazon": {
        "offset": (0, 18),
        "ha": "center",
    },
    "Alphabet": {
        "offset": (-24, 10),
        "ha": "right",
    },
    "Microsoft": {
        "offset": (24, -14),
        "ha": "left",
    },
    "NVIDIA": {
        "offset": (60, -5),
        "ha": "center",
    },
}

for stock in beta_values.index:

    ticker = ticker_labels[stock]

    return_gap = (
        historical_average_returns[stock]
        - capm_expected_returns[stock]
    ) * 100

    ax.annotate(
        f"{ticker} ({return_gap:+.2f} pp)",
        xy=(
            beta_values[stock],
            historical_average_returns[stock],
        ),
        xytext=ticker_label_settings[stock]["offset"],
        textcoords="offset points",
        ha=ticker_label_settings[stock]["ha"],
        fontsize=9,
    )


# --------------------------------------------------
# Historical vs CAPM return gaps
# --------------------------------------------------


for stock in beta_values.index:

    beta = beta_values[stock]

    expected_return = (
        capm_expected_returns[stock]
    )

    historical_return = (
        historical_average_returns[stock]
    )

    # vertical line between CAPM and historical return
    ax.plot(
        [beta, beta],
        [
            expected_return,
            historical_return,
        ],
        color="grey",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        zorder=2,
    )


# --------------------------------------------------
# Plot formatting
# --------------------------------------------------

ax.set_title(
    "Security Market Line: CAPM vs Historical Returns"
)

ax.set_xlabel(
    "Beta"
)

ax.set_ylabel(
    "Annualised Return"
)

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda y, _: f"{y:.0%}"
    )
)

ax.grid(
    alpha=0.25
)

ax.legend(
    loc="upper left"
)

ax.set_xlim(
    0,
    2.0,
)

plt.show()