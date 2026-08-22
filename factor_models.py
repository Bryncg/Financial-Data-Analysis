import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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
# Prepare stock returns
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
# Load Fama-French factor data
# --------------------------------------------------

# The Kenneth French files contain four descriptive rows
# before the actual CSV header, so skiprows=4 is required.

ff3_factors = pd.read_csv(
    "F-F_Research_Data_Factors_daily.csv",
    skiprows=4,
    index_col=0,
).dropna()

ff5_factors = pd.read_csv(
    "F-F_Research_Data_5_Factors_2x3_daily.csv",
    skiprows=4,
    index_col=0,
).dropna()


# --------------------------------------------------
# Clean factor data
# --------------------------------------------------

# Convert YYYYMMDD index values into normal pandas dates.

ff3_factors.index = pd.to_datetime(
    ff3_factors.index.astype(str),
    format="%Y%m%d",
)

ff5_factors.index = pd.to_datetime(
    ff5_factors.index.astype(str),
    format="%Y%m%d",
)

# Fama-French returns are supplied in percentage units.
# Divide by 100 so they use the same decimal scale as
# the Yahoo Finance stock returns.

ff3_factors = ff3_factors / 100
ff5_factors = ff5_factors / 100


# --------------------------------------------------
# Match all datasets to the same dates
# --------------------------------------------------

# Using one common date sample is important when comparing
# CAPM, FF3 and FF5. This prevents changes in alpha or R^2
# from being caused by different observation periods.

common_dates = (
    daily_returns.index
    .intersection(ff3_factors.index)
    .intersection(ff5_factors.index)
)

aligned_ff3_data = pd.concat(
    [
        daily_returns.loc[common_dates],
        ff3_factors.loc[common_dates],
    ],
    axis=1,
)

aligned_ff5_data = pd.concat(
    [
        daily_returns.loc[common_dates],
        ff5_factors.loc[common_dates],
    ],
    axis=1,
)


# --------------------------------------------------
# Calculate stock excess returns
# --------------------------------------------------

# Factor regressions use stock returns above the
# daily risk-free rate:
#
# Excess Return = Stock Return - RF

excess_returns_ff3 = (
    aligned_ff3_data[daily_returns.columns]
    - aligned_ff3_data["RF"].values.reshape(-1, 1)
)

excess_returns_ff5 = (
    aligned_ff5_data[daily_returns.columns]
    - aligned_ff5_data["RF"].values.reshape(-1, 1)
)

Y_ff3 = excess_returns_ff3.values
Y_ff5 = excess_returns_ff5.values


# --------------------------------------------------
# Prepare factor regression inputs
# --------------------------------------------------

# FF3 explains excess returns using:
# market, size and value factors.

X_ff3 = aligned_ff3_data[
    [
        "Mkt-RF",
        "SMB",
        "HML",
    ]
].values

# FF5 adds profitability and investment factors.

X_ff5 = aligned_ff5_data[
    [
        "Mkt-RF",
        "SMB",
        "HML",
        "RMW",
        "CMA",
    ]
].values

# CAPM uses only the market excess-return factor.

X_capm = aligned_ff3_data[
    ["Mkt-RF"]
].values


# --------------------------------------------------
# Fit CAPM, FF3 and FF5 regressions
# --------------------------------------------------

capm_reg = LinearRegression()
ff3_reg = LinearRegression()
ff5_reg = LinearRegression()

capm_reg.fit(
    X_capm,
    Y_ff3,
)

ff3_reg.fit(
    X_ff3,
    Y_ff3,
)

ff5_reg.fit(
    X_ff5,
    Y_ff5,
)


# --------------------------------------------------
# Predict stock excess returns
# --------------------------------------------------

capm_predicted_returns = (
    capm_reg.predict(X_capm)
)

ff3_predicted_returns = (
    ff3_reg.predict(X_ff3)
)

ff5_predicted_returns = (
    ff5_reg.predict(X_ff5)
)


# --------------------------------------------------
# Calculate individual stock R^2 values
# --------------------------------------------------

# R^2 measures how much of the variation in each stock's
# daily excess returns is explained by the model.

capm_r2_values = {}
ff3_r2_values = {}
ff5_r2_values = {}

for i, stock in enumerate(
    excess_returns_ff3.columns
):

    capm_r2_values[stock] = r2_score(
        Y_ff3[:, i],
        capm_predicted_returns[:, i],
    )

    ff3_r2_values[stock] = r2_score(
        Y_ff3[:, i],
        ff3_predicted_returns[:, i],
    )

    ff5_r2_values[stock] = r2_score(
        Y_ff5[:, i],
        ff5_predicted_returns[:, i],
    )


# --------------------------------------------------
# Annualise regression alpha
# --------------------------------------------------

# Regression intercepts are daily alpha estimates.
# Multiply by 252 trading days so they can be interpreted
# on an annualised basis.

annualised_alpha_capm = (
    capm_reg.intercept_ * 252
)

annualised_alpha_ff3 = (
    ff3_reg.intercept_ * 252
)

annualised_alpha_ff5 = (
    ff5_reg.intercept_ * 252
)


# --------------------------------------------------
# Store model results
# --------------------------------------------------

capm_results = pd.DataFrame(
    {
        "Alpha": annualised_alpha_capm,
        "MKT Beta": capm_reg.coef_[:, 0],
        "R^2": [
            capm_r2_values[stock]
            for stock
            in excess_returns_ff3.columns
        ],
    },
    index=excess_returns_ff3.columns,
)

ff3_results = pd.DataFrame(
    {
        "Alpha": annualised_alpha_ff3,
        "MKT Beta": ff3_reg.coef_[:, 0],
        "SMB Beta": ff3_reg.coef_[:, 1],
        "HML Beta": ff3_reg.coef_[:, 2],
        "R^2": [
            ff3_r2_values[stock]
            for stock
            in excess_returns_ff3.columns
        ],
    },
    index=excess_returns_ff3.columns,
)

ff5_results = pd.DataFrame(
    {
        "Alpha": annualised_alpha_ff5,
        "MKT Beta": ff5_reg.coef_[:, 0],
        "SMB Beta": ff5_reg.coef_[:, 1],
        "HML Beta": ff5_reg.coef_[:, 2],
        "RMW Beta": ff5_reg.coef_[:, 3],
        "CMA Beta": ff5_reg.coef_[:, 4],
        "R^2": [
            ff5_r2_values[stock]
            for stock
            in excess_returns_ff5.columns
        ],
    },
    index=excess_returns_ff5.columns,
)


# --------------------------------------------------
# Calculate adjusted R^2
# --------------------------------------------------

# Ordinary R^2 cannot decrease when more predictors are
# added. Adjusted R^2 applies a penalty for extra factors,
# making it more useful when comparing CAPM, FF3 and FF5.

n = len(common_dates)

k_capm = X_capm.shape[1]
k_ff3 = X_ff3.shape[1]
k_ff5 = X_ff5.shape[1]

capm_results["Adj R^2"] = (
    1
    - (1 - capm_results["R^2"])
    * (n - 1)
    / (n - k_capm - 1)
)

ff3_results["Adj R^2"] = (
    1
    - (1 - ff3_results["R^2"])
    * (n - 1)
    / (n - k_ff3 - 1)
)

ff5_results["Adj R^2"] = (
    1
    - (1 - ff5_results["R^2"])
    * (n - 1)
    / (n - k_ff5 - 1)
)


# --------------------------------------------------
# Model comparison calculations
# --------------------------------------------------

alpha_change_ff3_capm = (
    ff3_results["Alpha"]
    - capm_results["Alpha"]
)

alpha_change_ff5_capm = (
    ff5_results["Alpha"]
    - capm_results["Alpha"]
)

alpha_change_ff5_ff3 = (
    ff5_results["Alpha"]
    - ff3_results["Alpha"]
)

r2_improvement_ff3_capm = (
    ff3_results["R^2"]
    - capm_results["R^2"]
)

r2_improvement_ff5_capm = (
    ff5_results["R^2"]
    - capm_results["R^2"]
)

r2_improvement_ff5_ff3 = (
    ff5_results["R^2"]
    - ff3_results["R^2"]
)

adj_r2_improvement_ff3_capm = (
    ff3_results["Adj R^2"]
    - capm_results["Adj R^2"]
)

adj_r2_improvement_ff5_capm = (
    ff5_results["Adj R^2"]
    - capm_results["Adj R^2"]
)

adj_r2_improvement_ff5_ff3 = (
    ff5_results["Adj R^2"]
    - ff3_results["Adj R^2"]
)


# --------------------------------------------------
# Final validation
# --------------------------------------------------

# Check datasets contain complete finite values.

assert aligned_ff3_data.notnull().all().all(), (
    "FF3 aligned data contains NaN values"
)

assert aligned_ff5_data.notnull().all().all(), (
    "FF5 aligned data contains NaN values"
)

assert aligned_ff3_data.index.equals(
    aligned_ff5_data.index
), (
    "FF3 and FF5 datasets do not use the same dates"
)

assert np.isfinite(X_capm).all(), (
    "CAPM input contains non-finite values"
)

assert np.isfinite(X_ff3).all(), (
    "FF3 input contains non-finite values"
)

assert np.isfinite(X_ff5).all(), (
    "FF5 input contains non-finite values"
)

assert np.isfinite(Y_ff3).all(), (
    "FF3 excess returns contain non-finite values"
)

assert np.isfinite(Y_ff5).all(), (
    "FF5 excess returns contain non-finite values"
)


# Check regression coefficient dimensions.

assert capm_reg.coef_.shape == (
    len(excess_returns_ff3.columns),
    1,
), (
    "CAPM coefficient matrix has unexpected dimensions"
)

assert ff3_reg.coef_.shape == (
    len(excess_returns_ff3.columns),
    3,
), (
    "FF3 coefficient matrix has unexpected dimensions"
)

assert ff5_reg.coef_.shape == (
    len(excess_returns_ff5.columns),
    5,
), (
    "FF5 coefficient matrix has unexpected dimensions"
)


# Check regression outputs contain finite values.

assert np.isfinite(
    capm_reg.coef_
).all(), (
    "CAPM coefficients contain non-finite values"
)

assert np.isfinite(
    ff3_reg.coef_
).all(), (
    "FF3 coefficients contain non-finite values"
)

assert np.isfinite(
    ff5_reg.coef_
).all(), (
    "FF5 coefficients contain non-finite values"
)

assert np.isfinite(
    capm_reg.intercept_
).all(), (
    "CAPM alpha contains non-finite values"
)

assert np.isfinite(
    ff3_reg.intercept_
).all(), (
    "FF3 alpha contains non-finite values"
)

assert np.isfinite(
    ff5_reg.intercept_
).all(), (
    "FF5 alpha contains non-finite values"
)


# Adjusted R^2 should not exceed raw R^2.

assert (
    capm_results["Adj R^2"]
    <= capm_results["R^2"]
).all(), (
    "CAPM adjusted R^2 exceeds raw R^2"
)

assert (
    ff3_results["Adj R^2"]
    <= ff3_results["R^2"]
).all(), (
    "FF3 adjusted R^2 exceeds raw R^2"
)

assert (
    ff5_results["Adj R^2"]
    <= ff5_results["R^2"]
).all(), (
    "FF5 adjusted R^2 exceeds raw R^2"
)


# FF3 and FF5 both contain the CAPM market factor,
# so raw R^2 should not fall below the matched CAPM result.

assert (
    ff3_results["R^2"]
    >= capm_results["R^2"]
).all(), (
    "FF3 R^2 is unexpectedly below CAPM R^2"
)

assert (
    ff5_results["R^2"]
    >= capm_results["R^2"]
).all(), (
    "FF5 R^2 is unexpectedly below CAPM R^2"
)


# Check the two additional FF5 factors.

assert np.isfinite(
    ff5_results["RMW Beta"]
).all(), (
    "FF5 RMW Beta contains non-finite values"
)

assert np.isfinite(
    ff5_results["CMA Beta"]
).all(), (
    "FF5 CMA Beta contains non-finite values"
)


# --------------------------------------------------
# Format results for terminal
# --------------------------------------------------

# Keep all model results numeric.
# Only this separate table is converted into strings.

#model results for terminal display

model_comparison_display = pd.DataFrame(
    {
        "CAPM  Alpha": capm_results["Alpha"].map("{:.2%}".format),
        "FF3  Alpha": ff3_results["Alpha"].map("{:.2%}".format),
        "FF5  Alpha": ff5_results["Alpha"].map("{:.2%}".format),

        "CAPM  R^2": capm_results["R^2"].map("{:.3f}".format),
        "FF3  R^2": ff3_results["R^2"].map("{:.3f}".format),
        "FF5  R^2": ff5_results["R^2"].map("{:.3f}".format),
    },
    index=excess_returns_ff3.columns,
)

#model improvements for terminal display
model_improvements_display = pd.DataFrame(
    {
        "FF3-CAPM Adj R^2": adj_r2_improvement_ff3_capm.map("{:+.3f}".format),
        "FF5-CAPM Adj R^2": adj_r2_improvement_ff5_capm.map("{:+.3f}".format),
        "FF5-FF3 Adj R^2": adj_r2_improvement_ff5_ff3.map("{:+.3f}".format),
        "FF3-CAPM Alpha": alpha_change_ff3_capm.map(lambda x: f"{x * 100:+.2f} pp"),
        "FF5-CAPM Alpha": alpha_change_ff5_capm.map(lambda x: f"{x * 100:+.2f} pp"),
        "FF5-FF3 Alpha": alpha_change_ff5_ff3.map(lambda x: f"{x * 100:+.2f} pp"),
    },
    index=excess_returns_ff3.columns,
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print(
    "\nCAPM, Fama-French 3-Factor "
    "and 5-Factor Comparison"
)

print(
    f"\nMatched Daily Observations: {n}"
)

print(
    f"Sample Period: "
    f"{common_dates.min():%Y-%m-%d} "
    f"to {common_dates.max():%Y-%m-%d}"
)


print("\nFama-French 3-Factor Exposures")

print(
    ff3_results[
        [
            "MKT Beta",
            "SMB Beta",
            "HML Beta",
        ]
    ]
    .round(3)
    .to_string()
)


print("\nFama-French 5-Factor Exposures")

print(
    ff5_results[
        [
            "MKT Beta",
            "SMB Beta",
            "HML Beta",
            "RMW Beta",
            "CMA Beta",
        ]
    ]
    .round(3)
    .to_string()
)


print("\nModel Comparison")

print(
    model_comparison_display.to_string()
)

print("\nModel Improvements")
print(
    model_improvements_display.to_string()
)


print("\nValidation")

print(
    "Common Dates:",
    aligned_ff3_data.index.equals(
        aligned_ff5_data.index
    ),
)

print(
    "CAPM Coefficients Valid:",
    np.isfinite(
        capm_reg.coef_
    ).all(),
)

print(
    "FF3 Coefficients Valid:",
    np.isfinite(
        ff3_reg.coef_
    ).all(),
)

print(
    "FF5 Coefficients Valid:",
    np.isfinite(
        ff5_reg.coef_
    ).all(),
)

print(
    "RMW and CMA Values Valid:",
    (
        np.isfinite(
            ff5_results["RMW Beta"]
        ).all()
        and np.isfinite(
            ff5_results["CMA Beta"]
        ).all()
    ),
)


# --------------------------------------------------
# Fama-French 5-factor exposure figure
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(14, 7),
    constrained_layout=True,
)

factors = [
    "MKT Beta",
    "SMB Beta",
    "HML Beta",
    "RMW Beta",
    "CMA Beta",
]

factor_positions = np.arange(
    len(factors)
)

bar_width = 0.16


# Plot factor exposures for each stock.

for i, stock in enumerate(
    excess_returns_ff5.columns
):

    ax.bar(
        factor_positions
        + i * bar_width,
        ff5_results.loc[
            stock,
            factors,
        ],
        width=bar_width,
        label=stock,
    )


# Zero separates positive and negative factor exposure.

ax.axhline(
    0,
    color="black",
    linewidth=0.8,
    linestyle="--",
)


# Add factor loading values at the end of each bar.

for i, stock in enumerate(
    excess_returns_ff5.columns
):

    for j, factor in enumerate(
        factors
    ):

        value = ff5_results.loc[
            stock,
            factor,
        ]

        if value >= 0:

            label_y = value + 0.015
            vertical_alignment = "bottom"

        else:

            label_y = value - 0.015
            vertical_alignment = "top"

        ax.text(
            factor_positions[j]
            + i * bar_width,
            label_y,
            f"{value:.2f}",
            ha="center",
            va=vertical_alignment,
            fontsize=8,
        )


ax.set_xlabel(
    "Factors"
)

ax.set_ylabel(
    "Factor Loading"
)

ax.set_title(
    "Fama-French Five-Factor Exposures"
)

ax.set_xticks(
    factor_positions
    + bar_width * 2,
    factors,
)

ax.legend()

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.7,
)

ax.set_ylim(
    -1.15,
    1.9,
)


# --------------------------------------------------
# CAPM vs FF3 vs FF5 R^2 comparison
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(12, 6),
    constrained_layout=True,
)

bar_width = 0.25

stock_positions = np.arange(
    len(excess_returns_ff3.columns)
)


# CAPM bars.

ax.bar(
    stock_positions - bar_width,
    capm_results["R^2"],
    bar_width,
    label="CAPM",
)


# Fama-French 3-factor bars.

ax.bar(
    stock_positions,
    ff3_results["R^2"],
    bar_width,
    label="Fama-French 3-Factor",
)


# Fama-French 5-factor bars.

ax.bar(
    stock_positions + bar_width,
    ff5_results["R^2"],
    bar_width,
    label="Fama-French 5-Factor",
)


# Add R^2 values above each bar.

for i, stock in enumerate(
    excess_returns_ff3.columns
):

    ax.text(
        i - bar_width,
        capm_results.loc[
            stock,
            "R^2",
        ] + 0.01,
        f"{capm_results.loc[stock, 'R^2']:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    ax.text(
        i,
        ff3_results.loc[
            stock,
            "R^2",
        ] + 0.01,
        f"{ff3_results.loc[stock, 'R^2']:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    ax.text(
        i + bar_width,
        ff5_results.loc[
            stock,
            "R^2",
        ] + 0.01,
        f"{ff5_results.loc[stock, 'R^2']:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )


ax.set_xlabel(
    "Stocks"
)

ax.set_ylabel(
    r"$R^2$"
)

ax.set_title(
    r"CAPM vs Fama-French Models: $R^2$ Comparison"
)

ax.set_xticks(
    stock_positions,
    excess_returns_ff3.columns,
)

ax.legend()

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.7,
)

ax.set_ylim(
    0,
    0.75,
)

plt.show()