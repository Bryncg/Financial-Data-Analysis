import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# Download currency data
# --------------------------------------------------

# Download daily data for the selected currency pairs.
data = yf.download(
    [
        "EURUSD=X",
        "GBPUSD=X",
        "AUDUSD=X",
        "NZDUSD=X",
        "CAD=X",
        "JPY=X",
        "CHF=X"
    ],
    start="2020-01-01",
    end="2023-01-01"
)


# --------------------------------------------------
# Rename the currency pairs
# --------------------------------------------------

# Rename the Yahoo Finance tickers so they're easier to read
# when printed or plotted.
currency_labels = {
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "AUDUSD=X": "AUD/USD",
    "NZDUSD=X": "NZD/USD",
    "CAD=X": "USD/CAD",
    "JPY=X": "USD/JPY",
    "CHF=X": "USD/CHF"
}


# --------------------------------------------------
# Closing prices
# --------------------------------------------------

# Keep only the closing prices and rename the columns.
close_prices = data["Close"].rename(columns=currency_labels)

# Uncomment to inspect the downloaded prices.
# print(close_prices)


# --------------------------------------------------
# Daily returns
# --------------------------------------------------

# Convert prices into daily returns before calculating
# the correlations.
daily_returns = close_prices.pct_change().dropna()

# Remove the default DataFrame labels.
daily_returns.columns.name = None
daily_returns.index.name = None


# --------------------------------------------------
# Correlation matrix
# --------------------------------------------------

# Calculate the Pearson correlation for each currency pair.
correlation_matrix = daily_returns.corr()

# Print the matrix rounded to two decimal places.
print(correlation_matrix.round(2))


# --------------------------------------------------
# Plot the heatmap
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 10))

ax.set_title("Currency Correlation Matrix (Daily Returns)", fontsize=14, pad=20)

# Rotate the labels so they don't overlap.
ax.tick_params(axis="x", labelrotation=45)

# Display the correlation matrix as a heatmap.
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0,
    square=True,
    cbar_kws={"shrink": 0.8, "label": "Correlation"},
    vmin=-1,
    vmax=1,
    ax=ax,
)

# Stops labels being cut off.
plt.tight_layout(rect=[0, 0.08, 1, 0.96])

plt.show()
