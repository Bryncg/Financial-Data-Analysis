import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# Download stock data
# --------------------------------------------------

# Download daily price data for the selected technology stocks.
data = yf.download(
    [
        "NVDA",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "TSLA",
        "AMD",
        "INTC",
        "PLTR",
    ],
    start="2020-01-01",
    end=None,
)


# --------------------------------------------------
# Rename the stocks
# --------------------------------------------------

# Replace the ticker symbols with clearer company names
# for the printed matrices and heatmap.
stock_labels = {
    "NVDA": "NVIDIA",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta",
    "TSLA": "Tesla",
    "AMD": "AMD",
    "INTC": "Intel",
    "PLTR": "Palantir",
}


# --------------------------------------------------
# Closing prices
# --------------------------------------------------

# Keep only the closing prices, rename the columns and
# remove dates where any selected stock has missing data.
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

# Convert the closing prices into daily percentage returns.
daily_returns = closing_prices.pct_change().dropna()

# Remove the default DataFrame axis names.
daily_returns.columns.name = None
daily_returns.index.name = None


# --------------------------------------------------
# Covariance matrices
# --------------------------------------------------

# Calculate the covariance matrix from daily stock returns.
#
# Covariance measures how much two stocks move together,
# while also taking the size of their price movements into account.
covariance_matrix = daily_returns.cov()

# Convert daily covariance into annualised covariance.
#
# This assumes approximately 252 trading days in one year.
annualised_covariance_matrix = covariance_matrix * 252


# --------------------------------------------------
# Print the covariance matrices
# --------------------------------------------------

# Print both versions so the daily and annualised
# covariance values can be compared.
print("\nDaily Covariance Matrix")
print(covariance_matrix.round(6))

print("\nAnnualised Covariance Matrix")
print(annualised_covariance_matrix.round(4))


# --------------------------------------------------
# Unique covariance pairs
# --------------------------------------------------

# Keep only the upper half of the matrix.
#
# k=1 removes the diagonal variances and prevents each
# stock pair from being counted twice.
upper_triangle_positions = np.triu_indices_from(
    annualised_covariance_matrix,
    k=1,
)

upper_triangle_values = (
    annualised_covariance_matrix.values[
        upper_triangle_positions
    ]
)

# Find the locations of the highest and lowest
# covariance values between different stocks.
highest_position = upper_triangle_values.argmax()
lowest_position = upper_triangle_values.argmin()

highest_covariance = upper_triangle_values[
    highest_position
]

lowest_covariance = upper_triangle_values[
    lowest_position
]

# Calculate the average covariance across all unique stock pairs.
average_covariance = upper_triangle_values.mean()


# --------------------------------------------------
# Highest and lowest covariance pairs
# --------------------------------------------------

# Use the matrix positions to find the stock names belonging
# to the highest covariance value.
highest_covariance_pair = (
    annualised_covariance_matrix.columns[
        upper_triangle_positions[0][highest_position]
    ],
    annualised_covariance_matrix.columns[
        upper_triangle_positions[1][highest_position]
    ],
)

# Use the same method to find the lowest covariance pair.
lowest_covariance_pair = (
    annualised_covariance_matrix.columns[
        upper_triangle_positions[0][lowest_position]
    ],
    annualised_covariance_matrix.columns[
        upper_triangle_positions[1][lowest_position]
    ],
)


# --------------------------------------------------
# Individual stock variances
# --------------------------------------------------

# The diagonal of a covariance matrix contains the variance
# of each stock rather than covariance between two stocks.
variance_values = (
    annualised_covariance_matrix
    .values
    .diagonal()
)

# Find the positions of the stocks with the highest
# and lowest annualised variance.
highest_variance_stock = variance_values.argmax()
lowest_variance_stock = variance_values.argmin()

# Convert the positions back into company names.
highest_variance_stock_label = (
    annualised_covariance_matrix.columns[
        highest_variance_stock
    ]
)

lowest_variance_stock_label = (
    annualised_covariance_matrix.columns[
        lowest_variance_stock
    ]
)

# Store the corresponding variance values.
highest_variance = variance_values[
    highest_variance_stock
]

lowest_variance = variance_values[
    lowest_variance_stock
]


# --------------------------------------------------
# Prepare summary values
# --------------------------------------------------

# Separate the two company names in each covariance pair
# so they can be displayed cleanly in the summary panel.
highest_stock_1, highest_stock_2 = (
    highest_covariance_pair
)

lowest_stock_1, lowest_stock_2 = (
    lowest_covariance_pair
)


# --------------------------------------------------
# Build the summary panel
# --------------------------------------------------

summary_text = (
    f"ANNUALISED COVARIANCE SUMMARY\n\n"

    f"Highest Variance Stock\n"
    f"{highest_variance_stock_label}\n"
    f"{highest_variance:.4f}\n\n"

    f"Lowest Variance Stock\n"
    f"{lowest_variance_stock_label}\n"
    f"{lowest_variance:.4f}\n\n"

    f"Highest Covariance Pair\n"
    f"{highest_stock_1} / {highest_stock_2}\n"
    f"{highest_covariance:.4f}\n\n"

    f"Lowest Covariance Pair\n"
    f"{lowest_stock_1} / {lowest_stock_2}\n"
    f"{lowest_covariance:.4f}\n\n"

    f"Average Covariance\n"
    f"{average_covariance:.4f}"
)


# --------------------------------------------------
# Create the figure
# --------------------------------------------------

# Use a wider figure so the heatmap and summary panel
# can be displayed beside each other.
fig = plt.figure(figsize=(14, 8))

# Heatmap area on the left.
heatmap_ax = fig.add_axes(
    [0.06, 0.15, 0.65, 0.72]
)

# Summary panel on the right.
summary_ax = fig.add_axes(
    [0.75, 0.20, 0.22, 0.62]
)

# Hide the axes around the summary text.
summary_ax.axis("off")


# --------------------------------------------------
# Display the summary panel
# --------------------------------------------------

summary_ax.text(
    0,
    1,
    summary_text,
    fontsize=11,
    ha="left",
    va="top",
    transform=summary_ax.transAxes,
)


# --------------------------------------------------
# Plot the annualised covariance heatmap
# --------------------------------------------------

heatmap_ax.set_title(
    "Annualised Covariance Matrix of Daily Stock Returns",
    fontsize=14,
    pad=20,
)

sns.heatmap(
    annualised_covariance_matrix,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    cbar=True,
    center=0,
    square=True,
    cbar_kws={
        "shrink": 0.8,
        "label": "Covariance",
    },
    ax=heatmap_ax,
)

# Rotate and align the stock names along the x-axis.
heatmap_ax.tick_params(
    axis="x",
    labelrotation=45,
)

for label in heatmap_ax.get_xticklabels():
    label.set_horizontalalignment("right")

plt.show()