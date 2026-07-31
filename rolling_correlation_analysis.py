import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib.widgets import Slider


# --------------------------------------------------
# Download currency data
# --------------------------------------------------

# Download daily prices for the selected currency pairs.
data = yf.download(
    [
        "EURUSD=X",
        "GBPUSD=X",
        "AUDUSD=X",
        "NZDUSD=X",
        "CAD=X",
        "JPY=X",
        "CHF=X",
    ],
    start="2020-01-01",
    end="2023-01-01",
)


# --------------------------------------------------
# Rename the currency pairs
# --------------------------------------------------

# Replace the Yahoo Finance tickers with clearer labels.
currency_labels = {
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "AUDUSD=X": "AUD/USD",
    "NZDUSD=X": "NZD/USD",
    "CAD=X": "USD/CAD",
    "JPY=X": "USD/JPY",
    "CHF=X": "USD/CHF",
}


# --------------------------------------------------
# Closing prices
# --------------------------------------------------

# Keep only the closing prices and rename the columns.
close_prices = (
    data["Close"]
    .rename(columns=currency_labels)
    .dropna()
)

# Uncomment to check the downloaded prices.
# print(close_prices)


# --------------------------------------------------
# Daily returns
# --------------------------------------------------

# Convert closing prices into daily percentage returns.
# Correlation is calculated from returns rather than raw prices.
daily_returns = close_prices.pct_change().dropna()

# Remove the default DataFrame axis names.
daily_returns.columns.name = None
daily_returns.index.name = None


# --------------------------------------------------
# Rolling window settings
# --------------------------------------------------

# Number of trading days used in each correlation calculation.
rolling_window = 60


# --------------------------------------------------
# Rolling correlation calculation
# --------------------------------------------------

# Calculate a new correlation matrix for every date using
# the previous 60 trading days.
rolling_correlation = (
    daily_returns
    .rolling(window=rolling_window)
    .corr()
    .dropna()
)


# --------------------------------------------------
# Available rolling dates
# --------------------------------------------------

# Keep one copy of each date from the MultiIndex.
rolling_dates = (
    rolling_correlation
    .index
    .get_level_values(0)
    .unique()
)

first_date = rolling_dates[0]
middle_date = rolling_dates[len(rolling_dates) // 2]
last_date = rolling_dates[-1]


# --------------------------------------------------
# Extract example rolling matrices
# --------------------------------------------------

first_rolling_correlation = (
    rolling_correlation.loc[first_date]
)

middle_rolling_correlation = (
    rolling_correlation.loc[middle_date]
)

last_rolling_correlation = (
    rolling_correlation.loc[last_date]
)


# --------------------------------------------------
# Print example rolling matrices
# --------------------------------------------------

print(
    f"\nFirst {rolling_window}-Day Rolling Correlation Matrix"
)

print(
    f"Window Ending: {first_date.strftime('%Y-%m-%d')}"
)

print(first_rolling_correlation.round(2))


print(
    f"\nMiddle {rolling_window}-Day Rolling Correlation Matrix"
)

print(
    f"Window Ending: {middle_date.strftime('%Y-%m-%d')}"
)

print(middle_rolling_correlation.round(2))


print(
    f"\nLast {rolling_window}-Day Rolling Correlation Matrix"
)

print(
    f"Window Ending: {last_date.strftime('%Y-%m-%d')}"
)

print(last_rolling_correlation.round(2))


# --------------------------------------------------
# Calculate correlation summary
# --------------------------------------------------

def calculate_correlation_summary(
    current_matrix,
    previous_matrix=None,
):
    # Keep only the upper half of the matrix.
    # k=1 removes the diagonal values of 1.00 and
    # avoids counting each currency pair twice.
    upper_triangle_positions = np.triu_indices_from(
        current_matrix,
        k=1,
    )

    # Extract each unique currency-pair correlation.
    upper_triangle_values = (
        current_matrix.values[upper_triangle_positions]
    )

    # Find where the highest and lowest values appear.
    highest_position = upper_triangle_values.argmax()
    lowest_position = upper_triangle_values.argmin()

    highest_row = (
        upper_triangle_positions[0][highest_position]
    )

    highest_column = (
        upper_triangle_positions[1][highest_position]
    )

    lowest_row = (
        upper_triangle_positions[0][lowest_position]
    )

    lowest_column = (
        upper_triangle_positions[1][lowest_position]
    )

    # Find the currency names for the highest correlation pair.
    highest_pair = (
        current_matrix.index[highest_row],
        current_matrix.columns[highest_column],
    )

    # Find the currency names for the lowest correlation pair.
    lowest_pair = (
        current_matrix.index[lowest_row],
        current_matrix.columns[lowest_column],
    )

    highest_correlation = (
        upper_triangle_values[highest_position]
    )

    lowest_correlation = (
        upper_triangle_values[lowest_position]
    )

    # Signed average correlation.
    # Positive and negative values may cancel each other out.
    average_correlation = (
        upper_triangle_values.mean()
    )

    # Average relationship strength regardless of direction.
    average_absolute_correlation = (
        np.abs(upper_triangle_values).mean()
    )

    # Give the current window a simple regime label.
    # These thresholds are project-defined rather than
    # universal market rules.
    if average_absolute_correlation < 0.30:
        correlation_regime = "Low"

    elif average_absolute_correlation < 0.60:
        correlation_regime = "Moderate"

    else:
        correlation_regime = "High"

    # Compare the current matrix with the previous rolling date.
    if previous_matrix is not None:

        previous_values = (
            previous_matrix.values[
                upper_triangle_positions
            ]
        )

        correlation_changes = (
            upper_triangle_values
            - previous_values
        )

        average_change = (
            np.abs(correlation_changes).mean()
        )

        largest_change_position = (
            np.abs(correlation_changes).argmax()
        )

        largest_change_row = (
            upper_triangle_positions[0][
                largest_change_position
            ]
        )

        largest_change_column = (
            upper_triangle_positions[1][
                largest_change_position
            ]
        )

        largest_change_pair = (
            current_matrix.index[largest_change_row],
            current_matrix.columns[largest_change_column],
        )

        # Keep the previous value, current value and difference
        # for the pair that changed the most.
        largest_change_previous = (
            previous_values[largest_change_position]
        )

        largest_change_current = (
            upper_triangle_values[largest_change_position]
        )

        largest_change = (
            largest_change_current
            - largest_change_previous
        )

    else:

        average_change = None
        largest_change_pair = None
        largest_change_previous = None
        largest_change_current = None
        largest_change = None

    return {
        "Highest Pair": highest_pair,
        "Highest Correlation": highest_correlation,
        "Lowest Pair": lowest_pair,
        "Lowest Correlation": lowest_correlation,
        "Average Correlation": average_correlation,
        "Average Absolute Correlation": (
            average_absolute_correlation
        ),
        "Correlation Regime": correlation_regime,
        "Average Change": average_change,
        "Largest Change Pair": largest_change_pair,
        "Largest Change Previous": largest_change_previous,
        "Largest Change Current": largest_change_current,
        "Largest Change": largest_change,
    }


# --------------------------------------------------
# Create the interactive figure
# --------------------------------------------------

# Wider figure leaves room for the summary panel.
fig = plt.figure(figsize=(15, 9))

# Main heatmap area.
heatmap_ax = fig.add_axes(
    [0.06, 0.20, 0.60, 0.68]
)

# Summary panel on the right.
summary_ax = fig.add_axes(
    [0.71, 0.20, 0.25, 0.65]
)

# Slider underneath the heatmap.
slider_ax = fig.add_axes(
    [0.12, 0.08, 0.48, 0.04]
)

# The summary section is only used for text.
summary_ax.axis("off")


# --------------------------------------------------
# Update the dashboard
# --------------------------------------------------

def update_dashboard(date_position):

    # Slider positions are numeric, so convert the
    # selected value into an integer.
    date_position = int(date_position)

    current_date = rolling_dates[date_position]

    current_matrix = (
        rolling_correlation.loc[current_date]
    )

    # Compare against the immediately previous rolling date.
    if date_position > 0:

        previous_date = rolling_dates[
            date_position - 1
        ]

        previous_matrix = (
            rolling_correlation.loc[previous_date]
        )

    else:

        previous_matrix = None

    summary = calculate_correlation_summary(
        current_matrix,
        previous_matrix,
    )

    # Remove the previous heatmap and summary.
    heatmap_ax.clear()
    summary_ax.clear()
    summary_ax.axis("off")

    # Draw the matrix selected by the slider.
    sns.heatmap(
        current_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        center=0,
        square=True,
        cbar=False,
        vmin=-1,
        vmax=1,
        ax=heatmap_ax,
    )

    heatmap_ax.set_title(
        f"{rolling_window}-Day Rolling Currency "
        f"Correlation Matrix\n"
        f"Window Ending: "
        f"{current_date.strftime('%Y-%m-%d')}",
        pad=15,
    )

    heatmap_ax.tick_params(
        axis="x",
        labelrotation=45,
    )

    heatmap_ax.tick_params(
        axis="y",
        labelrotation=0,
    )

    # Currency names for the strongest positive pair.
    highest_currency_1, highest_currency_2 = (
        summary["Highest Pair"]
    )

    # Currency names for the strongest negative pair.
    lowest_currency_1, lowest_currency_2 = (
        summary["Lowest Pair"]
    )

    # The first date has no previous matrix to compare with.
    if summary["Average Change"] is not None:

        average_change_text = (
            f"{summary['Average Change']:.3f}"
        )

        largest_change_currency_1, (
            largest_change_currency_2
        ) = summary["Largest Change Pair"]

        largest_change_text = (
            f"{largest_change_currency_1} / "
            f"{largest_change_currency_2}\n"
            f"Previous: "
            f"{summary['Largest Change Previous']:+.3f}\n"
            f"Current: "
            f"{summary['Largest Change Current']:+.3f}\n"
            f"Change: "
            f"{summary['Largest Change']:+.3f}"
        )

    else:

        average_change_text = "No previous window"
        largest_change_text = "No previous window"

    # Text shown beside the heatmap.
    summary_text = (
        f"WINDOW SUMMARY\n\n"
        f"Date\n"
        f"{current_date.strftime('%d %b %Y')}\n\n"

        f"Strongest Positive Pair\n"
        f"{highest_currency_1} / "
        f"{highest_currency_2}\n"
        f"{summary['Highest Correlation']:+.2f}\n\n"

        f"Strongest Negative Pair\n"
        f"{lowest_currency_1} / "
        f"{lowest_currency_2}\n"
        f"{summary['Lowest Correlation']:+.2f}\n\n"

        f"Average Correlation\n"
        f"{summary['Average Correlation']:+.2f}\n\n"

        f"Average Absolute Correlation\n"
        f"{summary['Average Absolute Correlation']:.2f}\n\n"

        f"Correlation Regime\n"
        f"{summary['Correlation Regime']}\n\n"

        f"Average Change from Previous Day\n"
        f"{average_change_text}\n\n"

        f"Largest Pair Change\n"
        f"{largest_change_text}"
    )

    summary_ax.text(
        0,
        1,
        summary_text,
        va="top",
        ha="left",
        fontsize=11,
        transform=summary_ax.transAxes,
    )

    # Redraw the figure when the slider moves.
    fig.canvas.draw_idle()


# --------------------------------------------------
# Date slider
# --------------------------------------------------

# The slider stores the position of each date rather
# than storing the date itself.
date_slider = Slider(
    ax=slider_ax,
    label="Rolling Date",
    valmin=0,
    valmax=len(rolling_dates) - 1,
    valinit=len(rolling_dates) - 1,
    valstep=20,
    dragging=True,
    color="#1f77b4",
)


# --------------------------------------------------
# Connect and display the dashboard
# --------------------------------------------------

# Update the heatmap whenever the slider moves.
date_slider.on_changed(update_dashboard)

# Open the dashboard on the final available date.
update_dashboard(len(rolling_dates) - 1)

plt.show()