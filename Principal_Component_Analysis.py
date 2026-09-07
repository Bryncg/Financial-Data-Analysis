import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# download stock data
data = yf.download(
    [
        "NVDA", "AAPL", "MSFT",
        "JPM", "BAC", "BRK-B",
        "JNJ", "UNH", "PFE",
        "XOM", "CVX", "BP",
        "AMZN", "WMT", "KO",
        "CAT", "GE", "LMT"
    ],
    start="2015-01-01",
    end="2026-09-01",
    auto_adjust=True,
    progress=False,
)

# rename stock columns
stock_labels = {
    "NVDA": "NVIDIA",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "BRK-B": "Berkshire Hathaway",
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth Group",
    "PFE": "Pfizer",
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "BP": "BP",
    "AMZN": "Amazon",
    "WMT": "Walmart",
    "KO": "Coca-Cola",
    "CAT": "Caterpillar",
    "GE": "General Electric",
    "LMT": "Lockheed Martin",
}

# closing prices and daily returns
closing_prices = data["Close"].rename(columns=stock_labels)
daily_returns = closing_prices.pct_change(fill_method=None)
daily_returns.index.name = None
daily_returns.columns.name = None

# check missing values
missing_counts = daily_returns.isna().sum()
missing_dates = daily_returns[daily_returns.isna().any(axis=1)].index

print("Missing values count for each stock:")
print(missing_counts)
print("\nDates with missing values:")
print(missing_dates)

# first row is expected to be missing because there is no previous day for returns
daily_returns = daily_returns.dropna()

# inspect return structure
daily_returns_corr = daily_returns.corr()
print("\nDaily return correlation matrix:")
print(daily_returns_corr)

# compare daily volatility before scaling
daily_returns_std = daily_returns.std()
print("\nDaily standard deviation as percentage:")
print((daily_returns_std * 100).round(2).astype(str) + "%")


# standardise returns before PCA
scaler = StandardScaler()
daily_returns_scaled = scaler.fit_transform(daily_returns)
daily_returns_scaled = pd.DataFrame(
    daily_returns_scaled,
    index=daily_returns.index,
    columns=daily_returns.columns,
)

# check scaled means and standard deviations
daily_returns_scaled_mean_std = pd.DataFrame({
    "Mean": daily_returns_scaled.mean(),
    "Std": daily_returns_scaled.std(ddof=0),
})

print("\nMean and standard deviation of scaled daily returns:")
print(daily_returns_scaled_mean_std)


# fit PCA using all 18 components
pca = PCA(n_components=18)
daily_returns_pca = pca.fit_transform(daily_returns_scaled)
daily_returns_pca = pd.DataFrame(
    daily_returns_pca,
    index=daily_returns.index,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
)

# explained variance for each component
explained_variance_ratio = pca.explained_variance_ratio_
cum_explained_variance_ratio = explained_variance_ratio.cumsum()

pc_data = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(pca.n_components_)],
    "Explained Variance Ratio": explained_variance_ratio,
    "Cumulative Explained Variance Ratio": cum_explained_variance_ratio,
})

print("\nExplained and cumulative variance ratios:")
print(pc_data)

# minimum number of PCs needed for each threshold
min_steps_80 = (cum_explained_variance_ratio >= 0.80).argmax() + 1
min_steps_90 = (cum_explained_variance_ratio >= 0.90).argmax() + 1
min_steps_95 = (cum_explained_variance_ratio >= 0.95).argmax() + 1

print(f"\nMinimum PCs needed for 80% variance: {min_steps_80}")
print(f"Minimum PCs needed for 90% variance: {min_steps_90}")
print(f"Minimum PCs needed for 95% variance: {min_steps_95}")


# component weights for each stock
pca_components = pca.components_
pca_components_df = pd.DataFrame(
    pca_components,
    index=[f"PC{i+1}" for i in range(pca.n_components_)],
    columns=daily_returns.columns,
)

# rank stocks by absolute importance within each PC
important_features_df = pca_components_df.abs()
important_features_ranked_df = important_features_df.rank(ascending=False, axis=1)

print("\nRanked importance of stocks for each principal component:")
print(important_features_ranked_df)

# print the sorted stock importance for each PC
for i in range(pca.n_components_):
    pc = f"PC{i+1}"
    print(f"\nMost important features for {pc}:")
    sorted_features = important_features_df.loc[pc].sort_values(ascending=False)
    for j, (feature, value) in enumerate(sorted_features.items(), start=1):
        print(f"{j}. {feature} ({value})")

# signed weights for the first three PCs, sorted by absolute size
pc1_sorted = pca_components_df.loc["PC1"].reindex(
    pca_components_df.loc["PC1"].abs().sort_values(ascending=False).index
)
pc2_sorted = pca_components_df.loc["PC2"].reindex(
    pca_components_df.loc["PC2"].abs().sort_values(ascending=False).index
)
pc3_sorted = pca_components_df.loc["PC3"].reindex(
    pca_components_df.loc["PC3"].abs().sort_values(ascending=False).index
)

summary_pc1_pc3_df = pd.DataFrame({
    "PC1 Stocks": pc1_sorted.index,
    "PC1 Weights": pc1_sorted.values,
    "PC2 Stocks": pc2_sorted.index,
    "PC2 Weights": pc2_sorted.values,
    "PC3 Stocks": pc3_sorted.index,
    "PC3 Weights": pc3_sorted.values,
})

print("\nSummary table for PC1-PC3 sorted by absolute magnitude:")
print(summary_pc1_pc3_df)


# figure 1 - scree plot
plt.figure(figsize=(10, 6))
plt.bar(
    range(1, pca.n_components_ + 1),
    explained_variance_ratio,
    alpha=0.5,
    align="center",
    label="Individual Explained Variance",
)
plt.step(
    range(1, pca.n_components_ + 1),
    cum_explained_variance_ratio,
    where="mid",
    linewidth=1.5,
    label="Cumulative Explained Variance",
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Explained Variance")
plt.xticks(range(1, pca.n_components_ + 1))

# mark the variance thresholds
plt.vlines(min_steps_80, 0, 0.80, color="r", linestyle="--", linewidth=0.7)
plt.vlines(min_steps_90, 0, 0.90, color="g", linestyle="--", linewidth=0.7)
plt.vlines(min_steps_95, 0, 0.95, color="b", linestyle="--", linewidth=0.7)
plt.hlines(0.80, 1, min_steps_80, color="r", linestyle="--", linewidth=0.7)
plt.hlines(0.90, 1, min_steps_90, color="g", linestyle="--", linewidth=0.7)
plt.hlines(0.95, 1, min_steps_95, color="b", linestyle="--", linewidth=0.7)

plt.annotate(
    f"{min_steps_80} PCs: {cum_explained_variance_ratio[min_steps_80 - 1]:.1%}",
    xy=(min_steps_80, cum_explained_variance_ratio[min_steps_80 - 1]),
    xytext=(8.3, 0.77),
)
plt.annotate(
    f"{min_steps_90} PCs: {cum_explained_variance_ratio[min_steps_90 - 1]:.1%}",
    xy=(min_steps_90, cum_explained_variance_ratio[min_steps_90 - 1]),
    xytext=(12.3, 0.87),
)
plt.annotate(
    f"{min_steps_95} PCs: {cum_explained_variance_ratio[min_steps_95 - 1]:.1%}",
    xy=(min_steps_95, cum_explained_variance_ratio[min_steps_95 - 1]),
    xytext=(14.3, 0.92),
)

plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
plt.title("PCA Scree Plot: Individual and Cumulative Explained Variance")
plt.grid(axis="y", alpha=0.3)
plt.legend(loc="best")
plt.ylim(0, 1.03)
plt.tight_layout()
plt.show()


# figure 2 - signed PC1-PC8 component-weight heatmap
heatmap_data = pca_components_df.loc["PC1":"PC8"].T
max_abs_weight = np.abs(heatmap_data.values).max()

plt.figure(figsize=(11, 9))
sns.heatmap(
    heatmap_data,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    vmin=-max_abs_weight,
    vmax=max_abs_weight,
    cbar_kws={"label": "Component Weight"},
    linewidths=0.3,
)
plt.title("PCA Component Weights: PC1-PC8")
plt.xlabel("Principal Components")
plt.ylabel("Stocks")
plt.tight_layout()
plt.show()


# figure 3 - PC1, PC2 and PC3 component weights
max_abs_weight_pc1_pc3 = max(
    pc1_sorted.abs().max(),
    pc2_sorted.abs().max(),
    pc3_sorted.abs().max(),
)
x_limit = max_abs_weight_pc1_pc3 * 1.1

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

axes[0].barh(pc1_sorted.index, pc1_sorted.values, color="b", alpha=0.7)
axes[0].invert_yaxis()
axes[0].axvline(x=0, color="black", linewidth=0.8)
axes[0].set_xlabel("Component Weight")
axes[0].set_xlim(-x_limit, x_limit)
axes[0].set_ylabel("Stocks")
axes[0].set_title("PC1 Component Weights")
axes[0].grid(axis="x", alpha=0.3)

axes[1].barh(pc2_sorted.index, pc2_sorted.values, color="g", alpha=0.7)
axes[1].invert_yaxis()
axes[1].axvline(x=0, color="black", linewidth=0.8)
axes[1].set_xlabel("Component Weight")
axes[1].set_xlim(-x_limit, x_limit)
axes[1].set_ylabel("Stocks")
axes[1].set_title("PC2 Component Weights")
axes[1].grid(axis="x", alpha=0.3)

axes[2].barh(pc3_sorted.index, pc3_sorted.values, color="r", alpha=0.7)
axes[2].invert_yaxis()
axes[2].axvline(x=0, color="black", linewidth=0.8)
axes[2].set_xlabel("Component Weight")
axes[2].set_xlim(-x_limit, x_limit)
axes[2].set_ylabel("Stocks")
axes[2].set_title("PC3 Component Weights")
axes[2].grid(axis="x", alpha=0.3)

fig.suptitle("PCA Component Weights: PC1-PC3")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# figure 4 - reconstruction error as more PCs are added
reconstruction_error = []

for n in range(1, pca.n_components_ + 1):
    pc_subset = daily_returns_pca.loc[:, "PC1":f"PC{n}"]
    component_subset = pca_components_df.loc["PC1":f"PC{n}"]
    reconstructed = pc_subset.dot(component_subset)
    error = ((daily_returns_scaled - reconstructed) ** 2).mean().mean()
    reconstruction_error.append(error)

print("\nReconstruction error:")
print(reconstruction_error)

plt.figure(figsize=(10, 6))
plt.plot(
    range(1, pca.n_components_ + 1),
    reconstruction_error,
    marker="o",
    linestyle="-",
)
plt.title("PCA Reconstruction Error vs Number of Principal Components")
plt.xlabel("Number of Principal Components")
plt.ylabel("Reconstruction Error")
plt.xticks(range(1, pca.n_components_ + 1))

plt.axvline(x=min_steps_80, linestyle="--", linewidth=0.8, alpha=0.7)
plt.axvline(x=min_steps_90, linestyle="--", linewidth=0.8, alpha=0.7)
plt.axvline(x=min_steps_95, linestyle="--", linewidth=0.8, alpha=0.7)

plt.annotate(
    f"80% threshold\n{min_steps_80} PCs\nError = {reconstruction_error[min_steps_80 - 1]:.3f}",
    xy=(min_steps_80, reconstruction_error[min_steps_80 - 1]),
    xytext=(8.2, 0.24),
)
plt.annotate(
    f"90% threshold\n{min_steps_90} PCs\nError = {reconstruction_error[min_steps_90 - 1]:.3f}",
    xy=(min_steps_90, reconstruction_error[min_steps_90 - 1]),
    xytext=(12.2, 0.13),
)
plt.annotate(
    f"95% threshold\n{min_steps_95} PCs\nError = {reconstruction_error[min_steps_95 - 1]:.3f}",
    xy=(min_steps_95, reconstruction_error[min_steps_95 - 1]),
    xytext=(14.2, 0.07),
)
plt.annotate(
    "18 PCs\nFull reconstruction",
    xy=(18, reconstruction_error[-1]),
    xytext=(16.6, 0.035),
    ha="left",
    va="bottom",
    arrowprops=dict(arrowstyle="->", linewidth=0.8),
)

plt.ylim(bottom=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# figure 5 - raw-return PCA vs standardised-return PCA
pca_raw = PCA(n_components=pca.n_components_)
pca_raw.fit(daily_returns)

pca_raw_components_df = pd.DataFrame(
    pca_raw.components_,
    index=[f"PC{i+1}" for i in range(pca_raw.n_components_)],
    columns=daily_returns.columns,
)

pc1_comparison_df = pd.DataFrame({
    "Raw": pca_raw_components_df.loc["PC1"],
    "Standardised": pca_components_df.loc["PC1"],
    "Difference": pca_raw_components_df.loc["PC1"] - pca_components_df.loc["PC1"],
})
pc1_comparison_df["Absolute Difference"] = pc1_comparison_df["Difference"].abs()

explained_variance_comparison_df = pd.DataFrame({
    "Raw": pca_raw.explained_variance_ratio_,
    "Standardised": pca.explained_variance_ratio_,
}, index=[f"PC{i+1}" for i in range(pca_raw.n_components_)])

print("\nRaw vs standardised PC1 weights:")
print(pc1_comparison_df.sort_values("Absolute Difference", ascending=False))

print("\nRaw vs standardised explained variance:")
print(explained_variance_comparison_df)

x = np.arange(len(explained_variance_comparison_df))
pc1_comparison_sorted_df = pc1_comparison_df.sort_values(
    "Absolute Difference",
    ascending=False,
)
y = np.arange(len(pc1_comparison_sorted_df))

fig, ax = plt.subplots(2, 1, figsize=(12, 10))

ax[0].bar(
    x - 0.2,
    explained_variance_comparison_df["Raw"],
    width=0.4,
    label="Raw",
)
ax[0].bar(
    x + 0.2,
    explained_variance_comparison_df["Standardised"],
    width=0.4,
    label="Standardised",
)
ax[0].set_xticks(x)
ax[0].set_xticklabels(explained_variance_comparison_df.index)
ax[0].set_ylabel("Explained Variance Ratio")
ax[0].set_title("Raw vs Standardised PCA: Explained Variance")
ax[0].grid(axis="y", alpha=0.3)
ax[0].legend()

ax[1].barh(
    y - 0.2,
    pc1_comparison_sorted_df["Raw"],
    height=0.4,
    label="Raw",
)
ax[1].barh(
    y + 0.2,
    pc1_comparison_sorted_df["Standardised"],
    height=0.4,
    label="Standardised",
)
ax[1].invert_yaxis()
ax[1].set_yticks(y)
ax[1].set_yticklabels(pc1_comparison_sorted_df.index)
ax[1].set_xlabel("PC1 Component Weight")
ax[1].set_title("Raw vs Standardised PCA: PC1 Component Weights")
ax[1].set_xlim(0, 0.45)
ax[1].grid(axis="x", alpha=0.3)
ax[1].legend()

plt.tight_layout()
plt.show()


# rolling 252-day PCA
rolling_window = 252
rolling_pc1_variance = []
rolling_pca_dates = []
rolling_pcs_80 = []
rolling_pcs_90 = []
rolling_pcs_95 = []

for start in range(0, len(daily_returns) - rolling_window + 1):
    end = start + rolling_window
    window_data = daily_returns.iloc[start:end]

    # standardise within each window so the rolling PCA only uses that period
    window_data_standardised = (
        (window_data - window_data.mean())
        / window_data.std(ddof=0)
    )

    pca_window = PCA()
    pca_window.fit(window_data_standardised)

    cumulative_variance = pca_window.explained_variance_ratio_.cumsum()

    rolling_pc1_variance.append(pca_window.explained_variance_ratio_[0])
    rolling_pcs_80.append((cumulative_variance >= 0.80).argmax() + 1)
    rolling_pcs_90.append((cumulative_variance >= 0.90).argmax() + 1)
    rolling_pcs_95.append((cumulative_variance >= 0.95).argmax() + 1)
    rolling_pca_dates.append(window_data.index[-1])

print(f"\nNumber of rolling PCA results: {len(rolling_pc1_variance)}")
print(f"Number of rolling PCA dates: {len(rolling_pca_dates)}")

rolling_pca_df = pd.DataFrame({
    "PC1 Explained Variance": rolling_pc1_variance,
}, index=rolling_pca_dates)
rolling_pca_df.index.name = "Date"

rolling_pcs_df = pd.DataFrame({
    "PCs for 80% Variance": rolling_pcs_80,
    "PCs for 90% Variance": rolling_pcs_90,
    "PCs for 95% Variance": rolling_pcs_95,
}, index=rolling_pca_dates)
rolling_pcs_df.index.name = "Date"

print("\nRolling PC1 explained variance:")
print(rolling_pca_df.head())
print(rolling_pca_df.tail())

print("\nRolling PCA variance thresholds:")
print(rolling_pcs_df.head())
print(rolling_pcs_df.tail())


# figure 6 - rolling PC1 explained variance
market_events = {
    "US-China Trade War": "2018-03-22",
    "COVID-19 Pandemic": "2020-03-11",
}

label_positions = {
    "US-China Trade War": 0.30,
    "COVID-19 Pandemic": 0.30,
}

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    rolling_pca_df.index,
    rolling_pca_df["PC1 Explained Variance"],
    label="PC1 Explained Variance",
)
ax.set_xlabel("Date")
ax.set_ylabel("PC1 Explained Variance Ratio")
ax.set_title("Rolling 252-Day Window PCA: PC1 Explained Variance")
ax.grid(axis="y", alpha=0.3)

for event, date in market_events.items():
    event_date = pd.to_datetime(date)

    ax.axvline(
        event_date,
        color="red",
        linestyle="--",
        alpha=0.7,
    )
    ax.text(
        event_date,
        label_positions[event],
        event,
        rotation=90,
        verticalalignment="top",
        color="red",
    )

ax.legend()
plt.tight_layout()
plt.show()


# figure 7 - rolling number of PCs needed for 80% variance
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    rolling_pcs_df.index,
    rolling_pcs_df["PCs for 80% Variance"],
    label="PCs for 80% Variance",
)
ax.set_xlabel("Date")
ax.set_ylabel("Number of Principal Components")
ax.set_title("Rolling 252-Day Window PCA: PCs for 80% Variance")
ax.set_yticks(range(
    rolling_pcs_df["PCs for 80% Variance"].min(),
    rolling_pcs_df["PCs for 80% Variance"].max() + 1,
))
ax.grid(axis="y", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()


# final validation checks

# explained variance should sum to 1 and cumulative variance should only increase
assert np.isclose(explained_variance_ratio.sum(), 1.0)
assert np.all(np.diff(cum_explained_variance_ratio) >= 0)
assert np.isclose(cum_explained_variance_ratio[-1], 1.0)

# PCA component vectors should be orthogonal
component_identity = pca_components @ pca_components.T
assert np.allclose(
    component_identity,
    np.eye(pca.n_components_),
    atol=1e-10,
)

# transformed PC scores should be uncorrelated
pc_score_corr = daily_returns_pca.corr()
assert np.allclose(
    pc_score_corr.values,
    np.eye(pca.n_components_),
    atol=1e-10,
)

# reconstruction error should fall as more PCs are added
assert np.all(np.diff(reconstruction_error) <= 1e-12)
assert np.isclose(reconstruction_error[-1], 0.0, atol=1e-12)

# rolling PCA sanity checks
assert (
    len(rolling_pc1_variance)
    == len(rolling_pca_dates)
    == len(rolling_pcs_80)
    == len(rolling_pcs_90)
    == len(rolling_pcs_95)
)

assert not np.any(np.isnan(rolling_pc1_variance))
assert not np.any(np.isnan(rolling_pcs_80))
assert not np.any(np.isnan(rolling_pcs_90))
assert not np.any(np.isnan(rolling_pcs_95))

assert all(0 <= value <= 1 for value in rolling_pc1_variance)
assert all(1 <= pc <= 18 for pc in rolling_pcs_80)
assert all(1 <= pc <= 18 for pc in rolling_pcs_90)
assert all(1 <= pc <= 18 for pc in rolling_pcs_95)

assert all(
    rolling_pcs_80[i] <= rolling_pcs_90[i] <= rolling_pcs_95[i]
    for i in range(len(rolling_pcs_80))
)

assert rolling_pca_dates == sorted(rolling_pca_dates)

print("\nAll PCA validation checks passed.")
