# Stock Market Analysis

A collection of Python projects exploring quantitative finance, portfolio analytics and financial data analysis using historical market data from Yahoo Finance.

The repository is organised as a series of progressively more advanced projects, beginning with the analysis of a single stock and expanding into multi-stock comparisons and portfolio performance evaluation.

---

# Projects

## 1. Single Stock Analysis (`single_stock_analysis.py`)

Analyses an individual stock using historical adjusted closing prices.

Features:

- Total Return
- Compound Annual Growth Rate (CAGR)
- Annualised Volatility
- Best and Worst Trading Day
- Best and Worst Calendar Month
- Historical Price Chart

---

## 2. Multi-Stock Performance Comparison (`compare_stocks.py`)

Compares several companies over the same time period.

Current tickers:

- NVDA
- AAPL
- MSFT
- GOOGL
- AMZN

Features:

- Downloads adjusted historical prices
- Starting and Ending Prices
- Total Return
- Compound Annual Growth Rate (CAGR)
- Annualised Volatility
- Best and Worst Trading Day
- Best and Worst Calendar Month
- Formatted comparison table
- Historical price comparison chart

---

## 3. Equal-Weight Portfolio Analysis (`portfolio_analysis.py`)

Models an equally weighted portfolio consisting of:

- Nvidia
- Apple
- Microsoft
- Alphabet (Google)
- Amazon

The portfolio begins with an initial value of **$100,000**, allocating **20%** of the capital to each holding.

The current implementation assumes the portfolio is rebalanced back to its target weights each trading day. Transaction costs, taxes and slippage are not included.

The project calculates:

### Portfolio Performance

- Daily portfolio returns
- Portfolio growth through compounding
- Final portfolio value
- Portfolio profit
- Total return
- Compound Annual Growth Rate (CAGR)

### Risk Metrics

- Annualised portfolio volatility
- Sharpe Ratio (assuming a zero risk-free rate)
- Maximum Drawdown

### Individual Stock Metrics

- Starting and Ending Prices
- Total Return
- CAGR
- Annualised Volatility
- Sharpe Ratio
- Maximum Drawdown

### Visualisations

- Portfolio value over time
- Portfolio performance versus the S&P 500 benchmark
- Portfolio drawdown over time

### Reporting

- Portfolio summary table
- Individual stock comparison table

---

# Technologies Used

- Python
- Pandas
- Matplotlib
- yfinance

---

# Future Improvements

- Buy-and-hold portfolio modelling
- Monthly, quarterly and annual portfolio rebalancing
- Beta calculations
- Correlation matrix
- Covariance matrix
- Monte Carlo portfolio simulations
- Modern Portfolio Theory (MPT)
- Efficient Frontier optimisation
- Portfolio optimisation using maximum Sharpe and minimum variance portfolios

---

# Notes

This repository is intended as a learning project that develops practical quantitative finance skills using Python. Each project builds on concepts introduced in the previous one while introducing additional financial statistics, portfolio analytics and data visualisation techniques.