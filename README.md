# Stock Market Analysis

A collection of Python projects exploring quantitative finance, portfolio analytics and financial data analysis using historical market data from Yahoo Finance.

This repository documents my progress as I learn Python for quantitative finance. Each project builds on the previous one, starting with analysing a single stock before moving into portfolio construction, risk metrics and portfolio rebalancing.

---

# Projects

## 1. Single Stock Analysis (`single_stock_analysis.py`)

Analyses an individual stock using historical adjusted closing prices.

### Features

- Total Return
- Compound Annual Growth Rate (CAGR)
- Annualised Volatility
- Best and Worst Trading Day
- Best and Worst Calendar Month
- Historical Price Chart

---

## 2. Multi-Stock Performance Comparison (`compare_stocks.py`)

Compares the historical performance of several companies over the same time period.

### Current portfolio

- NVDA
- AAPL
- MSFT
- GOOGL
- AMZN

### Features

- Downloads historical adjusted prices
- Starting and Ending Prices
- Total Return
- Compound Annual Growth Rate (CAGR)
- Annualised Volatility
- Best and Worst Trading Day
- Best and Worst Calendar Month
- Comparison table
- Historical price comparison chart

---

## 3. Equal-Weight Portfolio Analysis (`portfolio_analysis.py`)

Builds an equally weighted portfolio using the same five stocks.

The portfolio starts with **$100,000**, allocating **20%** to each holding and rebalancing back to the target weights every trading day.

### Portfolio Metrics

- Daily Portfolio Returns
- Portfolio Growth
- Final Portfolio Value
- Portfolio Profit
- Total Return
- Compound Annual Growth Rate (CAGR)

### Risk Metrics

- Annualised Volatility
- Sharpe Ratio (0% risk-free rate)
- Maximum Drawdown

### Individual Stock Metrics

- Starting and Ending Prices
- Total Return
- CAGR
- Annualised Volatility
- Sharpe Ratio
- Maximum Drawdown

### Visualisations

- Portfolio Growth
- Portfolio vs S&P 500
- Portfolio Drawdown

### Reporting

- Portfolio Summary Table
- Individual Stock Comparison Table

---

## 4. Portfolio Rebalancing Comparison (`portfolio_rebalancing.py`)

Compares how different rebalancing frequencies affect portfolio performance over time.

The portfolio starts with **$100,000** invested equally across:

- NVDA
- AAPL
- MSFT
- GOOGL
- AMZN

Five strategies are compared against the **S&P 500** benchmark:

- Buy and Hold
- Daily Rebalancing
- Monthly Rebalancing
- Quarterly Rebalancing
- Annual Rebalancing
- S&P 500 Benchmark

### Performance Metrics

- Final Portfolio Value
- Total Return
- Compound Annual Growth Rate (CAGR)
- Annualised Volatility
- Sharpe Ratio (0% risk-free rate)
- Maximum Drawdown

### Output

- Strategy comparison table
- Portfolio value comparison chart
- Performance comparison against the S&P 500

---

# Technologies Used

- Python
- Pandas
- Matplotlib
- yfinance

---

# Future Improvements

Some ideas I'd like to add as I continue learning:

- Portfolio beta calculations
- Correlation matrix
- Covariance matrix
- Rolling volatility
- Rolling Sharpe ratio
- Monte Carlo portfolio simulations
- Modern Portfolio Theory (MPT)
- Efficient Frontier optimisation
- Portfolio optimisation using maximum Sharpe and minimum variance portfolios
- Portfolio performance attribution
- Factor analysis (Market, Size, Value, Momentum)

---

# Notes

This repository is primarily a learning project as I develop my Python skills alongside quantitative finance concepts.

Rather than focusing on writing production-ready software, the aim is to understand how portfolio statistics, risk measures and investment strategies are calculated before gradually refactoring and improving the code as I learn more.