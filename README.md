# Stock Market Analysis

A collection of Python projects exploring quantitative finance and stock market analysis using historical market data from Yahoo Finance.

## Projects

### 1. Single Stock Analysis (`single_stock_analysis.py`)

Analyses an individual stock and calculates:

- Total Return
- Annualised Return (CAGR)
- Annualised Volatility
- Best and Worst Trading Day
- Best and Worst Month
- Historical Price Chart

---

### 2. Multi-Stock Performance Comparison (`compare_stocks.py`)

Compares multiple companies over the same time period.

Current tickers:
- NVDA
- AAPL
- MSFT
- GOOGL
- AMZN

Features:
- Downloads adjusted historical prices
- Calculates:
  - Starting and Ending Prices
  - Total Return
  - Annualised Return (CAGR)
  - Annualised Volatility
  - Best and Worst Trading Day
  - Best and Worst Calendar Month
- Displays a formatted comparison table
- Plots adjusted historical share prices

## Technologies Used

- Python
- Pandas
- Matplotlib
- yfinance

## Future Improvements

- Portfolio performance analysis
- Sharpe Ratio
- Beta calculations
- Correlation matrix
- Monte Carlo simulations
- Portfolio optimisation (Modern Portfolio Theory)
- Efficient Frontier