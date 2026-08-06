# Future Project Ideas

This document tracks completed project extensions and planned future projects that build on the work in this repository.

The aim is to gradually develop my understanding of quantitative finance while improving my Python, data analysis, statistical modelling and financial visualisation skills.

Each project is intended to build on concepts introduced in earlier work rather than exist as an isolated analysis.

---

# Completed Project Extensions

## Project 5.5 - Rolling Correlation Dashboard

### Goal

Extend the static currency correlation matrix into a rolling correlation dashboard to examine how relationships between currency pairs change through time.

### Completed Features

- Downloaded historical foreign-exchange data
- Calculated daily percentage returns
- Calculated rolling 60-day Pearson correlation matrices
- Built an interactive heatmap with a date slider
- Identified the strongest positive correlation
- Identified the strongest negative correlation
- Calculated average correlation
- Calculated average absolute correlation
- Added correlation regime classifications
- Identified the largest change between rolling windows
- Compared previous and current correlation values
- Produced an animated demonstration for the repository README

### Concepts Learned

- Static versus rolling correlation
- Rolling windows
- MultiIndex DataFrames
- Changing market relationships
- Correlation regimes
- Interactive Matplotlib controls

### Possible Future Extensions

- Compare 20-day, 60-day and 252-day rolling windows
- Add filters for different currency groups
- Export selected correlation matrices
- Add automatic market-event annotations
- Compare rolling correlation with rolling volatility

---

## Project 6 - Stock Covariance Matrix

### Goal

Understand the difference between covariance and correlation and show how covariance is used in portfolio-risk calculations.

### Completed Features

- Downloaded historical stock-price data
- Calculated daily percentage returns
- Calculated daily covariance matrices
- Converted daily covariance into annualised covariance
- Extracted unique stock pairs using the upper triangle
- Identified the highest covariance pair
- Identified the lowest covariance pair
- Calculated average covariance
- Identified the highest-variance stock
- Identified the lowest-variance stock
- Produced an annualised covariance heatmap
- Added a summary panel beside the heatmap

### Concepts Learned

- Covariance versus correlation
- Variance on the diagonal of a covariance matrix
- Joint asset movement
- Daily and annualised covariance
- Why covariance is required for portfolio construction
- Upper-triangle matrix analysis

### Possible Future Extensions

- Rolling stock covariance dashboard
- Compare covariance across different market regimes
- Compare covariance and correlation side by side
- Cluster stocks using covariance or correlation
- Examine covariance stability through time

---

## Project 7 - Portfolio Variance and Risk Contribution

### Goal

Use an annualised covariance matrix and portfolio weights to calculate total portfolio risk and identify which holdings contribute most to that risk.

### Completed Features

- Built an equally weighted five-stock portfolio
- Calculated daily and annualised covariance matrices
- Calculated portfolio variance using weighted covariance
- Verified portfolio variance using the matrix equation \(w^T \Sigma w\)
- Calculated annualised portfolio volatility
- Calculated marginal risk contribution
- Calculated component risk contribution
- Calculated percentage risk contribution
- Confirmed that component risk contributions sum to portfolio volatility
- Confirmed that percentage risk contributions sum to 100%
- Built a formatted risk-contribution table
- Produced a sorted risk-contribution bar chart
- Compared capital weights with risk contributions

### Concepts Learned

- Portfolio variance
- Portfolio volatility
- Matrix multiplication in portfolio analysis
- Marginal risk contribution
- Component risk contribution
- Percentage risk contribution
- Equal capital weight versus equal risk weight
- How volatility and covariance both affect portfolio risk

### Possible Future Extensions

- Rolling risk contributions
- Equal-risk-contribution portfolio
- Risk-parity portfolio
- Compare equal weight with risk-based allocation
- Return and risk attribution by holding

---

## Project 8 - Rolling Portfolio Risk

### Goal

Extend static portfolio variance into a rolling analysis to measure how the risk of a fixed portfolio changes through time.

### Completed Features

- Used the same equally weighted five-stock portfolio
- Calculated rolling 60-day covariance matrices
- Annualised every rolling covariance matrix
- Calculated rolling portfolio variance
- Calculated rolling annualised portfolio volatility
- Stored the results as dated Pandas Series
- Calculated current, average, maximum and minimum variance
- Calculated current, average, maximum and minimum volatility
- Identified the dates of maximum and minimum portfolio risk
- Added an average-volatility reference line
- Added markers for maximum, minimum and current volatility
- Added manually selected historical market-event annotations
- Produced a polished rolling portfolio volatility chart

### Concepts Learned

- Time-varying portfolio risk
- Rolling covariance
- Rolling portfolio variance
- Rolling annualised volatility
- The difference between data frequency, window length and annualisation
- Volatility clustering
- Why rolling statistics can peak after the original market shock
- How fixed portfolio weights can still produce changing portfolio risk

### Possible Future Extensions

- Compare 20-day, 60-day and 252-day rolling windows
- Add rolling risk contributions by stock
- Compare rolling portfolio volatility with an S&P 500 benchmark
- Classify low-, normal- and high-risk regimes
- Add expanding-window risk estimates
- Compare rolling covariance with exponentially weighted covariance

---

# Planned Projects

## Project 9 - Rolling Sharpe Ratio

### Goal

Measure how the portfolio's risk-adjusted performance changes through time.

### Planned Features

- Calculate daily portfolio returns
- Calculate rolling cumulative returns
- Calculate rolling annualised returns
- Reuse rolling annualised volatility
- Introduce a realistic risk-free rate
- Calculate rolling excess returns
- Calculate the rolling Sharpe Ratio
- Identify current, average, maximum and minimum Sharpe Ratios
- Identify the strongest and weakest risk-adjusted periods
- Plot the rolling Sharpe Ratio through time
- Compare rolling return with rolling volatility

### Concepts to Learn

- Risk-adjusted performance
- Excess return
- Risk-free rate
- Rolling annualisation
- Positive and negative Sharpe Ratios
- Difference between high return and efficient return

---

## Project 10 - Monte Carlo Portfolio Simulation

### Goal

Generate many possible portfolio allocations and compare their expected return, volatility and Sharpe Ratio.

### Planned Features

- Generate random portfolio weights
- Ensure weights sum to 100%
- Calculate expected portfolio return
- Calculate portfolio variance and volatility
- Calculate Sharpe Ratio
- Store portfolio weights and statistics
- Simulate thousands of possible portfolios
- Visualise the risk-return distribution
- Identify approximate minimum-volatility and maximum-Sharpe portfolios

### Concepts to Learn

- Random simulation
- Portfolio weight generation
- Probability distributions
- Risk-return trade-offs
- Search through possible portfolio allocations
- Simulation reproducibility using random seeds

---

## Project 11 - Efficient Frontier

### Goal

Visualise the set of portfolios offering the highest expected return for each level of risk.

### Planned Features

- Plot simulated portfolios by return and volatility
- Highlight the equal-weight portfolio
- Highlight the minimum-volatility portfolio
- Highlight the maximum-Sharpe portfolio
- Identify inefficient portfolios
- Approximate the efficient frontier
- Compare portfolio allocations along the frontier

### Concepts to Learn

- Modern Portfolio Theory
- Dominated portfolios
- Efficient portfolios
- Risk-return optimisation
- Minimum-variance boundary
- Capital allocation choices

---

## Project 12 - Portfolio Optimisation

### Goal

Use numerical optimisation rather than random simulation to solve directly for optimal portfolio weights.

### Planned Features

- Minimum-variance portfolio
- Maximum-Sharpe portfolio
- Portfolio-weight constraints
- Fully invested constraint
- Long-only portfolio
- Optional short-selling comparison
- Target-return portfolios
- Compare optimised portfolios with equal weight
- Compare optimisation results with Monte Carlo estimates

### Concepts to Learn

- Objective functions
- Constraints
- Numerical optimisation
- Local versus global solutions
- Weight bounds
- Sensitivity to expected-return and covariance estimates

---

## Project 13 - CAPM and Portfolio Beta

### Goal

Measure portfolio sensitivity to the wider market and estimate expected returns using CAPM.

### Planned Features

- Download an S&P 500 benchmark
- Calculate individual-stock beta
- Calculate portfolio beta
- Estimate CAPM expected returns
- Compare realised returns with CAPM expectations
- Calculate Jensen's alpha
- Add an optional rolling-beta analysis

### Concepts to Learn

- Beta
- Market sensitivity
- Systematic risk
- Idiosyncratic risk
- CAPM
- Market risk premium
- Jensen's alpha

---

## Project 14 - Factor Models

### Goal

Explain portfolio returns using common systematic risk factors.

### Planned Features

- Introduce Fama-French factors
- Estimate market, size and value exposures
- Add momentum as an optional factor
- Run regression analysis
- Estimate factor-adjusted alpha
- Review coefficient significance
- Review model fit
- Compare raw returns with factor-adjusted returns

### Concepts to Learn

- Factor exposure
- Alpha versus beta
- Regression coefficients
- Statistical significance
- R-squared
- Common drivers of portfolio returns

---

## Project 15 - Return and Risk Attribution

### Goal

Identify which holdings drive portfolio return, volatility and drawdowns.

### Planned Features

- Contribution to portfolio return
- Contribution to portfolio variance
- Contribution to portfolio volatility
- Contribution to drawdown
- Compare return contribution with risk contribution
- Identify major performance drivers
- Produce attribution tables and charts

### Concepts to Learn

- Performance attribution
- Risk attribution
- Concentration
- Contribution versus standalone performance
- Portfolio drivers

---

## Project 16 - Principal Component Analysis

### Goal

Identify common hidden drivers across a group of stocks.

### Planned Features

- Standardise return data
- Apply Principal Component Analysis
- Calculate explained variance
- Interpret principal-component loadings
- Identify the dominant common factor
- Compare the number of stocks with the number of independent risk sources

### Concepts to Learn

- Dimensionality reduction
- Eigenvalues and eigenvectors
- Principal components
- Explained variance
- Common market factors
- Hidden portfolio concentration

---

## Project 17 - Cointegration and Pairs Trading

### Goal

Test whether two assets maintain a stable long-term relationship and build a simple relative-value strategy.

### Planned Features

- Compare correlation with cointegration
- Test for stationarity
- Estimate a hedge ratio
- Construct a spread
- Calculate rolling z-scores
- Define entry and exit signals
- Backtest a simple pairs strategy
- Examine periods when the signal breaks down

### Concepts to Learn

- Cointegration
- Stationarity
- Mean reversion
- Hedge ratios
- Z-scores
- Signal strength and breakdown

---

# Options and Derivatives Projects

## Project 18 - Options Fundamentals

### Planned Topics

- Calls and puts
- Strike prices
- Expiration dates
- Option premiums
- Intrinsic value
- Time value
- Moneyness
- Profit-and-loss diagrams
- Put-call parity

---

## Project 19 - Binomial Option Pricing

### Planned Topics

- One-period binomial model
- Multi-period price trees
- Risk-neutral probability
- Replicating portfolios
- European options
- American options
- Early exercise

---

## Project 20 - Black-Scholes Option Pricing

### Planned Topics

- Black-Scholes assumptions
- European call pricing
- European put pricing
- Risk-neutral valuation
- Comparison with binomial pricing
- Model limitations

---

## Project 21 - Option Greeks

### Planned Topics

- Delta
- Gamma
- Vega
- Theta
- Rho
- Sensitivity charts
- Delta hedging

---

## Project 22 - Monte Carlo Option Pricing

### Planned Topics

- Geometric Brownian motion
- Simulated price paths
- European option pricing
- Confidence intervals
- Convergence testing
- Comparison with Black-Scholes prices

---

## Project 23 - Implied Volatility and Volatility Surfaces

### Planned Topics

- Implied volatility
- Numerical root solving
- Volatility smiles
- Volatility skews
- Term structure
- Volatility surfaces
- Limitations of constant-volatility assumptions

---

# Additional Long-Term Ideas

- Value at Risk
- Conditional Value at Risk
- Historical Simulation VaR
- Parametric VaR
- Stress Testing
- Scenario Analysis
- Maximum Drawdown Modelling
- Volatility Forecasting
- Exponentially Weighted Moving Average
- GARCH Models
- Risk-Parity Portfolios
- Black-Litterman Model
- Momentum Strategies
- Trend-Following Strategies
- Relative-Value Strategies
- Index-Constituent Analysis
- Market-Breadth Analysis
- Portfolio Turnover
- Transaction Costs
- Slippage
- Walk-Forward Testing
- Out-of-Sample Testing