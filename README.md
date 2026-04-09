# Quantitative Backtesting Engine

A modular, professional-grade trading strategy backtesting system built in Python. Simulates rule-based strategies on real historical market data with realistic transaction costs, slippage modelling, and institutional-grade performance evaluation.

> **Educational project only. Not financial advice.**

---

## Overview

This engine fetches real OHLCV price data, generates trading signals with zero look-ahead bias, simulates bar-by-bar trade execution, and evaluates performance using standard quant finance metrics. It supports multi-strategy comparison and parameter optimization with walk-forward validation.

---

## Strategies Implemented

| Strategy | Logic |
|---|---|
| Moving Average Crossover | BUY when 20-day SMA crosses above 50-day SMA. SELL on cross below. |
| RSI Mean-Reversion | BUY when RSI crosses up through 30 (oversold recovery). SELL at 70. |
| Bollinger Band | BUY at lower band touch. SELL at upper band touch. |

All signals are shifted by one bar — execution happens on the next bar's open, not the same bar the signal was generated on.

---

## Performance Metrics

| Metric | What It Measures |
|---|---|
| Total Return | Raw P&L as a percentage of starting capital |
| CAGR | Annualised compounded return — time-normalised |
| Sharpe Ratio | Return per unit of total risk. Above 1.0 is considered good. |
| Sortino Ratio | Like Sharpe but penalises only downside volatility |
| Max Drawdown | Largest peak-to-trough loss — the gut-check metric |
| Calmar Ratio | CAGR divided by absolute Max Drawdown |
| Win Rate | Percentage of trades that were profitable |
| Profit Factor | Gross wins divided by gross losses. Above 1.5 is respectable. |
| Expectancy | Average expected return per trade — the most honest single number |

A buy-and-hold benchmark is included in every run for fair comparison.

---

## Project Structure

    quant_backtest/
    ├── data.py            Data fetching and validation via yfinance
    ├── strategy.py        Signal generation for all three strategies
    ├── backtest.py        Bar-by-bar trade simulation engine
    ├── metrics.py         All performance and risk calculations
    ├── visualize.py       Charts: signals, equity curve, drawdown, distributions
    ├── optimize.py        Grid search and walk-forward optimization
    ├── main.py            Entry point — configure tickers and settings here
    └── requirements.txt

---

## Tech Stack

| Library | Purpose |
|---|---|
| `yfinance` | Market data via Yahoo Finance |
| `pandas` | Data manipulation and time series |
| `numpy` | Numerical computation |
| `matplotlib` | Charts and visualisation |

---

## How to Run

**1. Clone the repository**

    git clone https://github.com/harshitat08/quant-backtest-engine.git
    cd quant-backtest-engine

**2. Create and activate a virtual environment**

    python3 -m venv venv
    source venv/bin/activate

**3. Install dependencies**

    pip install -r requirements.txt

**4. Run**

    python3 main.py

---

## Configuration

Open `main.py` and edit the CONFIG block at the top:

    "tickers":    ["AAPL", "RELIANCE.NS"],
    "start_date": "2019-01-01",
    "end_date":   "2024-12-31",

Any ticker supported by Yahoo Finance works — US stocks, Indian stocks (`.NS`), ETFs, indices.

---

## Output

All files are saved to the `output/` folder automatically:

- Price chart with buy/sell signal markers and indicator overlays
- Portfolio equity curve vs buy-and-hold benchmark
- Drawdown chart with max drawdown annotation
- Trade return distribution histogram
- Strategy comparison bar charts across 5 metrics
- Trade log CSV per strategy per ticker
- Equity curve CSV per ticker

---

## Key Design Decisions

**No look-ahead bias** — Signals are shifted one bar forward. The strategy never sees tomorrow's data when deciding today's trade.

**Realistic costs** — 0.1% commission and 0.05% slippage applied on every entry and exit. Configurable in BacktestConfig.

**Sharpe as optimization target** — Parameter grid search maximises Sharpe ratio, not raw return. Maximising return trivially rewards excessive risk-taking.

**Walk-forward validation** — Data is split into rolling train/test windows. Parameters are optimised on train, evaluated on unseen test data. This produces out-of-sample metrics that are far more honest than in-sample results alone.

---

## Possible Extensions

- XGBoost / LightGBM signal generation using engineered features
- Volatility-targeted position sizing (Kelly criterion or fixed fractional)
- Portfolio-level backtesting across multiple tickers simultaneously
- Sentiment features from news headlines or earnings call transcripts
- Live paper trading via Alpaca or Zerodha Kite API

---

## Limitations

**Survivorship bias** — Testing only on stocks that are currently listed ignores companies that went bankrupt or were delisted.

**Overfitting** — Optimised parameters look good in-sample but often degrade in live trading. Walk-forward helps but does not eliminate this.

**Market regime** — Moving average strategies perform well in trending markets but generate excessive whipsaws in sideways or high-volatility regimes.

**Cost underestimation** — Real-world costs including market impact, financing charges, and taxes are typically higher than the 0.1% modelled here, especially for smaller positions.

---

*Educational project only. Not financial advice.*