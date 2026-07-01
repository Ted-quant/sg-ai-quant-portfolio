# Day 13 Vectorbt Multi-Asset Backtest Notes

## 1. Objective

The objective of Day 13 was to expand the vectorbt backtest from a single SPY test to a multi-asset test.

The MA20/MA60 trend-following strategy was applied to five global assets.

Assets:
- NVDA
- MSFT
- SPY
- QQQ
- D05_SI

---

## 2. Strategy Rule

The strategy uses a simple moving average trend-following rule.

Entry rule:
- Enter when MA20 is greater than MA60.

Exit rule:
- Exit when MA20 is less than or equal to MA60.

The strategy was tested using vectorbt with 10,000 USD initial capital and 0.1% trading fee.

---

## 3. Main Results

NVDA:
- Strategy return: 182.33%
- Benchmark return: 380.51%
- Max drawdown: 55.42%
- Sharpe ratio: 1.29
- Total trades: 6

MSFT:
- Strategy return: -8.83%
- Benchmark return: 37.90%
- Max drawdown: 29.48%
- Sharpe ratio: -0.10
- Total trades: 9

SPY:
- Strategy return: 7.75%
- Benchmark return: 32.75%
- Max drawdown: 19.01%
- Sharpe ratio: 0.34
- Total trades: 8

QQQ:
- Strategy return: 0.14%
- Benchmark return: 41.56%
- Max drawdown: 27.45%
- Sharpe ratio: 0.10
- Total trades: 8

D05_SI:
- Strategy return: 39.53%
- Benchmark return: 59.15%
- Max drawdown: 14.67%
- Sharpe ratio: 1.23
- Total trades: 7

---

## 4. Interpretation

The MA20/MA60 strategy showed strong positive performance on NVDA and D05_SI.

However, it underperformed the buy-and-hold benchmark across all five assets.

This suggests that the simple moving average strategy reduced some market participation but also missed significant upside during strong upward trends.

The result also shows that strategy performance is highly asset-dependent.

A simple rule that works on one asset may not work well on another asset.

---

## 5. Key Research Insight

The key insight is that a basic trend-following strategy is not sufficient as a final trading model.

It can be useful as a baseline strategy, but it needs to be improved with additional filters, risk management rules, or machine learning-based signals.

This result supports the next stage of the project: developing stronger signal engineering and model-based strategies.

---

## 6. Summary

I expanded my vectorbt backtest from SPY to five global assets across US and SGX markets. The MA20/MA60 strategy performed well on NVDA and DBS, but it underperformed buy-and-hold across the full asset universe. This showed that simple trend-following performance is highly asset-dependent and motivated the need for stronger filters, risk controls, and machine learning-based signals.
