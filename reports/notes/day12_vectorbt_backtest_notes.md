# Day 12 Vectorbt Backtest Notes

## 1. Objective

The objective of Day 12 was to introduce vectorbt as a scalable backtesting engine.

A simple moving average strategy was tested on SPY using the rule:

Enter when MA20 is greater than MA60.
Exit when MA20 is less than or equal to MA60.

---

## 2. Backtest Setup

Asset:
- SPY

Input period:
- 2022-01-01 to 2024-12-31

Effective backtest period:
- 2022-03-29 to 2024-12-30

The effective start date is later than the input start date because MA60 requires enough historical data before valid signals can be generated.

Initial capital:
- 10,000 USD

Trading fee:
- 0.1% per transaction

---

## 3. Main Results

Strategy total return:
- 7.75%

Benchmark return:
- 32.75%

Maximum drawdown:
- 19.01%

Sharpe ratio:
- 0.34

Total trades:
- 8

Win rate:
- 42.86%

---

## 4. Interpretation

The strategy generated a positive return, but it significantly underperformed the SPY buy-and-hold benchmark.

This means that the MA20/MA60 trend-following rule reduced market participation but failed to capture enough upside during the test period.

The Sharpe ratio of 0.34 also suggests that the risk-adjusted performance was weak.

This result shows that a simple moving average strategy alone is not strong enough and should be improved with additional filters, risk management rules, or machine learning signals.

---

## 5. Summary

I used vectorbt to validate a simple MA20/MA60 trend-following strategy on SPY. The strategy produced a 7.75% return, but it underperformed the SPY buy-and-hold benchmark of 32.75%. This showed that while the strategy could avoid some unfavorable periods, it missed too much upside in a rising market. The result motivated the need for stronger signal engineering and more advanced model-based strategies.
