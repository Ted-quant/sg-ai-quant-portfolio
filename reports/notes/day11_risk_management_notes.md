# Day 11 Risk Management Analysis Notes

## 1. Objective

The objective of Day 11 was to extend the backtesting framework from return analysis to risk management analysis.

Instead of only comparing total return, this analysis evaluates each strategy using downside risk and exposure-based metrics.

The main focus was to understand whether each strategy improves risk-adjusted performance across multiple assets.

---

## 2. Standardized Analysis Setup

The analysis was standardized using the same asset universe and strategy set.

Asset universe:
- NVDA
- MSFT
- SPY
- QQQ
- D05_SI

Strategy set:
- Buy_and_Hold
- Current_Defensive
- Trend_Only
- Loose_RSI
- Breakout_60D

To avoid look-ahead bias, strategy positions were shifted by one day.

This means that signals generated using today's closing data were applied from the next trading day.

Formula:
strategy_return = position.shift(1) * daily_return

---

## 3. Risk Metrics

Annualized Volatility:
Measures how much strategy returns fluctuate on a yearly basis.

Daily VaR 95:
Measures the 5th percentile of daily strategy returns. It estimates the downside return threshold under a 95% confidence level.

Daily Expected Shortfall 95:
Measures the average return during the worst 5% of trading days. It is more conservative than VaR because it focuses on the severity of losses beyond the VaR threshold.

Market Exposure:
Measures the percentage of time the strategy was invested in the market.

Buy and Hold has 100% exposure because it is always invested.

Conditional strategies such as Trend_Only, Loose_RSI, and Breakout_60D have lower exposure because they only enter the market when specific rules are satisfied.

---

## 4. Main Findings

Buy and Hold generally produced higher total returns, but it also carried higher volatility and downside risk.

Breakout_60D had much lower market exposure across assets, which reduced Daily VaR, Expected Shortfall, and Annualized Volatility.

However, this lower risk came with lower upside participation.

This means Breakout_60D behaves more like a risk-management overlay than a return-maximizing strategy.

---

## 5. NVDA Outlier Handling

NVDA showed much higher return and volatility than the other assets.

As a result, it acted as an outlier in the full risk-return scatter plot.

To improve readability, two scatter plots were created:
- Full risk-return scatter plot including NVDA
- Zoomed risk-return scatter plot excluding NVDA

NVDA was not removed from the actual analysis. It was only excluded from the zoomed chart for visualization clarity.

---

## 6. Key Research Insight

The key insight is that reducing market exposure can significantly reduce downside risk, but it can also reduce return potential.

Therefore, strategy evaluation should not rely only on total return.

A stronger evaluation should compare return, volatility, maximum drawdown, VaR, Expected Shortfall, and market exposure together.

---

## 7. Summary

I extended the backtesting framework by adding risk management metrics such as Annualized Volatility, Historical VaR, Expected Shortfall, Maximum Drawdown, and Market Exposure.

The results showed that Breakout_60D reduced downside risk mainly by lowering market exposure, while Buy and Hold delivered higher returns with higher volatility.

I also created both full and zoomed risk-return scatter plots to handle NVDA as a high-return, high-volatility outlier without removing it from the actual analysis.
