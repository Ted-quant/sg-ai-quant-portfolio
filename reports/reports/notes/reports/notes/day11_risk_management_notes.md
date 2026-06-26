# Day 11 Risk Management Analysis Notes

## 1. Objective

The objective of this analysis was to evaluate the downside risk of each strategy across multiple assets.

Instead of only comparing total returns, this analysis focused on risk metrics that are commonly used in portfolio management and quantitative finance.

The main metrics were:

* Annualized Volatility
* Daily VaR 95%
* Daily Expected Shortfall 95%
* Maximum Drawdown
* Market Exposure
* Risk-Return Profile

---

## 2. Risk Metrics

### Annualized Volatility

Annualized volatility measures how much the strategy return fluctuates on a yearly basis.

It was calculated from daily strategy returns and annualized using the square root of 252 trading days.

### Daily VaR 95%

Daily VaR 95% estimates the downside return threshold at the 5th percentile of daily strategy returns.

For example, if Daily VaR 95% is -2%, it means that historically, the strategy had a 5% chance of losing more than 2% in one day.

### Daily Expected Shortfall 95%

Expected Shortfall measures the average loss during the worst 5% of trading days.

It is more conservative than VaR because it focuses on the severity of losses beyond the VaR threshold.

### Market Exposure

Market exposure measures the percentage of time that the strategy was invested in the market.

A market exposure of 100% means the strategy was always invested, while a lower exposure means the strategy spent more time in cash.

---

## 3. Main Results

Buy and Hold had the highest market exposure because it remained invested throughout the entire period.

Breakout_60D had much lower market exposure across all assets, usually around 12% to 16%.

This lower exposure significantly reduced downside risk metrics such as VaR, Expected Shortfall, and annualized volatility.

For example, NVDA Buy and Hold had much higher volatility and downside risk than the Breakout_60D strategy.

However, the lower risk came at the cost of lower total return.

---

## 4. Risk-Return Interpretation

The risk-return scatter plot showed that NVDA acted as a high-return and high-volatility outlier.

Because NVDA compressed the other assets into a smaller area of the chart, a zoomed scatter plot excluding NVDA was also created for visualization clarity.

NVDA was not removed from the actual analysis. It was only excluded from the zoomed chart to make the risk-return relationship of the other assets easier to interpret.

---

## 5. Strategy-Level Interpretation

### Buy and Hold

Buy and Hold generally produced higher returns but also carried higher volatility and downside risk.

This strategy is simple and fully exposed to market movements.

### Trend Only

Trend Only reduced market exposure compared with Buy and Hold while still maintaining meaningful upside participation.

It worked particularly well for strong momentum assets.

### Loose RSI

Loose RSI was a balanced strategy that reduced some downside risk while participating more than the strict defensive strategy.

### Current Defensive

Current Defensive was more conservative because it required both a trend condition and an RSI filter.

This reduced risk, but it also limited upside participation.

### Breakout_60D

Breakout_60D behaved more like a risk-management overlay than a return-maximizing strategy.

It entered the market only after strong breakout signals, which reduced downside exposure significantly.

However, because it stayed in cash most of the time, it also missed many upward moves.

---

## 6. Key Research Insight

The main insight is that lower market exposure can significantly reduce downside risk, but it can also reduce return potential.

Therefore, strategy performance should not be evaluated only by total return.

A complete evaluation should include return, volatility, drawdown, VaR, Expected Shortfall, and market exposure.

---

## 7. Summary

I extended the backtesting framework by adding risk management metrics such as annualized volatility, historical VaR, Expected Shortfall, maximum drawdown, and market exposure. The analysis showed that Breakout_60D reduced downside risk mainly by lowering market exposure, while Buy and Hold delivered higher returns with higher volatility. I also created both full and zoomed risk-return scatter plots to handle NVDA as a high-return, high-volatility outlier without removing it from the actual analysis.
