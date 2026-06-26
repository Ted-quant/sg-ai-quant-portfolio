# Day 10 Breakout Strategy Notes

## 1. Objective

The objective of this experiment was to add a 60-day breakout strategy and compare it with the existing rule-based strategies.

The breakout strategy was added to test whether resistance-breakout signals can improve risk-adjusted performance by entering the market only during strong upward price movements.

---

## 2. Strategy Definition

### Breakout 60D

The Breakout_60D strategy buys when:

```text
Today's Close > Yesterday's 60-day high
```

This means the model enters a position only when the current closing price breaks above the highest closing price from the previous 60 trading days.

To avoid look-ahead bias, the 60-day high was shifted by one day:

```python
df["rolling_60d_high"] = df["Close"].rolling(window=60).max().shift(1)
```

This ensures that the strategy only uses information that would have been available before today's close.

---

## 3. Key Results

The Breakout_60D strategy generally produced lower total returns than Buy and Hold.

However, it significantly reduced maximum drawdown across several assets.

For example:

* NVDA Buy and Hold MDD: -59.43%

* NVDA Breakout_60D MDD: -15.28%

* SPY Buy and Hold MDD: -21.61%

* SPY Breakout_60D MDD: -3.10%

* QQQ Buy and Hold MDD: -29.20%

* QQQ Breakout_60D MDD: -3.43%

* D05_SI Buy and Hold MDD: -16.61%

* D05_SI Breakout_60D MDD: -3.05%

---

## 4. Interpretation

The Breakout_60D strategy was highly selective. It did not stay invested at all times, unlike Buy and Hold.

As a result, it missed some upside during strong bull markets, especially in NVDA.

However, the strategy avoided many large drawdowns because it only entered the market after a strong breakout signal.

This made the strategy more defensive and improved risk-adjusted performance for some assets.

---

## 5. Asset-Level Insights

### NVDA

For NVDA, Buy and Hold and Trend Only performed better than Breakout_60D in total return.

This suggests that NVDA had strong long-term momentum, and a highly selective breakout strategy missed too much upside.

### SPY

For SPY, Breakout_60D produced a much lower maximum drawdown and a higher Sharpe ratio than Buy and Hold.

This suggests that selective entry based on breakout signals helped reduce downside exposure.

### QQQ

QQQ showed the strongest improvement in risk-adjusted performance.

Breakout_60D had lower total return than Buy and Hold, but it achieved a much higher Sharpe ratio and significantly lower maximum drawdown.

### D05_SI

For D05_SI, Breakout_60D reduced maximum drawdown substantially, although total return was lower than Buy and Hold.

This suggests that breakout-based entry may be useful as a defensive strategy for Singapore bank exposure.

---

## 6. Main Research Insight

The Breakout_60D strategy is not a return-maximizing strategy.

Instead, it behaves more like a risk-management strategy.

It reduces downside exposure by entering only after strong price breakouts, which can improve Sharpe ratio for some assets such as SPY and QQQ.

---

## 7. Sumamry

The 60-day breakout strategy produced lower total returns than Buy and Hold, but it significantly reduced maximum drawdown across several assets. For SPY and QQQ, the breakout strategy improved Sharpe ratio, suggesting that selective entry based on resistance breakout can improve risk-adjusted performance by reducing downside exposure.
