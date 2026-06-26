# Day 9 Strategy Experiment Notes

## 1. Objective

The objective of this experiment was to compare multiple rule-based trading strategies across different assets and evaluate whether the original defensive strategy was too restrictive.

The tested assets were:

* NVDA
* MSFT
* SPY
* QQQ
* D05_SI

The tested strategies were:

* Buy and Hold
* Current Defensive
* Trend Only
* Loose RSI

---

## 2. Strategy Definitions

### Buy and Hold

This strategy stays invested at all times.

### Current Defensive

This strategy buys only when:

```text
MA20 > MA60 and RSI < 70
```

This means the model only enters a position when the short-term trend is stronger than the long-term trend and RSI is below the traditional overbought threshold of 70.

### Trend Only

This strategy buys when:

```text
MA20 > MA60
```

This removes the RSI restriction and focuses only on trend-following.

### Loose RSI

This strategy buys when:

```text
MA20 > MA60 and RSI < 80
```

This keeps the RSI filter but relaxes the threshold from 70 to 80, allowing the strategy to stay invested longer during strong upward trends.

---

## 3. Key Results

The experiment showed that Buy and Hold delivered the highest total return for most assets, especially NVDA.

However, among the active rule-based strategies, Trend Only and Loose RSI generally improved performance compared with the original Current Defensive strategy.

For NVDA, Trend Only significantly improved total return compared with Current Defensive. This suggests that the original RSI < 70 filter was too restrictive for a strong momentum asset.

For SPY and QQQ, Loose RSI performed better than Current Defensive, suggesting that relaxing the RSI threshold from 70 to 80 allowed the strategy to capture more upside while still keeping some overbought filter.

---

## 4. Interpretation

The original Current Defensive strategy reduced downside risk but sacrificed too much upside participation.

The results suggest that RSI should not be applied too rigidly across all assets. In strong momentum assets such as NVDA, high RSI may reflect continued buying pressure rather than an immediate reversal signal.

Therefore, strategy rules should be adapted based on asset characteristics rather than applied uniformly across all markets.

---

## 5. Main Research Insight

The initial defensive rule was too conservative. Removing or relaxing the RSI filter improved performance across several assets, especially for high-momentum assets and index ETFs.

This supports the idea that technical indicators should be tested empirically rather than assumed to work in the same way across all assets.

---

## 6. Interview Summary

The strategy experiment showed that the original defensive rule was too restrictive. For NVDA, removing the RSI filter improved performance because the stock had strong upward momentum. For SPY and QQQ, relaxing the RSI threshold from 70 to 80 improved results, suggesting that a moderate RSI filter may work better than a strict defensive rule.
