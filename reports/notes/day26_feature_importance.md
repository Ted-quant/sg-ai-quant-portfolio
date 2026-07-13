# Day 26 — Feature Importance & Model Explainability

## Objective

The goal of Day 26 was to understand why the ExtraTrees model performed better than the previous RandomForest baseline.

After Day 25 identified ExtraTrees as the strongest model candidate, I analyzed feature importance to see which technical indicators contributed most to the model's trading decisions.

## Model Used

```text
Model: ExtraTreesClassifier
Target: Forward 5-day return > +1%
Assets: D05_SI, MSFT, NVDA, QQQ, SPY
Training period: 2020-2024
Test period: 2025-2026
```

## Features Tested

```text
Feature          Meaning
return_5d        Recent 5-day return
return_20d       Recent 20-day return
ma10_ratio       Current price relative to 10-day moving average
ma20_ratio       Current price relative to 20-day moving average
ma60_ratio       Current price relative to 60-day moving average
volatility_20d   Recent 20-day volatility
rsi_14           14-day RSI indicator
```

## Average Feature Importance

```text
Feature           Importance   Rank
ma60_ratio          0.210961     1
volatility_20d      0.186665     2
rsi_14              0.185380     3
ma20_ratio          0.121848     4
return_20d          0.113009     5
ma10_ratio          0.097598     6
return_5d           0.084539     7
```

## Interpretation

The most important feature was `ma60_ratio`, which measures where the current price is relative to its 60-day moving average.

This suggests that the ExtraTrees model relied heavily on medium-term trend information.

The second most important feature was `volatility_20d`, meaning the model also considered recent risk and price instability.

The third most important feature was `rsi_14`, showing that overbought and oversold conditions also influenced the model's decisions.

## Main Insight

The model did not rely mainly on short-term price movement.

Instead, the model focused more on:

```text
1. Medium-term trend
2. Recent volatility
3. RSI-based overbought or oversold conditions
```

This suggests that the ExtraTrees model behaved more like a risk-aware trend-following model than a simple short-term momentum model.

## Asset-Level Observations

```text
Asset    Most Important Feature   Interpretation
D05_SI   ma60_ratio               Medium-term trend was most important
MSFT     ma60_ratio               Medium-term trend was most important
NVDA     volatility_20d           Volatility was most important due to high price swings
QQQ      ma60_ratio               Trend and RSI were important
SPY      rsi_14                   RSI was most important for broad market timing
```

## Key Takeaway

The ExtraTrees model's strongest signals came from medium-term trend, volatility, and RSI.

This is useful because the model can now be explained as more than a black-box classifier. It used financially meaningful indicators to generate trading signals.

## Summary

After identifying ExtraTrees as the strongest model candidate, I analyzed feature importance to understand what the model was using to make trading decisions. The most important features were the 60-day moving-average ratio, 20-day volatility, and RSI. This suggests that the model was not simply chasing short-term returns, but was combining trend, risk, and overbought or oversold information.



