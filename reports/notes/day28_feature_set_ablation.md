# Day 28 — Feature Set Ablation Test

## Objective

The goal of Day 28 was to test whether the ExtraTrees model really needed all seven input features, or whether a smaller feature set could maintain similar performance with less noise.

This experiment was designed after the Day 26 feature importance analysis showed that `ma60_ratio`, `volatility_20d`, and `rsi_14` were the most important variables.

## Feature Sets Tested

```text
Feature Set   Description
Full          All seven original features
Top3          ma60_ratio, volatility_20d, rsi_14
RiskAware     ma60_ratio, volatility_20d, rsi_14, ma20_ratio
Trend         ma10_ratio, ma20_ratio, ma60_ratio, return_20d
Momentum      return_5d, return_20d, ma10_ratio, ma20_ratio
```

## Portfolio-Level Results: Robust Assets

```text
Feature Set    Return    Volatility   Sharpe   MDD       Calmar
Buy & Hold     54.25%     20.63%       1.46   -22.17%     2.45
Full           38.17%      9.57%       2.24   -10.77%     3.55
Top3           14.32%      5.25%       1.68    -3.65%     3.92
RiskAware      11.06%      5.17%       1.34    -3.65%     3.03
Trend          42.46%     17.86%       1.37   -19.18%     2.21
Momentum       52.42%     20.62%       1.43   -22.17%     2.36
```

## Interpretation

The Full feature set remained the strongest main model candidate because it produced the best balance between return, Sharpe ratio, and drawdown control.

The Top3 feature set significantly reduced volatility and maximum drawdown, but it sacrificed too much return. This makes it useful as a defensive alternative, but not as the main strategy.

The Momentum feature set produced high returns, but its volatility and maximum drawdown were close to Buy & Hold. This suggests that momentum-only features were too aggressive and did not provide enough risk control.

## Key Takeaway

```text
Full Features = balanced main model
Top3 Features = defensive version
Momentum Features = aggressive but risky version
```

The result suggests that using all seven features allows the model to combine short-term momentum, medium-term trend, volatility, and RSI-based market condition indicators.

## Research Decision

Based on this test, I decided to keep the Full feature set as the main model input because it provided the best balance between return generation and risk control.

## Summary

After feature importance analysis, I ran a feature-set ablation test to check whether the most important variables alone could maintain performance. The Top3 feature set significantly reduced drawdown and volatility, but it also reduced return too much. Therefore, I decided to keep the Full feature set as the main model because it provided the best balance between return, Sharpe ratio, and drawdown control.
