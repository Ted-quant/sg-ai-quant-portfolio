# Day 24 — Inverse Volatility Portfolio Weighting

## Objective

The objective of this experiment was to compare equal-weight portfolio construction with inverse volatility weighting.

In Day 23, the Robust Assets ML Portfolio showed strong risk-adjusted performance using equal weighting across D05_SI, NVDA, and SPY.

Day 24 tested whether assigning lower weights to more volatile assets could further improve portfolio risk-adjusted performance.

## Methodology

The portfolio used the Robust Assets universe:

- D05_SI
- NVDA
- SPY

The ML strategy used:

- Asset-specific probability thresholds from Day 19
- Asset-specific regime filters from Day 21
- Transaction costs and slippage from Day 22

Two allocation methods were compared:

1. Equal Weight
2. Inverse Volatility Weighting

Inverse volatility weights were calculated using a 60-day rolling annualized volatility estimate.

The idea was to allocate lower weights to more volatile assets and higher weights to less volatile assets.

## Average Inverse Volatility Weights

```text
Asset    Average Weight
D05_SI      39.51%
NVDA        16.62%
SPY         43.87%
```

The weighting logic worked as expected. NVDA received the lowest average weight because it had the highest volatility, while SPY and D05_SI received higher weights.

## Results

```text
Portfolio                     Total Return   Volatility   Sharpe   MDD       Calmar   VaR 95%   ES 95%
Equal Weight ML Net             25.09%         8.81%       2.13    -2.53%     9.90    -0.61%    -1.15%
Equal Weight Buy & Hold         49.37%        19.25%       1.81   -13.59%     3.63    -1.80%    -2.63%
Inverse Vol ML Net              24.24%         7.95%       2.28    -4.04%     6.00    -0.63%    -1.02%
Inverse Vol Buy & Hold          46.26%        14.86%       2.18   -13.35%     3.47    -1.28%    -2.17%
```

## Interpretation

Inverse volatility weighting worked as expected by reducing exposure to the most volatile asset, NVDA.

It reduced annualized volatility from 8.81% to 7.95% and improved Sharpe ratio from 2.13 to 2.28.

However, equal weighting produced better total return, lower maximum drawdown, and a higher Calmar ratio.

This means that inverse volatility weighting improved average volatility control, but did not improve drawdown control in this test.

## Main Insight

Inverse volatility weighting is not automatically better than equal weighting.

In this test, equal weighting was the stronger final portfolio candidate because it achieved:

- Higher total return
- Lower maximum drawdown
- Higher Calmar ratio

The best interpretation is that inverse volatility weighting made the portfolio smoother, but equal weighting captured a better return-drawdown balance.

## Limitations

- Inverse volatility weights were based only on rolling historical volatility.
- The method did not account for correlations between assets.
- The result may be sensitive to the 60-day volatility lookback window.
- Transaction costs from rebalancing weights were not explicitly modeled.
- The test used only the Robust Assets universe.

## Summary

The Day 24 experiment showed that inverse volatility weighting reduced volatility and improved Sharpe ratio, but equal weighting produced the better overall return-drawdown profile. The Robust Assets Equal Weight ML Portfolio remains the strongest final candidate at this stage.
