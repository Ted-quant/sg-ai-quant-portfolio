# Day 21 — Regime Filter Sensitivity Test

## Objective

The objective of this experiment was to compare different moving average regime filters and identify which filter provided the best balance between return and downside risk during the 2025–2026 out-of-sample period.

In Day 20, the 200-day moving average regime filter reduced drawdown and volatility, but it often reduced total return by lowering market exposure too much.

Therefore, Day 21 compared four versions:

1. No filter
2. 100-day moving average filter
3. 150-day moving average filter
4. 200-day moving average filter

## Methodology

The RandomForest model was trained on 2022–2024 data and tested on the 2025–2026 out-of-sample period.

The asset-specific thresholds from Day 19 were used:

- D05_SI: 0.40
- MSFT: 0.50
- NVDA: 0.55
- QQQ: 0.55
- SPY: 0.40

The base ML signal was:

predicted_probability >= asset_specific_threshold

For each moving average filter, the final buy signal required:

predicted_probability >= asset_specific_threshold
and
Close > moving_average

The tested moving average windows were 100, 150, and 200 days.

To avoid look-ahead bias, positions were shifted by one day before calculating strategy returns.

## Selection Metric

The best filter was selected using the Calmar ratio.

Calmar ratio is calculated as:

strategy_return / absolute maximum drawdown

This metric was used because the goal of the regime filter was not only to maximize total return, but also to improve the balance between return and drawdown.

## Best Filter by Asset

| Asset | Best Filter | Return | Benchmark Return | Sharpe | MDD | Calmar Ratio | Market Exposure |
|---|---|---:|---:|---:|---:|---:|---:|
| D05_SI | MA100 | 49.22% | 66.62% | 2.06 | -7.73% | 6.37 | 79.37% |
| MSFT | No Filter | -7.61% | -7.75% | -0.09 | -26.66% | -0.29 | 55.61% |
| NVDA | MA200 | 30.22% | 47.18% | 1.08 | -8.37% | 3.61 | 21.39% |
| QQQ | MA100 | 10.80% | 42.90% | 0.98 | -4.73% | 2.28 | 22.99% |
| SPY | MA100 | 19.00% | 29.42% | 1.25 | -6.82% | 2.78 | 62.30% |

## Interpretation

The results showed that the 100-day moving average filter was the most effective general regime filter.

MA100 was selected as the best filter for D05_SI, QQQ, and SPY.

This suggests that the 200-day moving average filter may be too slow for several assets, causing the strategy to miss recovery periods.

However, NVDA performed best with the MA200 filter. This may be because NVDA is more volatile and benefits from a slower defensive trend filter.

MSFT did not show a clear improvement from moving average filters. This suggests that the current technical feature set may not be strong enough for MSFT, and additional features or model improvements may be needed.

## Asset-Level Analysis

### D05_SI

The MA100 filter reduced maximum drawdown from -19.10% to -7.73%.

Total return decreased from 53.79% to 49.22%, but the drawdown improvement was large enough to improve the Calmar ratio significantly.

This suggests that MA100 provided a better return-drawdown balance for D05_SI.

### MSFT

No filter was selected as the best option by Calmar ratio.

Although the MA200 filter reduced drawdown and slightly improved total return compared with no filter, the overall performance remained weak.

This suggests that MSFT may require better features, a different model, or a different threshold logic.

### NVDA

No filter produced the highest return at 68.26%, but maximum drawdown was -23.92%.

The MA200 filter reduced drawdown to -8.37%, while still generating a positive return of 30.22%.

This made MA200 the best filter for NVDA by Calmar ratio.

The result suggests that slower defensive filters may work better for high-volatility assets.

### QQQ

MA100 improved both total return and drawdown compared with no filter.

Return increased from 8.99% to 10.80%, while maximum drawdown improved from -17.00% to -4.73%.

This was one of the clearest improvements from the regime filter sensitivity test.

### SPY

MA100 reduced maximum drawdown from -18.76% to -6.82%.

Total return decreased slightly from 20.67% to 19.00%, but the improvement in drawdown and Sharpe ratio made MA100 the best filter by Calmar ratio.

This suggests that MA100 may be more suitable than MA200 for broad market ETF timing.

## Main Insight

The 200-day moving average filter was not always the best defensive filter.

The 100-day moving average filter produced a better balance between return and drawdown for D05_SI, QQQ, and SPY.

NVDA was the exception, where the 200-day moving average filter provided the best risk-adjusted result.

This suggests that regime filters should be asset-specific rather than fixed across all assets.

## Limitations

- The result is still based on the 2025–2026 out-of-sample period only.
- Moving average filters are simple trend filters and may react late to sudden market changes.
- Transaction costs and slippage are still not included.
- The model still uses only technical indicators.
- The regime filter was selected using Calmar ratio, but other selection criteria may lead to different choices.
- Asset-specific filter selection may introduce overfitting if not validated further.

## Future Improvements

- Validate asset-specific regime filters using additional walk-forward windows.
- Add transaction costs and slippage.
- Compare moving average filters with VIX-based regime filters.
- Add macroeconomic features such as VIX, interest rates, and USD/SGD.
- Compare RandomForest with XGBoost and LightGBM.
- Test whether regime filter selection should be based on Calmar ratio, Sharpe ratio, or drawdown constraint.

## Summary

The Day 21 experiment showed that MA100 was the strongest general regime filter across D05_SI, QQQ, and SPY, while MA200 worked best for NVDA. This suggests that 200MA may be too slow for some assets, and that regime filters should be selected based on asset behavior and return-drawdown balance rather than applied uniformly.
