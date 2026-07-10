# Day 20 — Asset-Specific Threshold + Regime Filter Test

## Objective

The objective of this experiment was to test whether adding a 200-day moving average regime filter could improve the ML strategy's downside risk control during the 2025–2026 final out-of-sample period.

In Day 19, the final OOS test showed that the best probability threshold differed by asset. This suggested that one common threshold may not work equally well across different assets.

Therefore, Day 20 tested two strategies:

1. Asset-specific threshold only
2. Asset-specific threshold + 200-day moving average regime filter

## Methodology

The RandomForest model was trained on 2022–2024 data and tested on the 2025–2026 out-of-sample period.

The asset-specific thresholds were taken from the Day 19 final OOS test:

- D05_SI: 0.40
- MSFT: 0.50
- NVDA: 0.55
- QQQ: 0.55
- SPY: 0.40

The threshold-only strategy bought the asset when:

predicted_probability >= asset_specific_threshold

The threshold + regime filter strategy bought the asset only when:

predicted_probability >= asset_specific_threshold
and
Close > 200-day moving average

To avoid look-ahead bias, positions were shifted by one day before calculating strategy returns.

## Results

| Asset | Strategy Type | Return | Benchmark Return | Sharpe | Benchmark Sharpe | MDD | Benchmark MDD | Market Exposure |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D05_SI | Threshold Only | 53.03% | 65.79% | 1.72 | 1.95 | -19.10% | -19.26% | 87.80% |
| D05_SI | Threshold + Regime Filter | 49.90% | 65.79% | 1.81 | 1.95 | -14.49% | -19.26% | 85.94% |
| MSFT | Threshold Only | -7.61% | -7.75% | -0.09 | -0.05 | -26.66% | -34.50% | 55.61% |
| MSFT | Threshold + Regime Filter | -6.08% | -7.75% | -0.41 | -0.05 | -12.07% | -34.50% | 15.24% |
| NVDA | Threshold Only | 68.26% | 47.18% | 1.23 | 0.80 | -23.92% | -36.88% | 37.17% |
| NVDA | Threshold + Regime Filter | 30.22% | 47.18% | 1.08 | 0.80 | -8.37% | -36.88% | 21.39% |
| QQQ | Threshold Only | 8.99% | 42.90% | 0.42 | 1.17 | -17.00% | -22.77% | 39.84% |
| QQQ | Threshold + Regime Filter | 7.03% | 42.90% | 0.57 | 1.17 | -6.00% | -22.77% | 27.81% |
| SPY | Threshold Only | 20.67% | 29.42% | 0.82 | 1.06 | -18.76% | -18.76% | 81.55% |
| SPY | Threshold + Regime Filter | 12.61% | 29.42% | 0.80 | 1.06 | -10.00% | -18.76% | 67.65% |

## Interpretation

The 200-day moving average regime filter generally reduced downside risk and volatility.

However, it did not consistently improve total return.

This means the regime filter acted more as a risk control tool than a return enhancement tool.

## Asset-Level Analysis

### D05_SI

The regime filter reduced maximum drawdown from -19.10% to -14.49%.

Total return decreased slightly from 53.03% to 49.90%.

This suggests that the regime filter improved risk control, but the asset still benefited from staying invested for much of the period.

### MSFT

The regime filter improved both total return and drawdown.

The threshold-only strategy returned -7.61%, while the regime-filtered strategy returned -6.08%.

Maximum drawdown improved significantly from -26.66% to -12.07%.

This suggests that the regime filter helped avoid weak trend periods for MSFT.

### NVDA

NVDA had the strongest threshold-only result.

The threshold-only strategy returned 68.26%, outperforming the benchmark return of 47.18%.

However, adding the regime filter reduced return to 30.22%.

The benefit was much lower drawdown: -8.37% compared with -23.92% for threshold-only and -36.88% for Buy and Hold.

This suggests that the regime filter made NVDA more defensive, but at the cost of missing part of the upside.

### QQQ

The regime filter reduced maximum drawdown from -17.00% to -6.00%.

Total return decreased slightly from 8.99% to 7.03%.

This suggests that the filter improved downside protection but did not solve the weak total return problem for QQQ.

### SPY

The regime filter reduced maximum drawdown from -18.76% to -10.00%.

However, total return decreased from 20.67% to 12.61%.

This suggests that the regime filter reduced risk but also reduced participation in broad market recoveries.

## Main Insight

The main finding is that the 200-day moving average regime filter is useful for reducing drawdown and volatility, but it is not a universal return booster.

The filter worked best as a defensive risk management layer.

The ML strategy should therefore be framed as a selective exposure and risk control system rather than a pure return-maximizing model.

## Limitations

- The 200-day moving average filter may react slowly to market recoveries.
- The filter can reduce upside participation after temporary drawdowns.
- The model still uses only technical indicators.
- Transaction costs and slippage are not yet included.
- The regime filter is simple and may need to be compared with alternatives such as VIX-based filters or shorter moving averages.

## Future Improvements

- Compare 100-day, 150-day, and 200-day moving average regime filters.
- Add VIX-based market regime filters.
- Add transaction costs and slippage.
- Test asset-specific regime filters.
- Compare RandomForest with XGBoost and LightGBM.
- Add macroeconomic and news sentiment features.

## Summary

The Day 20 experiment showed that adding a 200-day moving average regime filter reduced drawdown and volatility across most assets, but often lowered total return. This suggests that the regime filter is useful as a defensive risk management layer rather than a return enhancement tool.
