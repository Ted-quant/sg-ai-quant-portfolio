# Day 18 — Walk-Forward Threshold Validation

## Objective

The objective of this experiment was to check whether the ML probability threshold results were robust across different out-of-sample years.

In Day 17, lower probability thresholds such as 0.40 and 0.45 improved the ML signal in the 2024 test period. However, this could have been specific to the 2024 bull market.

To reduce the risk of overfitting, I performed a walk-forward validation.

## Methodology

Two walk-forward windows were tested:

1. Train on 2022, test on 2023
2. Train on 2022–2023, test on 2024

For each test year, the RandomForest model generated predicted probabilities for each asset and date.

The strategy bought the asset when the predicted probability was greater than or equal to a selected threshold.

The tested thresholds were:

- 0.40
- 0.45
- 0.50
- 0.55
- 0.60

To avoid look-ahead bias, the position was shifted by one day before calculating strategy returns.

## Best Threshold Results

| Test Year | Asset | Best Threshold | Strategy Return | Benchmark Return | Strategy Sharpe | Market Exposure |
|---|---|---:|---:|---:|---:|---:|
| 2023 | D05.SI | 0.45 | 10.34% | 5.75% | 0.97 | 56.85% |
| 2023 | MSFT | 0.45 | 20.24% | 58.35% | 1.61 | 19.68% |
| 2023 | NVDA | 0.40 | 142.72% | 246.10% | 2.45 | 44.98% |
| 2023 | QQQ | 0.40 | 17.92% | 55.91% | 1.46 | 35.34% |
| 2023 | SPY | 0.40 | 25.52% | 26.71% | 2.21 | 59.84% |
| 2024 | D05.SI | 0.40 | 39.06% | 50.08% | 2.33 | 66.12% |
| 2024 | MSFT | 0.40 | 29.29% | 18.60% | 1.62 | 88.16% |
| 2024 | NVDA | 0.45 | 202.58% | 179.73% | 2.45 | 93.88% |
| 2024 | QQQ | 0.40 | 34.24% | 29.39% | 1.92 | 81.22% |
| 2024 | SPY | 0.40 | 29.70% | 26.68% | 2.77 | 61.22% |

## Interpretation

The walk-forward validation showed that lower probability thresholds, especially 0.40 and 0.45, were generally the best-performing thresholds across both 2023 and 2024.

This confirms that the default 0.50 classification threshold was often too conservative for the tested market environment.

Lower thresholds increased market exposure, allowing the strategy to participate more in upward market moves.

## 2023 Analysis

In 2023, the ML strategy outperformed Buy and Hold on D05.SI and came close to Buy and Hold on SPY.

However, it underperformed Buy and Hold on MSFT, NVDA, and QQQ.

This suggests that the ML signal was still too conservative for strong momentum-driven technology stocks in 2023.

## 2024 Analysis

In 2024, the ML strategy showed stronger results.

The tuned strategy outperformed Buy and Hold on MSFT, NVDA, QQQ, and SPY.

D05.SI still underperformed Buy and Hold, but the strategy return improved significantly compared with the original 0.50 threshold.

This suggests that threshold tuning helped the model capture more of the 2024 equity rally.

## Main Insight

The lower threshold result was not limited to a single test year.

Across both 2023 and 2024, the best threshold was usually 0.40 or 0.45.

This suggests that the model's probability scores were conservative, and lowering the decision threshold improved market participation.

## Limitations

- 2023 and 2024 were both generally favorable years for equities, especially US technology stocks.
- The strategy still underperformed Buy and Hold on several strong 2023 momentum assets.
- A lower threshold may perform worse in bearish or sideways markets.
- The model does not yet include macroeconomic, fundamental, or news sentiment features.
- Further testing on 2025–2026 data is needed as a fresh out-of-sample validation period.

## Future Improvements

- Add 2025–2026 as a final out-of-sample test period.
- Test asset-specific thresholds.
- Add market regime filters.
- Add macro features such as VIX, interest rates, and USD/SGD.
- Add fundamental features for DBS and MSFT.
- Add LLM-based news sentiment signals.
- Compare RandomForest with XGBoost and LightGBM.

## Summary

I performed a walk-forward validation by training the model on 2022 and testing on 2023, then training on 2022–2023 and testing on 2024. The results showed that lower thresholds around 0.40–0.45 generally performed best across both years. This suggests that the default 0.50 threshold was too conservative. However, the strategy still underperformed some strong 2023 momentum assets, so further regime-aware validation is needed.
