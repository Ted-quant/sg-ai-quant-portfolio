# Day 19 — Final Out-of-Sample Test: 2025–2026

## Objective

The objective of this experiment was to test whether the probability thresholds selected during the 2023–2024 walk-forward validation period remained effective on a fresh 2025–2026 out-of-sample period.

This final out-of-sample test is important because the 2023–2024 validation period was generally favorable for equities, especially US technology stocks.

## Methodology

The RandomForest model was trained on data from 2022 to 2024.

The model was then tested on the 2025–2026 out-of-sample period.

The model predicts the probability that the forward 5-trading-day return will exceed 1%.

The tested probability thresholds were:

- 0.40
- 0.45
- 0.50
- 0.55
- 0.60

The trading signal bought the asset when the predicted probability was greater than or equal to the selected threshold.

To avoid look-ahead bias, the position was shifted by one day before calculating strategy returns.

## Best Threshold Results

```text
Asset    Best Threshold   Strategy Return   Benchmark Return   Excess Return   Strategy Sharpe   Benchmark Sharpe   Strategy MDD   Benchmark MDD   Market Exposure
D05_SI        0.40             42.86%            63.30%           -20.44%             1.45              1.90          -19.61%        -19.26%          89.36%
MSFT          0.50             -0.01%           -10.45%            10.44%             0.10             -0.13          -20.64%        -34.50%          51.74%
NVDA          0.55             54.69%            49.05%             5.64%             1.24              0.82          -10.07%        -36.88%          31.64%
QQQ           0.55             36.20%            45.12%            -8.92%             1.49              1.22           -5.70%        -22.77%          40.21%
SPY           0.40             15.45%            29.60%           -14.15%             0.65              1.07          -18.76%        -18.76%          80.70%
```

## Interpretation

The final out-of-sample results were mixed.

The lower thresholds around 0.40–0.45 did not generalize perfectly across all assets in the 2025–2026 period.

In the 2023–2024 walk-forward validation, the best thresholds were usually 0.40 or 0.45. However, in the final 2025–2026 out-of-sample test, MSFT performed best at 0.50, while NVDA and QQQ performed best at 0.55.

This suggests that a single common threshold may not be optimal across different assets and market regimes.

## Asset-Level Analysis

### D05_SI

The ML strategy generated a strong positive return of 42.86%, but it underperformed the Buy and Hold benchmark return of 63.30%.

The market exposure was very high at 89.36%, meaning the strategy behaved close to Buy and Hold.

### MSFT

The ML strategy returned approximately 0%, while Buy and Hold returned -10.45%.

This suggests that the ML signal helped reduce downside exposure during a weak period for MSFT.

### NVDA

NVDA was the strongest result.

The ML strategy returned 54.69%, outperforming the Buy and Hold benchmark return of 49.05%.

The strategy also reduced maximum drawdown from -36.88% to -10.07%, while using only 31.64% market exposure.

This indicates that the ML signal added value for NVDA on a risk-adjusted basis.

### QQQ

The ML strategy underperformed Buy and Hold in total return, but it significantly reduced maximum drawdown.

The strategy MDD was -5.70%, compared with -22.77% for Buy and Hold.

This suggests that the ML signal may be useful as a downside-risk control tool, even when it does not maximize total return.

### SPY

The SPY strategy underperformed Buy and Hold.

The market exposure was high at 80.70%, but the strategy still failed to match the benchmark return.

This suggests that the current feature set may not be strong enough for broad market ETF timing.

## Main Insight

The final OOS test showed that the low-threshold result from 2023–2024 was not fully robust across the 2025–2026 period.

However, the ML strategy still showed useful behavior in several cases:

- NVDA: higher return, lower drawdown, and lower exposure than Buy and Hold
- MSFT: reduced losses compared with Buy and Hold
- QQQ: lower drawdown despite lower total return

The key insight is that the ML model may be more useful as a selective exposure and risk control tool than as a simple return-maximizing trading signal.

## Limitations

- The 2025–2026 test period may be incomplete depending on the latest available market data.
- A single probability threshold may not work equally well across all assets.
- The model still uses only technical indicators.
- The current model does not include macroeconomic, fundamental, or news sentiment features.
- Transaction costs and slippage are not yet included.
- The model may require market regime filters to avoid overexposure in weak environments.

## Future Improvements

- Test asset-specific probability thresholds.
- Add market regime filters such as moving-average trend filters or VIX-based filters.
- Add macro features such as VIX, interest rates, and USD/SGD.
- Add fundamental features for DBS and MSFT.
- Add LLM-based news sentiment signals.
- Compare RandomForest with XGBoost and LightGBM.
- Add transaction cost and slippage assumptions.

## Summary

The 2025–2026 final out-of-sample test showed mixed results. The lower thresholds that worked well in 2023–2024 did not generalize perfectly to all assets. However, the ML strategy outperformed Buy and Hold on NVDA and reduced downside risk on MSFT and QQQ. This suggests that the model may be more useful as a selective exposure and risk management tool than as a universal return-maximizing signal.
