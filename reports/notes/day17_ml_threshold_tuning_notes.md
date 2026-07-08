# Day 17 — ML Probability Threshold Tuning

## Objective

The objective of this experiment was to improve the ML-based trading signal by testing different probability thresholds.

In the previous ML signal backtest, the default classification threshold of 0.50 produced conservative signals. Several assets had low market exposure, which caused the strategy to miss a large part of the 2024 equity rally.

## Methodology

The RandomForest model produced a predicted probability for each asset and date.

This probability represents the model's confidence that the asset would generate a forward 5-trading-day return greater than 1%.

Different buy thresholds were tested:

- 0.40
- 0.45
- 0.50
- 0.55
- 0.60

The strategy bought the asset when predicted_probability was greater than or equal to the selected threshold.

To avoid look-ahead bias, the position was shifted by one day before calculating strategy returns.

## Key Results

### D05.SI

The 0.40 threshold improved the strategy return from 3.31% at the 0.50 threshold to 29.27%.

However, Buy and Hold still performed better because DBS had a strong upward move in 2024.

### MSFT

The 0.40 threshold produced the best result.

- Strategy Return: 32.35%
- Benchmark Return: 18.60%
- Market Exposure: 88.57%

This shows that the ML signal added value for MSFT after threshold tuning.

### NVDA

The 0.40 threshold produced a return of 175.41%, close to the Buy and Hold benchmark return of 179.73%.

This happened because NVDA was in a strong momentum-driven rally in 2024, and the lower threshold increased market exposure to almost full participation.

### QQQ

The 0.45 threshold produced strong results.

- Strategy Return: 34.15%
- Benchmark Return: 29.39%
- Market Exposure: 58.78%

The strategy outperformed Buy and Hold while using lower market exposure.

### SPY

The 0.40 threshold significantly improved the result.

- Strategy Return: 28.86%
- Benchmark Return: 26.68%
- Market Exposure: 63.67%

The original 0.50 threshold was too conservative for SPY, causing the model to miss much of the 2024 rally.

## Interpretation

The default 0.50 classification threshold was too conservative for a strong equity market environment.

Lowering the threshold increased market exposure, allowing the strategy to participate more in the 2024 rally.

This was especially helpful for SPY, QQQ, MSFT, and NVDA.

However, a lower threshold may not always work in bearish or sideways markets because it increases the number of buy signals.

## Main Insight

The model itself was not necessarily useless. The main issue was that the default decision threshold was too strict for a strong upward market regime.

Threshold tuning improved the ML signal by increasing market exposure and making the strategy more competitive against Buy and Hold.

## Limitations

- The result is tested only on the 2024 out-of-sample period.
- A 0.40 threshold may overfit the 2024 bull market.
- Different assets may require different thresholds.
- The model does not yet include macroeconomic, fundamental, or sentiment features.
- Further validation through walk-forward testing is required.

## Future Improvements

- Test asset-specific probability thresholds.
- Add walk-forward validation.
- Add macro features such as VIX, interest rates, and USD/SGD.
- Add fundamental features for DBS and MSFT.
- Add LLM-based news sentiment signals.
- Compare RandomForest with XGBoost and LightGBM.

## Summary

I found that the default 0.50 classification threshold was too conservative for the 2024 equity rally. By testing thresholds from 0.40 to 0.60, I showed that lower thresholds increased market exposure and improved performance across several assets. The tuned ML strategy outperformed Buy and Hold on MSFT, QQQ, and SPY, while NVDA nearly matched Buy and Hold due to high market participation.