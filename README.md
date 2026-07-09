# 🇸🇬 SG AI Quant Portfolio

AI-powered quantitative trading strategy for Singapore and US equity markets.

## Overview

This project builds a full quantitative finance pipeline from data collection to strategy backtesting, risk analysis, machine learning signal generation, and portfolio reporting.

The project is designed as a job portfolio for Singapore quantitative finance and financial analytics roles.

## Tech Stack

- Language: Python 3.11
- Backtesting: QuantConnect (LEAN), vectorbt
- Data: yfinance (SGX + US), Alpaca Markets API
- ML/AI: scikit-learn, XGBoost, LightGBM
- LLM Signal: Claude API (Anthropic)
- Version Control: GitHub

## Target Markets

| Market | Assets |
|---|---|
| SGX Singapore | DBS Bank (D05.SI / D05_SI), Sea Limited, Singtel |
| US Equities | SPY, QQQ, NVDA, MSFT |
| Asia ETFs | EWS, EWJ, EWY |
| FX / Macro | USD/SGD |

## Project Structure

- data/: SGX, US, FX, and news data
- strategies/: trading strategy modules
- models/: ML and LLM signal models
- backtest/: backtesting scripts
- reports/: charts and performance reports
- notebooks/: research and EDA

## Author

Chae Youngjun | Quant Finance Portfolio | 2026

---

## Recent Progress: Strategy and Risk Management

### Day 9-11 Summary

I expanded the project from basic backtesting into a multi-strategy risk management framework.

The current analysis compares 5 assets and 5 strategies.

### Assets

- NVDA
- MSFT
- SPY
- QQQ
- D05_SI

### Strategies

- Buy and Hold
- Current Defensive
- Trend Only
- Loose RSI
- 60-Day Breakout

### Key Risk Metrics

- Total Return
- Maximum Drawdown
- Sharpe Ratio
- Annualized Volatility
- Daily VaR 95%
- Daily Expected Shortfall 95%
- Market Exposure

### Main Insight

The 60-Day Breakout strategy reduced downside risk and volatility by lowering market exposure, but it also reduced upside participation compared with Buy and Hold and Trend Only.

### Generated Outputs

- backtest/results/strategy_experiment_summary.csv
- backtest/results/risk_management_summary.csv
- reports/charts/risk_var_95_comparison.png
- reports/charts/risk_expected_shortfall_comparison.png
- reports/charts/risk_annualized_volatility_comparison.png
- reports/charts/risk_market_exposure_comparison.png
- reports/charts/risk_return_scatter_full.png
- reports/charts/risk_return_scatter_zoomed_ex_nvda.png

### Summary

Built a standardized strategy comparison and risk management framework across SGX and US assets. The framework evaluates both return and downside risk using VaR, Expected Shortfall, volatility, drawdown, and market exposure.

---

## Day 12-16 — Vectorbt and ML Signal Development

Before threshold tuning, I expanded the project from manual backtesting to vectorbt-based validation and machine learning signal generation.

### Key Steps

- Built a vectorbt moving-average backtest for SPY.
- Expanded the vectorbt framework to multiple assets: NVDA, MSFT, SPY, QQQ, and D05.SI.
- Created a supervised ML dataset using momentum, moving-average ratios, volatility, and RSI features.
- Trained a RandomForest model to predict whether the forward 5-trading-day return would exceed 1%.
- Backtested ML-based buy signals against Buy and Hold benchmarks.

### Main Insight

The initial ML model achieved around 54% out-of-sample classification accuracy, but the trading performance depended heavily on market exposure and probability threshold selection.

This showed that classification accuracy alone is not enough to evaluate a trading model. The model must also be tested through return, drawdown, Sharpe ratio, and market exposure.

---

## Day 17 — ML Probability Threshold Tuning

After the initial ML signal backtest, I tested different probability thresholds to improve the trading signal.

The original RandomForest signal used the default 0.50 classification threshold. This made the model too conservative, especially for SPY, QQQ, and D05.SI, where market exposure was low during a strong 2024 equity rally.

I tested probability thresholds from 0.40 to 0.60.

### Key Finding

Lower thresholds increased market exposure and improved performance in several assets.

| Asset | Best Threshold | Strategy Return | Benchmark Return | Interpretation |
|---|---:|---:|---:|---|
| D05.SI | 0.40 | 29.27% | 50.08% | Improved exposure but still underperformed Buy and Hold |
| MSFT | 0.40 | 32.35% | 18.60% | ML signal outperformed Buy and Hold |
| NVDA | 0.40 | 175.41% | 179.73% | Nearly matched Buy and Hold due to high exposure |
| QQQ | 0.45 | 34.15% | 29.39% | Outperformed Buy and Hold with lower exposure |
| SPY | 0.40 | 28.86% | 26.68% | Outperformed Buy and Hold after exposure increased |

### Interpretation

The experiment showed that the ML model was not simply wrong. The default threshold was too strict for a strong upward market regime.

By lowering the threshold, the strategy participated more in the 2024 rally and became more competitive against Buy and Hold.

### Limitation

The best threshold may be specific to the 2024 bull market. Further walk-forward testing is needed to check whether the threshold remains effective in different market regimes.

---

## Day 18 — Walk-Forward Threshold Validation

I performed a walk-forward validation to test whether the ML probability threshold results were robust across different out-of-sample years.

### Walk-Forward Setup

| Train Period | Test Period |
|---|---|
| 2022 | 2023 |
| 2022-2023 | 2024 |

For each test year, I tested probability thresholds from 0.40 to 0.60.

### Key Finding

The best-performing thresholds were usually 0.40 or 0.45, suggesting that the default 0.50 classification threshold was too conservative.

| Test Year | Asset | Best Threshold | Strategy Return | Benchmark Return |
|---|---|---:|---:|---:|
| 2023 | D05.SI | 0.45 | 10.34% | 5.75% |
| 2023 | MSFT | 0.45 | 20.24% | 58.35% |
| 2023 | NVDA | 0.40 | 142.72% | 246.10% |
| 2023 | QQQ | 0.40 | 17.92% | 55.91% |
| 2023 | SPY | 0.40 | 25.52% | 26.71% |
| 2024 | D05.SI | 0.40 | 39.06% | 50.08% |
| 2024 | MSFT | 0.40 | 29.29% | 18.60% |
| 2024 | NVDA | 0.45 | 202.58% | 179.73% |
| 2024 | QQQ | 0.40 | 34.24% | 29.39% |
| 2024 | SPY | 0.40 | 29.70% | 26.68% |

### Interpretation

The walk-forward results showed that lower probability thresholds improved market exposure and helped the ML strategy participate more in upward market moves.

The tuned strategy outperformed Buy and Hold on several 2024 assets, including MSFT, NVDA, QQQ, and SPY.

However, it still underperformed some strong 2023 momentum assets such as NVDA, MSFT, and QQQ. This suggests that further improvements are needed, such as asset-specific thresholds, regime filters, macro features, and LLM-based news sentiment signals.

### Limitation

The best threshold may still be influenced by the 2023-2024 bull market environment. A final out-of-sample test on 2025-2026 data is needed to check whether the strategy generalizes to more recent market conditions.

---

## Current Research Insight

The project shows that ML-based trading signals should not be evaluated only by classification accuracy.

A RandomForest model with around 54% classification accuracy can still produce useful trading signals if the probability threshold, market exposure, and risk metrics are properly analyzed.

The key finding so far is that the default 0.50 probability threshold was too conservative for the 2023-2024 equity market environment. Lower thresholds around 0.40-0.45 increased market participation and improved performance in several assets.

However, the strategy still requires further validation across different market regimes, especially bearish or sideways markets.

## Next Steps

- Add 2025-2026 as a final out-of-sample test period.
- Test asset-specific probability thresholds.
- Add market regime filters.
- Add macro features such as VIX, interest rates, and USD/SGD.
- Add fundamental features for DBS and MSFT.
- Add LLM-based news sentiment signals.
- Compare RandomForest with XGBoost and LightGBM.
- Convert the final results into an English PDF portfolio report.

## Summary

I built a multi-asset AI quant research pipeline covering SGX and US assets. The project started with manual strategy backtesting and risk metrics, then expanded into vectorbt validation and RandomForest-based ML signal generation.

The ML model predicts whether each asset will generate a forward 5-trading-day return greater than 1%. Initial results showed that the default 0.50 classification threshold was too conservative, so I tested probability thresholds from 0.40 to 0.60.

Walk-forward validation showed that lower thresholds around 0.40-0.45 generally improved market exposure and helped the strategy participate more in upward market moves. However, further testing on 2025-2026 data and different market regimes is needed before treating the model as robust.

---

## Day 19 — Final Out-of-Sample Test: 2025–2026

I tested whether the probability thresholds selected during the 2023–2024 walk-forward validation period remained effective on a fresh 2025–2026 out-of-sample period.

The results were mixed.

The lower thresholds around 0.40–0.45 did not generalize perfectly across all assets. In the final OOS test, MSFT performed best at 0.50, while NVDA and QQQ performed best at 0.55.

### Key Results

| Asset | Best Threshold | Strategy Return | Benchmark Return | Strategy MDD | Benchmark MDD | Market Exposure |
|---|---:|---:|---:|---:|---:|---:|
| D05_SI | 0.40 | 42.86% | 63.30% | -19.61% | -19.26% | 89.36% |
| MSFT | 0.50 | -0.01% | -10.45% | -20.64% | -34.50% | 51.74% |
| NVDA | 0.55 | 54.69% | 49.05% | -10.07% | -36.88% | 31.64% |
| QQQ | 0.55 | 36.20% | 45.12% | -5.70% | -22.77% | 40.21% |
| SPY | 0.40 | 15.45% | 29.60% | -18.76% | -18.76% | 80.70% |

### Interpretation

The final OOS test showed that a single common probability threshold may not be optimal across all assets.

NVDA was the strongest result. The ML strategy outperformed Buy and Hold while reducing maximum drawdown from -36.88% to -10.07% and using only 31.64% market exposure.

MSFT and QQQ showed that the model may be useful as a downside-risk control tool, even when it does not always maximize total return.

### Main Insight

The model should not be treated as a universal return-maximizing signal. Instead, the results suggest that ML probability signals may be more useful for selective exposure control and risk management.

### Next Step

The next improvement is to test asset-specific thresholds and market regime filters instead of using one common threshold for every asset.
