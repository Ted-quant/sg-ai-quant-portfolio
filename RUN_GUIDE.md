# Run Guide

This guide explains how to reproduce the final model backtest.

## 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the final model backtest

```bash
python backtest/final_model_backtest.py
```

## 4. Main outputs

```text
backtest/results/final_model_asset_summary.csv
backtest/results/final_model_portfolio_summary.csv
backtest/results/final_model_equity_curves.csv
reports/charts/final_model_robust_assets_equity_curve.png
```

## 5. Final model configuration

```text
Model: ExtraTreesClassifier
Feature Set: Full Features
Target: 5-day forward return > +1%
Position Sizing: Binary
Portfolio: Robust Assets
Assets: D05_SI, NVDA, SPY
Transaction Cost + Slippage: 10 bps per position change
Data Frequency: Daily
```

## 6. Final result summary

```text
Strategy               Return    Volatility   Sharpe   MDD       Calmar
Buy & Hold              54.25%     20.63%      1.46   -22.17%     2.45
Final ExtraTrees ML     38.17%      9.57%      2.24   -10.77%     3.55
```

## 7. Interpretation

The final model did not maximize raw return. Instead, it improved risk-adjusted performance by reducing volatility and maximum drawdown while improving Sharpe and Calmar ratios.
