# sg-ai-quant-portfolio
AI-powered quantitative trading strategy for Singapore market
# 🇸🇬 SG AI Quant Portfolio

AI-powered automated quantitative trading strategy targeting 
Singapore (SGX) and US equity markets.

## Overview
This project builds a full quant pipeline from data collection 
to automated signal generation, designed as a job portfolio for 
Singapore quantitative finance roles.

## Tech Stack
- **Language:** Python 3.11
- **Backtesting:** QuantConnect (LEAN), vectorbt
- **Data:** yfinance (SGX + US), Alpaca Markets API
- **ML/AI:** scikit-learn, XGBoost, LightGBM
- **LLM Signal:** Claude API (Anthropic)

## Target Markets
| Market | Assets |
|--------|--------|
| SGX (Singapore) | DBS Bank, Sea Limited, Singtel |
| US Equities | SPY, QQQ, NVDA |
| Asia ETFs | EWS, EWJ, EWY |

## Project Structure
\`\`\`
sg-ai-quant-portfolio/
├── data/          # SGX, US, FX, news data
├── strategies/    # Trading strategy modules
├── models/        # ML and LLM signal models
├── backtest/      # Backtesting scripts
├── reports/       # Charts and performance reports
└── notebooks/     # Research and EDA
\`\`\`

## Author
Chae Youngjun | Quant Finance Portfolio | 2026

## Recent Progress: Strategy and Risk Management

### Day 9-11 Summary

I expanded the project from basic backtesting into a multi-strategy risk management framework.

The current analysis compares 5 assets and 5 strategies:

Assets:
- NVDA
- MSFT
- SPY
- QQQ
- D05_SI

Strategies:
- Buy and Hold
- Current Defensive
- Trend Only
- Loose RSI
- 60-Day Breakout

Key risk metrics added:
- Total Return
- Maximum Drawdown
- Sharpe Ratio
- Annualized Volatility
- Daily VaR 95%
- Daily Expected Shortfall 95%
- Market Exposure

Main insight:
The 60-Day Breakout strategy reduced downside risk and volatility by lowering market exposure, but it also reduced upside participation compared with Buy and Hold and Trend Only.

Generated outputs:
- backtest/results/strategy_experiment_summary.csv
- backtest/results/risk_management_summary.csv
- reports/charts/risk_var_95_comparison.png
- reports/charts/risk_expected_shortfall_comparison.png
- reports/charts/risk_annualized_volatility_comparison.png
- reports/charts/risk_market_exposure_comparison.png
- reports/charts/risk_return_scatter_full.png
- reports/charts/risk_return_scatter_zoomed_ex_nvda.png

Interview summary:
I built a standardized strategy comparison and risk management framework across SGX and US assets. The framework evaluates both return and downside risk using VaR, Expected Shortfall, volatility, drawdown, and market exposure.
