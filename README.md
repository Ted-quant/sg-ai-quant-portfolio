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
