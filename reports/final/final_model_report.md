# AI-Based Quantitative Portfolio Allocation Model

## 1. Project Overview

This project develops a machine-learning-based portfolio allocation model using daily market data.

This project was motivated by my university coursework in Equity Investment and Risk Management. In Equity Investment, I learned how investors evaluate assets, returns, and portfolio performance. In Risk Management, I learned that investment decisions should also consider volatility, drawdown, Value at Risk, Expected Shortfall, and downside protection. This project was designed to connect those concepts by building a machine-learning-based portfolio allocation model that focuses on risk-adjusted performance rather than raw return alone.

The goal of the project is not to build a model that always maximizes raw return. Instead, the goal is to build a risk-controlled portfolio allocation system that improves risk-adjusted performance compared with a simple Buy & Hold benchmark.

The final strategy uses an ExtraTreesClassifier to generate trading signals across a robust asset portfolio consisting of D05_SI, NVDA, and SPY.

## 2. Research Objective

The main research question was:

```text
Can a machine learning model improve risk-adjusted portfolio performance by reducing volatility and drawdown while maintaining meaningful upside exposure?
```

The project evaluates the model using total return, volatility, Sharpe ratio, maximum drawdown, Calmar ratio, Value at Risk, and Expected Shortfall.

## 3. Data and Assets

The model uses daily price data from yfinance.

```text
Asset    Description
D05_SI   DBS Group Holdings listed in Singapore
NVDA     NVIDIA listed in the United States
SPY      S&P 500 ETF
```

These assets were selected as the final robust portfolio because previous validation tests showed that they provided the strongest balance between return generation and risk control.

## 4. Final Model Configuration

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

## 5. Features Used

The final model uses seven daily technical features.

```text
Feature          Meaning
return_5d        Recent 5-day return
return_20d       Recent 20-day return
ma10_ratio       Current price relative to 10-day moving average
ma20_ratio       Current price relative to 20-day moving average
ma60_ratio       Current price relative to 60-day moving average
volatility_20d   Recent 20-day volatility
rsi_14           14-day RSI indicator
```

The Full feature set was kept because it produced the best balance between return, Sharpe ratio, and drawdown control in the feature-set ablation test.

## 6. Model Selection Process

Several model types and strategy designs were tested before selecting the final model.

```text
Experiment Area              Result
Model comparison             ExtraTreesClassifier was the strongest ML candidate
Feature importance           ma60_ratio, volatility_20d, and rsi_14 were most important
Ensemble averaging           ExtraTrees alone outperformed ensemble methods
Feature set ablation         Full Features provided the best balance
Target definition            5-day +1% target was strongest
Position sizing              Binary buy-or-cash sizing remained strongest
```

This process showed that additional complexity did not automatically improve performance. The final configuration was selected through empirical validation rather than assumption.

## 7. Final Backtest Result

```text
Strategy               Return    Volatility   Sharpe   MDD       Calmar   VaR 95   ES 95
Buy & Hold              54.25%     20.63%      1.46   -22.17%     2.45    -1.94%   -2.97%
Final ExtraTrees ML     38.17%      9.57%      2.24   -10.77%     3.55    -0.71%   -1.30%
```

## 8. Result Interpretation

The Buy & Hold portfolio produced a higher raw return than the final ML strategy.

However, the final ML strategy delivered stronger risk-adjusted performance.

Compared with Buy & Hold, the final ML model:

```text
Reduced volatility from 20.63% to 9.57%
Reduced maximum drawdown from -22.17% to -10.77%
Improved Sharpe ratio from 1.46 to 2.24
Improved Calmar ratio from 2.45 to 3.55
Reduced 95% VaR from -1.94% to -0.71%
Reduced 95% Expected Shortfall from -2.97% to -1.30%
```

This shows that the final model is best understood as a risk-controlled allocation system rather than a pure return-maximizing trading strategy.

## 9. Asset-Level Insight

```text
Asset    Model Return   Benchmark Return   Model Sharpe   Model MDD   Market Exposure
D05_SI      65.91%          66.99%             1.94        -19.26%       100.00%
NVDA        22.00%          45.30%             1.09         -6.50%         5.07%
SPY         26.56%          29.25%             0.98        -18.76%        91.47%
```

The strongest risk-control example was NVDA.

NVDA had high upside potential but also high drawdown risk. The ML model reduced NVDA market exposure to 5.07%, which lowered maximum drawdown from -36.88% under Buy & Hold to -6.50% under the ML strategy.

This demonstrates that the model was able to reduce exposure to high-volatility assets when predicted conditions were not favorable enough.

## 10. Key Research Lessons

The project produced several important research lessons.

```text
1. The best model was not necessarily the most complex model.
2. Ensemble averaging did not outperform the standalone ExtraTrees model.
3. Feature reduction improved defensiveness but reduced return too much.
4. Target definition had a major impact on model behavior.
5. Probability-based position sizing reduced volatility but weakened total performance.
6. The final model worked best as a risk-controlled allocation model.
```

## 11. Limitations

This project has several limitations.

```text
1. The model uses daily data only.
2. The strategy does not yet use intraday bars or tick-level order book data.
3. Transaction costs and slippage are modeled using a simplified 10 bps assumption.
4. The model is tested on a limited set of assets.
5. The strategy is a research backtest, not a live trading system.
```

These limitations are important because real-world trading would require more detailed execution modeling, broader asset coverage, and live paper-trading validation.

## 12. Future Extensions

Future versions of this project could include:

```text
1. QuantConnect / LEAN implementation
2. Intraday data testing using hourly or 15-minute bars
3. More detailed transaction cost and slippage modeling
4. Alpaca paper-trading pipeline
5. Streamlit dashboard for portfolio monitoring
6. Expanded asset universe including SGX, US equities, ETFs, and FX
```

## 13. Final Summary

The final ExtraTrees ML strategy did not outperform Buy & Hold in raw return. However, it significantly improved risk-adjusted performance by reducing volatility, maximum drawdown, Value at Risk, and Expected Shortfall while improving Sharpe and Calmar ratios.

Therefore, the final model is best interpreted as a risk-controlled machine learning portfolio allocation system.
---

## Final Project Status

This project is now considered a completed v1.1 portfolio project.

The local Python research workflow successfully produced a final machine learning portfolio allocation model using ExtraTreesClassifier. The final model did not maximize raw return, but it improved key risk-adjusted metrics compared with Buy & Hold, including volatility, maximum drawdown, Sharpe ratio, Calmar ratio, Value at Risk, and Expected Shortfall.

In addition, a simplified QuantConnect web backtest prototype was successfully executed using SPY and NVDA. This prototype does not represent a full deployment of the trained ExtraTrees model, but it verifies that the project's core trading logic can be translated into an event-driven algorithmic trading platform.

Overall, this project demonstrates the full research workflow from investment idea to data collection, feature engineering, model validation, risk analysis, portfolio construction, dashboard presentation, and platform implementation testing.

## Final Summary

```text
I built an end-to-end machine learning portfolio allocation project that connects Equity Investment, Risk Management, and Python-based quantitative research. The final ExtraTrees model improved risk-adjusted performance compared with Buy & Hold, and I also extended the project into a QuantConnect web prototype to verify that the strategy logic could run in an event-driven algorithmic trading environment.
```
