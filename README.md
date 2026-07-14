# 🇸🇬 SG AI Quant Portfolio

AI-powered risk-controlled quantitative portfolio allocation model for Singapore and US equity markets.

## Project Overview

This project was motivated by my coursework in **Equity Investment** and **Risk Management**.

Through these courses, I became interested in how investment decisions should be evaluated not only by raw return, but also by volatility, drawdown, downside risk, and portfolio-level risk control. I built this project to translate those classroom concepts into a practical Python-based machine learning portfolio model.

This project builds an end-to-end quantitative finance research pipeline covering:

```text
1. Data collection
2. Strategy backtesting
3. Risk analysis
4. Machine learning signal generation
5. Transaction cost and slippage testing
6. Portfolio-level allocation
7. Model validation
8. Dashboard and platform-extension planning
```

The project is designed as a GitHub portfolio for Singapore quantitative finance, systematic trading, and financial analytics roles.

The final strategy is best interpreted as a **risk-controlled portfolio allocation system**, not a pure return-maximizing trading model.

---

## Final Model Summary

The final model was selected after testing multiple model designs, feature sets, target definitions, ensemble methods, and position-sizing approaches.

```text
Final Model Configuration

Model: ExtraTreesClassifier
Feature Set: Full Features
Target: 5-day forward return > +1%
Position Sizing: Binary
Portfolio: Robust Assets
Assets: D05_SI, NVDA, SPY
Data Frequency: Daily
Transaction Cost + Slippage: 10 bps per position change
```

The model predicts whether each asset will generate a forward 5-trading-day return greater than +1%.

If the predicted probability passes the selected threshold, the asset becomes eligible for allocation. Active assets are then combined into an equal-weight robust portfolio.

---

## Final Backtest Result

The final ML strategy produced lower raw return than Buy & Hold, but significantly improved risk-adjusted performance.

```text
Strategy               Return    Volatility   Sharpe   MDD       Calmar   VaR 95   ES 95
Buy & Hold              54.25%     20.63%      1.46   -22.17%     2.45    -1.94%   -2.97%
Final ExtraTrees ML     38.17%      9.57%      2.24   -10.77%     3.55    -0.71%   -1.30%
```

Compared with Buy & Hold, the final ML model:

```text
Reduced volatility from 20.63% to 9.57%
Reduced maximum drawdown from -22.17% to -10.77%
Improved Sharpe ratio from 1.46 to 2.24
Improved Calmar ratio from 2.45 to 3.55
Reduced 95% VaR from -1.94% to -0.71%
Reduced 95% Expected Shortfall from -2.97% to -1.30%
```

## Key Interpretation

The strategy did not outperform Buy & Hold in total return.

However, it significantly improved downside-risk control and risk-adjusted performance.

```text
Buy & Hold produced higher raw return.
Final ExtraTrees ML produced stronger risk-adjusted performance.
```

This makes the project suitable as a portfolio allocation and risk management case study rather than a simple directional trading model.

---

## Asset-Level Final Result

```text
Asset    Model Return   Benchmark Return   Model Sharpe   Model MDD   Market Exposure
D05_SI      65.91%          66.99%             1.94        -19.26%       100.00%
NVDA        22.00%          45.30%             1.09         -6.50%         5.07%
SPY         26.56%          29.25%             0.98        -18.76%        91.47%
```

The strongest risk-control example was NVDA.

```text
NVDA Buy & Hold MDD: -36.88%
NVDA ML Strategy MDD: -6.50%
NVDA ML Market Exposure: 5.07%
```

This shows that the model reduced exposure to a high-volatility asset when the predicted risk-adjusted opportunity was not strong enough.

---

## Why the Final Model Was Selected

The final model was not chosen by assumption. It was selected through empirical validation.

```text
Experiment Area              Final Decision
Model comparison             ExtraTreesClassifier
Feature importance           ma60_ratio, volatility_20d, and rsi_14 were most important
Ensemble averaging           ExtraTrees alone outperformed ensemble methods
Feature set ablation         Full Features provided the best balance
Target definition            5-day +1% target was strongest
Position sizing              Binary buy-or-cash sizing remained strongest
```

Several more complex alternatives were tested, including ensemble averaging and probability-based position sizing. However, these methods did not improve the overall portfolio result.

```text
ExtraTrees + Full Features + 5D_1pct target + Binary sizing
```

remained the strongest overall configuration because it provided the best balance between return, volatility, Sharpe ratio, drawdown control, and Calmar ratio.

---

## Tech Stack

```text
Language              Python
Data                  yfinance
Backtesting           Custom Python backtesting modules, vectorbt
Machine Learning      scikit-learn
Model                 ExtraTreesClassifier
Risk Metrics          Sharpe ratio, volatility, maximum drawdown, VaR, Expected Shortfall, Calmar ratio
Portfolio             Equal-weight robust asset allocation
Dashboard             Streamlit
Platform Extension    QuantConnect / LEAN skeleton
Version Control       GitHub
```

---

## Target Markets

```text
Market           Assets
SGX Singapore    DBS Bank (D05.SI / D05_SI), Singtel
US Equities      SPY, QQQ, NVDA, MSFT
Asia ETFs        EWS, EWJ, EWY
FX / Macro       USD/SGD
```

The final robust portfolio currently focuses on:

```text
D05_SI
NVDA
SPY
```

---

## Project Structure

```text
sg-ai-quant-portfolio/
├── backtest/        # Backtesting, validation, and final model scripts
├── dashboard/       # Streamlit dashboard prototype
├── data/            # Downloaded market data and data scripts
├── models/          # ML dataset and model-building scripts
├── quantconnect/    # QuantConnect / LEAN strategy skeleton
├── reports/         # Charts, notes, final report, and research outputs
├── strategies/      # Trading signal and indicator modules
├── README.md        # GitHub project overview
├── RUN_GUIDE.md     # Reproducibility guide
└── requirements.txt # Python dependencies
```

---

## Methodology

### 1. Strategy and Risk Management

The project started with a multi-strategy backtesting framework across SGX and US assets.

Tested strategies included:

```text
Buy and Hold
Defensive strategy
Trend-only strategy
Loose RSI strategy
60-day breakout strategy
```

Key risk metrics included:

```text
Total return
Maximum drawdown
Sharpe ratio
Annualized volatility
Daily VaR 95%
Daily Expected Shortfall 95%
Market exposure
Calmar ratio
```

Main insight:

```text
A strategy should not be judged only by raw return. Risk-adjusted metrics and downside protection are essential for evaluating portfolio quality.
```

### 2. ML Signal Development

I created a supervised ML dataset using daily technical features.

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

The model was trained to predict:

```text
forward_5d_return > +1%
```

Main insight:

```text
Classification accuracy alone is not enough to evaluate a trading model. The signal must be tested through portfolio return, drawdown, Sharpe ratio, volatility, and market exposure.
```

### 3. Model Validation

The project tested several model and strategy design choices.

```text
Validation Area              Purpose
Threshold tuning             Check how probability threshold affects exposure and returns
Walk-forward testing         Check whether threshold results were stable across years
Out-of-sample testing        Test the model on 2025-2026 data
Regime filters               Test moving-average-based downside protection
Transaction costs            Add realistic cost and slippage assumptions
Portfolio allocation         Move from asset-level signals to portfolio-level performance
Model comparison             Compare RandomForest, ExtraTrees, GradientBoosting, and LogisticRegression
Feature importance           Understand what the model was using
Ensemble testing             Check whether model averaging improved performance
Feature ablation             Check whether fewer features improved robustness
Target definition            Test whether the prediction target was appropriate
Position sizing              Test whether probability-based sizing improved results
Final backtest               Consolidate the selected final model
```

---

## Research Development Summary

### Model Comparison

Several scikit-learn models were compared under the same validation framework.

```text
Model Candidates
RandomForest
ExtraTrees
GradientBoosting
HistGradientBoosting
LogisticRegression
```

Main insight:

```text
ExtraTreesClassifier became the strongest ML candidate because it produced the best balance between risk-adjusted return and drawdown control.
```

### Feature Importance

The ExtraTrees model's most important features were:

```text
Feature           Importance Rank
ma60_ratio        1
volatility_20d    2
rsi_14            3
```

Main insight:

```text
The model behaved like a risk-aware trend-following model because it relied mainly on medium-term trend, recent volatility, and RSI-based market condition indicators.
```

### Ensemble Test

I tested whether averaging multiple model probabilities improved signal stability.

Main insight:

```text
Ensemble averaging did not outperform the standalone ExtraTrees model. Weaker models diluted the stronger ExtraTrees signal.
```

### Feature Set Ablation

I compared the full feature set against smaller feature groups.

```text
Feature Set    Interpretation
Full           Balanced main model
Top3           Defensive but lower return
RiskAware      Defensive but too conservative
Trend          Higher exposure but weaker risk control
Momentum       Aggressive and closer to Buy & Hold risk
```

Main insight:

```text
Full Features remained the best main model input because it provided the strongest balance between return generation and risk control.
```

### Target Definition Test

I tested several prediction targets.

```text
Target      Meaning
5D_0pct     5-day forward return > 0%
5D_1pct     5-day forward return > +1%
10D_1pct    10-day forward return > +1%
10D_2pct    10-day forward return > +2%
```

Main insight:

```text
The 5-day +1% target worked best because it captured meaningful short-term upside without becoming too noisy or too close to Buy & Hold exposure.
```

### Probability-Based Position Sizing

I tested whether model confidence should be converted into fractional exposure.

```text
Sizing Method   Meaning
Binary          0% or 100% position
ThreeStep       0%, 50%, or 100% position
FourStep        0%, 30%, 60%, or 100% position
```

Main insight:

```text
Fractional position sizing reduced volatility, but it also reduced return and Sharpe ratio too much. Binary position sizing remained the strongest overall choice.
```

---

## Key Findings

```text
1. The best model was not necessarily the most complex model.
2. ExtraTrees outperformed the earlier RandomForest baseline.
3. Feature importance showed that trend, volatility, and RSI were the most important signals.
4. Ensemble averaging did not improve performance.
5. Full Features worked better than smaller defensive feature sets.
6. The 5-day +1% target was the strongest prediction target.
7. Probability-based position sizing reduced exposure but weakened overall performance.
8. The final model is best framed as a risk-controlled allocation system.
```

---

## Dashboard

A Streamlit dashboard prototype was added to make the final model results easier to review visually.

To run the dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard summarizes:

```text
Final model configuration
Portfolio-level performance
Buy & Hold vs Final ExtraTrees ML comparison
Equity curve
Asset-level results
Interpretation and future extensions
```

---

## QuantConnect / LEAN Prototype

A QuantConnect / LEAN-style strategy skeleton was added to show how the local research strategy could be translated into a production-style event-driven trading framework.

```text
Location:
quantconnect/final_model_skeleton.py
```

The skeleton demonstrates:

```text
Daily data updates
Rolling feature calculation
Signal generation structure
SetHoldings-based portfolio allocation
LEAN-style algorithm architecture
```

This is not yet a full reproduction of the trained sklearn model. It is a platform-implementation prototype showing how the research logic could be transferred into an algorithmic trading engine.

---

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the final model backtest:

```bash
python backtest/final_model_backtest.py
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

Main final outputs:

```text
backtest/results/final_model_asset_summary.csv
backtest/results/final_model_portfolio_summary.csv
backtest/results/final_model_equity_curves.csv
reports/charts/final_model_robust_assets_equity_curve.png
```

More details are available in:

```text
RUN_GUIDE.md
reports/final/final_model_report.md
reports/notes/
```

---

## Limitations

```text
1. The model currently uses daily data only.
2. The strategy does not yet use intraday bars or tick-level order book data.
3. Transaction costs and slippage are modeled using a simplified 10 bps assumption.
4. The model is tested on a limited set of assets.
5. FX conversion costs are not yet modeled.
6. Portfolio construction does not yet include correlation optimization.
7. The QuantConnect file is currently a skeleton, not a fully deployed live strategy.
8. The current version should be treated as a research portfolio, not a live trading system.
```

---

## Future Extensions

```text
1. Implement the strategy in QuantConnect / LEAN
2. Test intraday data such as hourly or 15-minute bars
3. Add a more detailed transaction cost and slippage model
4. Build an Alpaca paper-trading signal pipeline
5. Expand the asset universe across SGX, US equities, ETFs, and FX
6. Add correlation-aware portfolio optimization
7. Improve model explainability with permutation importance or SHAP
```

---

## Final Portfolio Insight

```text
The final ML strategy sacrificed some upside return, but it significantly improved downside protection and risk-adjusted performance.
```

This project shows how concepts from Equity Investment, Risk Management, machine learning, and portfolio construction can be combined into a practical quantitative finance research workflow.

---

## Author

Chae Youngjun  
Quant Finance Portfolio Project  
Singapore Edition, 2026
