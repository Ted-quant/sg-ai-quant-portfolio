# 🇸🇬 Risk-Controlled ML Portfolio Allocator

A Python-based machine learning portfolio allocation project for Singapore and US equity markets.

This project was built to translate concepts from **Equity Investment** and **Risk Management** into a practical quantitative research workflow. Instead of maximizing raw return, the strategy focuses on **risk-adjusted performance, downside protection, and portfolio-level risk control**.

---

## Project Summary

The model predicts whether each asset is likely to generate a **5-day forward return greater than +1%**.

If the predicted probability passes the selected threshold, the asset becomes active within its fixed portfolio sleeve. If the signal is inactive, that sleeve remains in cash.

```text
Final Model        ExtraTreesClassifier
Target             5-day forward return > +1%
Features           Returns, moving-average ratios, volatility, RSI
Assets             D05_SI, NVDA, SPY
Portfolio Design   Fixed equal-weight asset sleeves
Data Frequency     Daily
Cost Assumption    10 bps per absolute position change
```

The final strategy is best interpreted as a **risk-controlled allocation model**, not a pure return-maximizing trading system.

---

## Final Backtest Result

```text
Strategy               Return    Volatility   Sharpe   MDD       Calmar   VaR 95   ES 95
Buy & Hold              54.25%     20.63%      1.46   -22.17%     2.45    -1.94%   -2.97%
Final ExtraTrees ML     38.17%      9.57%      2.24   -10.77%     3.55    -0.71%   -1.30%
```

The ML strategy produced lower total return than Buy & Hold, but improved the main risk-adjusted metrics:

```text
Volatility reduced from 20.63% to 9.57%
Maximum drawdown reduced from -22.17% to -10.77%
Sharpe ratio improved from 1.46 to 2.24
Calmar ratio improved from 2.45 to 3.55
95% VaR improved from -1.94% to -0.71%
95% Expected Shortfall improved from -2.97% to -1.30%
```

---

## Why This Project Matters

This project shows a full quantitative research workflow:

```text
1. Market data collection
2. Feature engineering
3. Machine learning signal generation
4. Walk-forward and out-of-sample testing
5. Transaction cost and slippage assumptions
6. Portfolio-level risk evaluation
7. Dashboard presentation
8. QuantConnect / LEAN prototype
9. Signal-to-order execution bridge
```

The key lesson is that a trading model should not be judged only by prediction accuracy or total return. It should also be evaluated by volatility, drawdown, downside risk, transaction costs, and portfolio behavior.

---

## Model Validation

The final model was selected through empirical testing rather than assumption.

```text
Validation Area              Final Decision
Model comparison             ExtraTreesClassifier
Feature importance           ma60_ratio, volatility_20d, rsi_14 were most important
Ensemble averaging           ExtraTrees alone performed better than model averaging
Feature set ablation         Full feature set provided the best balance
Target definition            5-day +1% target was strongest
Position sizing              Binary buy-or-cash sizing remained strongest
Portfolio allocation         Fixed one-third asset sleeves
```

The most important features suggest that the model behaves like a **risk-aware trend-following system**, using medium-term trend, volatility, and RSI to decide when exposure is attractive.

---

## Asset-Level Result

```text
Asset    Model Return   Benchmark Return   Model Sharpe   Model MDD   Market Exposure
D05_SI      65.91%          66.99%             1.94        -19.26%       100.00%
NVDA        22.00%          45.30%             1.09         -6.50%         5.07%
SPY         26.56%          29.25%             0.98        -18.76%        91.47%
```

The strongest risk-control example was NVDA:

```text
NVDA Buy & Hold MDD       -36.88%
NVDA ML Strategy MDD       -6.50%
NVDA ML Market Exposure     5.07%
```

The model substantially reduced exposure to NVDA when the predicted opportunity was not strong enough.

---

## Execution-Cost Bridge

This project is connected to a separate **execution-cost simulator** through a reproducible CSV bridge.

```text
Project 1: Decides whether and how much to hold
Project 2: Evaluates how resulting orders could be executed
```

The bridge converts final model signals into BUY and SELL order events for downstream synthetic execution-cost analysis.

```text
Final ExtraTrees signal
→ Binary target position
→ Position change
→ BUY / SELL order event
→ Execution-cost simulator
```

This creates a clear link between portfolio allocation and trade execution.

---

## Dashboard and QuantConnect Prototype

A Streamlit dashboard was added to make the final results easier to review.

```bash
streamlit run dashboard/app.py
```

A QuantConnect / LEAN-style skeleton was also added to show how the strategy logic could be transferred into an event-driven algorithmic trading framework.

```text
quantconnect/final_model_skeleton.py
```

The QuantConnect version is a platform prototype, not a full live deployment of the trained ML model.

---

## Tech Stack

```text
Language              Python
Data                  yfinance
Backtesting           Custom Python modules, vectorbt
Machine Learning      scikit-learn
Final Model           ExtraTreesClassifier
Risk Metrics          Sharpe, volatility, MDD, VaR, ES, Calmar
Dashboard             Streamlit
Platform Extension    QuantConnect / LEAN skeleton
Version Control       Git and GitHub
```

---

## Project Structure

```text
sg-ai-quant-portfolio/
├── backtest/        # Backtesting, validation, and final model scripts
├── dashboard/       # Streamlit dashboard
├── data/            # Market data and fixed snapshots
├── models/          # ML dataset and model scripts
├── quantconnect/    # QuantConnect / LEAN prototype
├── reports/         # Charts, notes, and final report
├── strategies/      # Strategy and indicator modules
├── README.md
├── RUN_GUIDE.md
└── requirements.txt
```

---

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the final model backtest:

```bash
python3 backtest/final_model_backtest.py
```

Run the execution bridge:

```bash
python3 backtest/export_execution_bridge.py
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

Main outputs:

```text
backtest/results/final_model_asset_summary.csv
backtest/results/final_model_portfolio_summary.csv
backtest/results/final_model_equity_curves.csv
backtest/results/final_model_execution_orders.csv
reports/charts/final_model_robust_assets_equity_curve.png
```

---

## Limitations

```text
1. The model uses daily data only.
2. Transaction costs are simplified using a 10 bps assumption.
3. FX conversion costs are not modeled.
4. The final portfolio uses a limited asset universe.
5. The execution-cost simulator uses synthetic order books, not actual fills.
6. The QuantConnect implementation is a prototype, not a live trading system.
```

This project should be treated as quantitative finance research, not investment advice or a production trading system.

---

## Interview Summary

```text
I built an end-to-end machine learning portfolio allocation project that connects investment theory, risk management, and Python-based quantitative research. The final ExtraTrees model did not maximize raw return, but it improved Sharpe ratio, drawdown, VaR, and Expected Shortfall compared with Buy & Hold. I also connected the signal model to an execution-cost simulator and added a QuantConnect prototype to show how the research could move toward a more practical trading workflow.
```

---

## Author

Chae Youngjun  
Quant Finance Portfolio Project  
Singapore Edition, 2026
