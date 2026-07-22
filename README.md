# 🇸🇬 Risk-Controlled ML Portfolio Allocator

AI-powered, risk-controlled quantitative portfolio allocation model for Singapore and US equity markets.

## Project Overview

This project was motivated by my coursework in **Equity Investment** and **Risk Management**.

Through these courses, I became interested in how investment decisions should be evaluated not only by raw return, but also by volatility, drawdown, downside risk, transaction costs, and portfolio-level risk control.

I built this project to translate those classroom concepts into a practical Python-based machine learning portfolio research workflow.

The project covers:

```text
1. Market data collection
2. Strategy backtesting
3. Risk analysis
4. Machine learning signal generation
5. Model validation
6. Transaction cost and slippage testing
7. Portfolio-level allocation
8. Signal-to-order execution bridge
9. Dashboard and platform-extension planning
```

The project is designed as a GitHub portfolio for quantitative finance, systematic trading, risk analytics, and financial data roles.

The final strategy is best interpreted as a **risk-controlled portfolio allocation system**, not a pure return-maximizing trading model.

---

## Final Model Summary

The final model was selected after testing multiple model designs, feature sets, target definitions, ensemble methods, and position-sizing approaches.

```text
Final Model Configuration

Model                         ExtraTreesClassifier
Feature Set                   Full Features
Target                        5-day forward return > +1%
Position Sizing               Binary
Portfolio Structure           Three fixed equal-weight asset sleeves
Final Assets                  D05_SI, NVDA, SPY
Data Frequency                Daily
Signal Execution Assumption   One-day delay
Baseline Transaction Cost     10 bps per absolute position change
```

The model predicts whether each asset will generate a forward five-trading-day return greater than +1%.

If the predicted probability passes the selected threshold, the asset becomes active within its assigned portfolio sleeve.

Each of the three final assets is assigned an equal one-third portfolio sleeve. When an asset signal is inactive, that sleeve remains in cash rather than being reallocated to the other active assets.

For example:

```text
Only NVDA active

D05_SI sleeve    0.0% invested
NVDA sleeve     33.3% invested
SPY sleeve       0.0% invested
Cash            66.7%
```

This fixed-sleeve structure limits concentration risk when only a small number of model signals are active.

---

## Final Backtest Result

The results below represent the committed **v1.1 research snapshot**.

The final ML strategy produced lower total return after the simplified transaction-cost assumption than Buy & Hold, but significantly improved risk-adjusted performance.

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

### Key Interpretation

The strategy did not outperform Buy & Hold in total return.

However, it significantly improved downside-risk control and risk-adjusted performance.

```text
Buy & Hold produced higher total return.
Final ExtraTrees ML produced stronger risk-adjusted performance.
```

This makes the project suitable as a portfolio allocation and risk-management case study rather than a simple directional trading model.

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
NVDA Buy & Hold MDD       -36.88%
NVDA ML Strategy MDD       -6.50%
NVDA ML Market Exposure     5.07%
```

The model substantially reduced exposure to a high-volatility asset when its predicted opportunity did not pass the selected threshold.

Because each asset has a fixed one-third portfolio sleeve, inactive NVDA periods remain in cash rather than being reallocated to D05_SI or SPY.

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
Portfolio allocation         Fixed one-third asset sleeves
```

Several more complex alternatives were tested, including ensemble averaging and probability-based position sizing. These methods did not improve the overall portfolio result.

```text
ExtraTrees
+ Full Features
+ 5D_1pct target
+ Binary sizing
+ Fixed equal-weight asset sleeves
```

remained the strongest overall configuration because it provided the best balance between return, volatility, Sharpe ratio, drawdown control, market exposure, and Calmar ratio.

---

## Tech Stack

```text
Language              Python
Data                  yfinance
Backtesting           Custom Python modules, vectorbt
Machine Learning      scikit-learn
Final Model           ExtraTreesClassifier
Risk Metrics          Sharpe, volatility, MDD, VaR, ES, Calmar
Portfolio             Fixed equal-weight asset sleeves
Dashboard             Streamlit
Platform Extension    QuantConnect / LEAN skeleton
Version Control       Git and GitHub
Integration Format    Reproducible CSV bridge
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

The final robust portfolio focuses on:

```text
D05_SI
NVDA
SPY
```

---

## Project Structure

```text
sg-ai-quant-portfolio/
├── backtest/
│   ├── final_model_backtest.py
│   ├── export_execution_bridge.py
│   └── results/
│       ├── final_model_asset_summary.csv
│       ├── final_model_portfolio_summary.csv
│       ├── final_model_equity_curves.csv
│       ├── final_model_execution_bridge.csv
│       └── final_model_execution_orders.csv
├── dashboard/
│   └── app.py
├── data/
│   └── snapshots/
│       └── final_model_bridge_prices_2026-07-06.csv
├── models/
├── quantconnect/
├── reports/
├── strategies/
├── README.md
├── RUN_GUIDE.md
└── requirements.txt
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

Key evaluation metrics included:

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
A strategy should not be judged only by raw return.
Risk-adjusted performance and downside protection are essential.
```

### 2. ML Signal Development

I created a supervised machine learning dataset using daily technical features.

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
Classification accuracy alone is not enough to evaluate a trading model.

The signal must be tested through return, drawdown, Sharpe ratio,
volatility, downside risk, market exposure, and transaction costs.
```

### 3. Model Validation

The model uses separate research periods.

```text
Period           Purpose
2020-2023        Model training
2024             Threshold and regime-filter validation
2025 onward      Out-of-sample testing
```

The validation process tested:

```text
Validation Area              Purpose
Threshold tuning             Test probability thresholds
Walk-forward testing         Check stability across time
Out-of-sample testing        Evaluate unseen periods
Regime filters               Test moving-average downside filters
Transaction costs            Apply a simplified cost assumption
Portfolio allocation         Aggregate asset-level strategies
Model comparison             Compare multiple classifiers
Feature importance           Interpret model inputs
Ensemble testing             Test probability averaging
Feature ablation             Test reduced feature sets
Target definition            Compare prediction labels
Position sizing              Compare binary and fractional sizing
Final backtest               Consolidate the selected model
```

### 4. Signal Timing and Baseline Transaction Costs

The backtest applies a one-day delay between the signal and the return attributed to the resulting position.

```text
Day t market data
→ Day t signal and target position
→ Day t position-change cost
→ Position affects return from the following trading day
```

The baseline model applies 10 basis points to every absolute binary position change.

```text
Position Change    Interpretation    Baseline Cost
0 → 1              Entry / BUY       10 bps
1 → 0              Exit / SELL       10 bps
1 → 1              Hold               0 bps
0 → 0              Stay in cash       0 bps
```

The reported ML performance is calculated after subtracting this simplified cost assumption.

---

## Research Development Summary

### Model Comparison

The following scikit-learn models were compared under the same validation framework:

```text
RandomForest
ExtraTrees
GradientBoosting
HistGradientBoosting
LogisticRegression
```

Main insight:

```text
ExtraTreesClassifier provided the strongest balance between
risk-adjusted return and drawdown control.
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
The model behaved like a risk-aware trend-following model because
it relied mainly on medium-term trend, recent volatility, and RSI.
```

### Ensemble Test

I tested whether averaging multiple model probabilities improved signal stability.

Main insight:

```text
Ensemble averaging did not outperform the standalone ExtraTrees model.
Weaker models diluted the stronger ExtraTrees signal.
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
Full Features provided the strongest balance between return generation
and risk control.
```

### Target Definition Test

Several prediction targets were tested.

```text
Target      Meaning
5D_0pct     5-day forward return > 0%
5D_1pct     5-day forward return > +1%
10D_1pct    10-day forward return > +1%
10D_2pct    10-day forward return > +2%
```

Main insight:

```text
The 5-day +1% target captured meaningful short-term upside without
becoming too noisy or too close to Buy & Hold exposure.
```

### Probability-Based Position Sizing

I tested whether model confidence should be converted into fractional exposure.

```text
Sizing Method   Meaning
Binary          0% or 100% of the assigned asset sleeve
ThreeStep       0%, 50%, or 100% of the assigned asset sleeve
FourStep        0%, 30%, 60%, or 100% of the assigned asset sleeve
```

Main insight:

```text
Fractional position sizing reduced volatility, but it also reduced
return and Sharpe ratio too much. Binary sizing remained strongest.
```

---

## Execution-Cost Simulator Integration

A reproducible CSV bridge connects this final ExtraTrees model with a separate **execution-cost-simulator** project.

The purpose of the bridge is to convert portfolio signals into auditable order events that can be evaluated using a more detailed synthetic execution-cost framework.

```text
Final ExtraTrees signal
→ Binary target position
→ Absolute position change
→ BUY or SELL order event
→ Fixed 10 bps baseline
→ Synthetic execution-cost scenario analysis
```

The bridge preserves:

```text
The final ExtraTrees model specification
The 2020-2023 training period
The 2024 validation period
The one-day position delay
The fixed one-third portfolio sleeves
The original 10 bps baseline cost
The BUY and SELL direction of each position change
```

### Run the Bridge Exporter

```bash
python3 backtest/export_execution_bridge.py
```

### Bridge Outputs

```text
data/snapshots/final_model_bridge_prices_2026-07-06.csv

backtest/results/final_model_execution_bridge.csv

backtest/results/final_model_execution_orders.csv
```

The daily bridge contains signal, probability, position, return, and baseline-cost information for every test-period row.

The order bridge contains only rows where the binary position changes.

```text
0 → 1    BUY
1 → 0    SELL
```

### Bridge Snapshot Summary

```text
Test period end             2026-07-06
Snapshot data through       2026-07-13
D05_SI order events          1
NVDA order events           26
SPY order events            15
```

The snapshot includes five future trading days after the test-period end because the model's supervised target uses a five-trading-day forward return.

The stored snapshot prevents later yfinance data revisions from changing the bridge outputs on repeated runs.

### Connection to the Execution-Cost Project

The companion execution-cost project evaluates synthetic execution scenarios using:

```text
Order size
Market liquidity
Bid-ask spread
Order-book depth
Market impact
TWAP
VWAP
Hybrid TWAP-VWAP execution
Volume-forecast uncertainty
Pre-trade TCA recommendations
```

This creates a clear division of responsibility between the two projects.

```text
Project 1
Decides whether an asset should be held.

Project 2
Evaluates how a resulting order could be executed and what it might cost.
```

The integration is a **synthetic cost-adjusted signal analysis**.

It does not represent:

```text
Live trading
Actual historical fills
Exchange-provided order-book data
Broker transaction records
A production execution system
```

---

## Dashboard

A Streamlit dashboard prototype was added to make the final model results easier to review visually.

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard summarizes:

```text
Final model configuration
Portfolio-level performance
Buy & Hold vs Final ExtraTrees ML
Equity curve
Asset-level results
Risk interpretation
Future extensions
```

---

## QuantConnect / LEAN Prototype

A QuantConnect / LEAN-style strategy skeleton was added to show how the local research strategy could be translated into a production-style, event-driven framework.

```text
quantconnect/final_model_skeleton.py
```

The skeleton demonstrates:

```text
Daily data updates
Rolling feature calculation
Signal-generation structure
SetHoldings-based allocation
LEAN-style algorithm architecture
```

This is not a full reproduction of the trained scikit-learn model.

It is a platform-implementation prototype showing how the research logic could be transferred into an algorithmic trading engine.

---

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the original final-model backtest:

```bash
python3 backtest/final_model_backtest.py
```

Run the reproducible execution bridge:

```bash
python3 backtest/export_execution_bridge.py
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

Main final-model outputs:

```text
backtest/results/final_model_asset_summary.csv
backtest/results/final_model_portfolio_summary.csv
backtest/results/final_model_equity_curves.csv
reports/charts/final_model_robust_assets_equity_curve.png
```

Execution-bridge outputs:

```text
data/snapshots/final_model_bridge_prices_2026-07-06.csv
backtest/results/final_model_execution_bridge.csv
backtest/results/final_model_execution_orders.csv
```

Additional details are available in:

```text
RUN_GUIDE.md
reports/final/final_model_report.md
reports/notes/
```

### Reproducibility Note

The original final-model script downloads adjusted market data from yfinance when it is executed.

Because data vendors may update historical adjusted prices and because the original test period does not use a permanently stored input dataset, rerunning the original script at a later date may produce slightly different summary values.

The execution bridge addresses this limitation for the cross-project integration by storing and reusing a fixed price snapshot.

The committed v1.1 performance table remains the project's original final research result, while the bridge snapshot provides a reproducible signal-to-order dataset for downstream execution-cost analysis.

---

## Key Findings

```text
1. The best-performing model was not necessarily the most complex model.

2. ExtraTrees outperformed the earlier RandomForest baseline.

3. Medium-term trend, volatility, and RSI were the most important features.

4. Ensemble averaging did not improve performance.

5. Full Features worked better than smaller defensive feature sets.

6. The 5-day +1% target was the strongest prediction target.

7. Probability-based sizing reduced exposure but weakened performance.

8. Fixed asset sleeves limited concentration when few signals were active.

9. The final model is best framed as a risk-controlled allocation system.

10. A reproducible CSV bridge now connects investment signals with
    downstream synthetic execution-cost analysis.
```

---

## Limitations

```text
1. The final model uses daily data only.

2. The original backtest does not model explicit next-open execution prices.

3. The original transaction-cost model uses a simplified 10 bps assumption.

4. The model is tested on a limited number of final portfolio assets.

5. FX conversion costs between SGD and USD are not modeled.

6. Portfolio construction does not include correlation optimization.

7. Fixed asset sleeves are not dynamically reallocated among active assets.

8. The original yfinance-based backtest may change slightly when rerun
   because adjusted historical data can be revised.

9. The execution bridge uses a fixed snapshot, but it is not an
   exchange-grade historical dataset.

10. The connected execution-cost simulator uses synthetic order books
    and scenarios rather than actual historical fills.

11. The QuantConnect implementation is a skeleton rather than a live system.

12. The project should be treated as quantitative finance research,
    not as investment advice or a production trading system.
```

---

## Future Extensions

```text
1. Implement the trained strategy more fully in QuantConnect / LEAN.

2. Test hourly or 15-minute market data.

3. Calibrate the connected execution-cost simulator with intraday volume,
   empirical spreads, asset-specific liquidity, and market-impact estimates.

4. Compare synthetic execution costs against broker paper-trading records.

5. Add FX conversion and cross-currency transaction costs.

6. Expand the asset universe across SGX, US equities, ETFs, and FX.

7. Add correlation-aware portfolio optimization.

8. Test dynamic reallocation among active portfolio sleeves.

9. Improve model explainability using permutation importance or SHAP.

10. Build stronger data-versioning and experiment-tracking workflows.
```

---

## Final Portfolio Insight

```text
The final ML strategy sacrificed some upside return, but it significantly
improved downside protection and risk-adjusted performance.
```

The first project determines **whether and how much to hold within fixed portfolio sleeves**.

The connected execution-cost project extends the research by evaluating **how resulting position changes could be executed under different synthetic liquidity and execution conditions**.

Together, the two projects demonstrate an end-to-end research workflow covering:

```text
Signal generation
Model validation
Risk-controlled allocation
Transaction-cost assumptions
Order-event generation
Execution strategy comparison
Pre-trade transaction cost analysis
```

---

## Author

Chae Youngjun  
Quant Finance Portfolio Project  
Singapore Edition, 2026
