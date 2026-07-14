# 🇸🇬 SG AI Quant Portfolio

AI-powered risk-controlled quantitative trading strategy for Singapore and US equity markets.

## Project Overview

This project was motivated by my coursework in Equity Investment and Risk Management. Through these courses, I became interested in how investment decisions should be evaluated not only by raw return, but also by volatility, drawdown, downside risk, and portfolio-level risk control. I built this project to translate those classroom concepts into a practical Python-based machine learning portfolio model.

This project builds an end-to-end quantitative finance research pipeline covering data collection, strategy backtesting, risk analysis, machine learning signal generation, transaction cost testing, and portfolio-level allocation.

The project is designed as a GitHub portfolio for Singapore quantitative finance, systematic trading, and financial analytics roles.

The final strategy is best interpreted as a **risk-controlled portfolio allocation system**.

## Final Strategy Summary

The final candidate strategy uses:

- RandomForest probability signals
- Asset-specific probability thresholds
- Asset-specific moving average regime filters
- Transaction cost and slippage assumptions
- Robust asset selection
- Equal-weight portfolio allocation

The model predicts whether each asset will generate a forward 5-trading-day return greater than 1%.

The strongest final candidate is:

    Robust Assets Equal Weight ML Portfolio
    Assets: D05_SI, NVDA, SPY

## Final Portfolio Result

The final robust ML portfolio produced lower raw return than Buy and Hold, but significantly improved downside risk control and risk-adjusted performance.

    Portfolio                    Total Return   Volatility   Sharpe   MDD       Calmar
    Robust Assets ML Net           25.09%         8.81%       2.13    -2.53%     9.90
    Robust Assets Buy & Hold       49.37%        19.25%       1.81   -13.59%     3.63

## Key Interpretation

The strategy did not outperform Buy and Hold in total return.

However, it reduced maximum drawdown from -13.59% to -2.53% and improved the Calmar ratio from 3.63 to 9.90.

This suggests that the ML strategy is more useful for **selective exposure control and downside-risk management** than for pure return maximization.

## Tech Stack

- Language: Python 3.11
- Data: yfinance
- Backtesting: vectorbt, custom Python backtesting modules
- Machine Learning: scikit-learn RandomForestClassifier
- Risk Metrics: Sharpe ratio, volatility, maximum drawdown, VaR, Expected Shortfall, Calmar ratio
- Portfolio Construction: equal weight, inverse volatility weighting
- Version Control: GitHub

## Target Markets

    Market           Assets
    SGX Singapore    DBS Bank (D05.SI / D05_SI), Singtel
    US Equities      SPY, QQQ, NVDA, MSFT
    Asia ETFs        EWS, EWJ, EWY
    FX / Macro       USD/SGD

## Project Structure

    sg-ai-quant-portfolio/
    ├── data/          # SGX, US, FX, and news data
    ├── strategies/    # Trading strategy modules
    ├── models/        # ML and signal models
    ├── backtest/      # Backtesting and validation scripts
    ├── reports/       # Charts, notes, and performance reports
    └── notebooks/     # Research and exploratory analysis

## Methodology

### 1. Strategy and Risk Management

The project started with a multi-strategy backtesting framework across SGX and US assets.

Tested strategies included:

- Buy and Hold
- Defensive strategy
- Trend-only strategy
- Loose RSI strategy
- 60-day breakout strategy

Key risk metrics included:

- Total return
- Maximum drawdown
- Sharpe ratio
- Annualized volatility
- Daily VaR 95%
- Daily Expected Shortfall 95%
- Market exposure

Main insight:

The 60-day breakout strategy reduced downside risk and volatility by lowering market exposure, but it also reduced upside participation.

### 2. ML Signal Development

I created a supervised ML dataset using technical features such as:

- 5-day return
- 20-day return
- Moving average ratios
- 20-day volatility
- RSI 14

The RandomForest model was trained to predict whether the forward 5-trading-day return would exceed +1%.

Initial classification accuracy was around 54%, but trading performance depended heavily on market exposure and threshold selection.

Main insight:

Classification accuracy alone is not enough to evaluate a trading model. The signal must also be tested through return, drawdown, Sharpe ratio, and market exposure.

## Research Development Summary

### Day 17 — Probability Threshold Tuning

The default 0.50 classification threshold was too conservative during the 2024 test period.

Lower thresholds increased market exposure and improved performance in several assets.

    Asset    Best Threshold   Strategy Return   Benchmark Return   Interpretation
    D05.SI        0.40             29.27%            50.08%          Improved exposure but underperformed Buy and Hold
    MSFT          0.40             32.35%            18.60%          Outperformed Buy and Hold
    NVDA          0.40            175.41%           179.73%          Nearly matched Buy and Hold
    QQQ           0.45             34.15%            29.39%          Outperformed with lower exposure
    SPY           0.40             28.86%            26.68%          Outperformed after exposure increased

### Day 18 — Walk-Forward Threshold Validation

I tested whether the lower-threshold result was consistent across 2023 and 2024.

    Test Year   Asset    Best Threshold   Strategy Return   Benchmark Return
    2023        D05.SI        0.45             10.34%             5.75%
    2023        MSFT          0.45             20.24%            58.35%
    2023        NVDA          0.40            142.72%           246.10%
    2023        QQQ           0.40             17.92%            55.91%
    2023        SPY           0.40             25.52%            26.71%
    2024        D05.SI        0.40             39.06%            50.08%
    2024        MSFT          0.40             29.29%            18.60%
    2024        NVDA          0.45            202.58%           179.73%
    2024        QQQ           0.40             34.24%            29.39%
    2024        SPY           0.40             29.70%            26.68%

Main insight:

Lower thresholds around 0.40–0.45 generally worked better in the 2023–2024 validation period, but the result was still influenced by a favorable equity market environment.

### Day 19 — Final Out-of-Sample Test: 2025–2026

I tested whether the 2023–2024 threshold results generalized to a fresh 2025–2026 out-of-sample period.

The results were mixed.

    Asset    Best Threshold   Strategy Return   Benchmark Return   Strategy MDD   Benchmark MDD   Market Exposure
    D05_SI        0.40             42.86%            63.30%          -19.61%        -19.26%          89.36%
    MSFT          0.50             -0.01%           -10.45%          -20.64%        -34.50%          51.74%
    NVDA          0.55             54.69%            49.05%          -10.07%        -36.88%          31.64%
    QQQ           0.55             36.20%            45.12%           -5.70%        -22.77%          40.21%
    SPY           0.40             15.45%            29.60%          -18.76%        -18.76%          80.70%

Main insight:

A single common threshold was not optimal across all assets. Asset-specific threshold selection was needed.

### Day 20 — Asset-Specific Threshold + Regime Filter

I tested whether adding a 200-day moving average regime filter improved downside risk control.

    Asset    Strategy Type                Return    Benchmark Return   MDD       Benchmark MDD   Market Exposure
    D05_SI   Threshold Only               53.03%        65.79%        -19.10%      -19.26%          87.80%
    D05_SI   Threshold + Regime Filter    49.90%        65.79%        -14.49%      -19.26%          85.94%
    MSFT     Threshold Only               -7.61%        -7.75%        -26.66%      -34.50%          55.61%
    MSFT     Threshold + Regime Filter    -6.08%        -7.75%        -12.07%      -34.50%          15.24%
    NVDA     Threshold Only               68.26%        47.18%        -23.92%      -36.88%          37.17%
    NVDA     Threshold + Regime Filter    30.22%        47.18%         -8.37%      -36.88%          21.39%
    QQQ      Threshold Only                8.99%        42.90%        -17.00%      -22.77%          39.84%
    QQQ      Threshold + Regime Filter     7.03%        42.90%         -6.00%      -22.77%          27.81%
    SPY      Threshold Only               20.67%        29.42%        -18.76%      -18.76%          81.55%
    SPY      Threshold + Regime Filter    12.61%        29.42%        -10.00%      -18.76%          67.65%

Main insight:

The 200-day moving average filter acted mainly as a defensive risk management layer. It reduced drawdown but often reduced upside participation.

### Day 21 — Regime Filter Sensitivity Test

I compared 100-day, 150-day, and 200-day moving average filters.

    Asset    Best Filter   Strategy Return   Benchmark Return   Strategy MDD   Calmar Ratio   Market Exposure
    D05_SI   MA100             49.22%            66.62%           -7.73%          6.37            79.37%
    MSFT     No Filter         -7.61%            -7.75%          -26.66%         -0.29            55.61%
    NVDA     MA200             30.22%            47.18%           -8.37%          3.61            21.39%
    QQQ      MA100             10.80%            42.90%           -4.73%          2.28            22.99%
    SPY      MA100             19.00%            29.42%           -6.82%          2.78            62.30%

Main insight:

MA100 was the strongest general regime filter for D05_SI, QQQ, and SPY. NVDA worked better with MA200 due to higher volatility.

### Day 22 — Transaction Costs and Slippage

I tested whether the strategy remained robust after adding transaction costs and slippage.

Cost assumptions:

- Transaction cost: 5 bps per trade
- Slippage: 5 bps per trade
- Total cost: 10 bps per position change

    Asset    Gross Return   Net Return   Benchmark Return   Net Sharpe   Net MDD    Total Turnover   Cost Drag
    D05_SI      49.22%        42.94%          66.62%            1.84      -8.28%          43           4.30%
    MSFT        -7.61%       -12.38%          -7.75%           -0.24     -27.83%          53           5.30%
    NVDA        30.22%        22.28%          47.18%            0.84      -8.92%          63           6.30%
    QQQ         10.80%         4.77%          42.90%            0.46      -5.21%          56           5.60%
    SPY         19.00%        14.45%          29.42%            0.98      -7.20%          39           3.90%

Main insight:

Transaction costs reduced returns, but the strategy did not completely collapse. D05_SI, NVDA, and SPY remained the strongest post-cost candidates.

### Day 23 — Portfolio-Level Allocation Test

I combined individual asset signals into portfolio-level allocation systems.

    Portfolio                    Assets             Total Return   Volatility   Sharpe   MDD       Calmar
    All Assets ML Net            All                  15.54%         8.26%       1.25    -6.66%     2.33
    All Assets Buy & Hold        All                  28.77%        19.87%       0.98   -20.32%     1.42
    Robust Assets ML Net         D05_SI,NVDA,SPY      28.96%         8.95%       2.01    -5.91%     4.90
    Robust Assets Buy & Hold     D05_SI,NVDA,SPY      43.88%        20.64%       1.32   -21.68%     2.02

Main insight:

The robust ML portfolio achieved lower raw return than Buy and Hold, but it significantly improved Sharpe ratio, drawdown, volatility, and Calmar ratio.

### Day 24 — Inverse Volatility Portfolio Weighting

I compared equal-weight portfolio construction with inverse volatility weighting.

Average inverse volatility weights:

    Asset    Average Weight
    D05_SI      39.51%
    NVDA        16.62%
    SPY         43.87%

Final comparison:

    Portfolio                     Total Return   Volatility   Sharpe   MDD       Calmar
    Equal Weight ML Net             25.09%         8.81%       2.13    -2.53%     9.90
    Equal Weight Buy & Hold         49.37%        19.25%       1.81   -13.59%     3.63
    Inverse Vol ML Net              24.24%         7.95%       2.28    -4.04%     6.00
    Inverse Vol Buy & Hold          46.26%        14.86%       2.18   -13.35%     3.47

Main insight:

Inverse volatility weighting reduced volatility and improved Sharpe ratio, but equal weighting produced better total return, lower maximum drawdown, and a higher Calmar ratio.

The final candidate remains the Robust Assets Equal Weight ML Portfolio.

## Key Findings

1. The RandomForest model should not be evaluated only by classification accuracy.
2. Probability threshold selection strongly affects trading performance.
3. A single common threshold does not work equally well across all assets.
4. Regime filters improve downside-risk control but can reduce upside participation.
5. Transaction costs reduce performance, but the best signals remain reasonably robust.
6. Portfolio-level construction improves the usefulness of the ML signals.
7. The final model is best framed as a risk-controlled allocation system.

## Limitations

- The model currently uses only technical indicators.
- Transaction costs and slippage are simplified.
- FX conversion costs are not yet modeled.
- Portfolio construction does not yet include correlation optimization.
- The strategy has not yet been deployed on QuantConnect or Alpaca Paper Trading.
- The current version should be treated as a research portfolio, not a live trading system.

## Generated Outputs

Key output files include:

    backtest/results/ml_final_oos_2025_2026_summary.csv
    backtest/results/ml_regime_filter_sensitivity_summary.csv
    backtest/results/ml_transaction_cost_slippage_summary.csv
    backtest/results/ml_portfolio_allocation_summary.csv
    backtest/results/ml_inverse_volatility_portfolio_summary.csv

    reports/charts/portfolio_allocation_test_equity_curves.png
    reports/charts/inverse_volatility_portfolio_equity_curves.png

Detailed research notes are stored in:

    reports/notes/

## Author

Chae Youngjun  
Quant Finance Portfolio Project  
Singapore Edition, 2026
---

## Final Model Selection

After a series of model validation experiments, the final strategy was selected based on risk-adjusted portfolio performance rather than raw return maximization.

```text
Final Model Configuration

Model: ExtraTreesClassifier
Feature Set: Full Features
Target: 5-day forward return > +1%
Position Sizing: Binary
Portfolio: Robust Assets
Assets: D05_SI, NVDA, SPY
Transaction Cost + Slippage: 10 bps per position change
```

The final model was selected after testing several alternative designs:

```text
Experiment Area              Final Decision
Model comparison             ExtraTreesClassifier
Feature importance           ma60_ratio, volatility_20d, rsi_14 were most important
Ensemble averaging           ExtraTrees alone outperformed ensemble methods
Feature set ablation         Full Features provided the best balance
Target definition            5-day +1% target was the strongest
Position sizing              Binary buy-or-cash sizing remained strongest
```

## Final Backtest Result

```text
Strategy               Return    Volatility   Sharpe   MDD       Calmar   VaR 95   ES 95
Buy & Hold              54.25%     20.63%      1.46   -22.17%     2.45    -1.94%   -2.97%
Final ExtraTrees ML     38.17%      9.57%      2.24   -10.77%     3.55    -0.71%   -1.30%
```

The Buy & Hold portfolio produced a higher raw return. However, the Final ExtraTrees ML strategy delivered stronger risk-adjusted performance.

Compared with Buy & Hold, the final ML model:

```text
Reduced volatility from 20.63% to 9.57%
Reduced maximum drawdown from -22.17% to -10.77%
Improved Sharpe ratio from 1.46 to 2.24
Improved Calmar ratio from 2.45 to 3.55
Reduced 95% VaR from -1.94% to -0.71%
Reduced 95% Expected Shortfall from -2.97% to -1.30%
```

## Final Model Interpretation

The final strategy is best understood as a risk-controlled portfolio allocation model, not a pure return-maximizing trading model.

The model did not attempt to stay fully invested at all times. Instead, it used machine learning probability signals to decide when market conditions were favorable enough to hold each asset.

The strongest risk-control example was NVDA:

```text
NVDA Buy & Hold MDD: -36.88%
NVDA ML Strategy MDD: -6.50%
NVDA ML Market Exposure: 5.07%
```

This shows that the model reduced exposure to high-volatility assets when the predicted risk-adjusted opportunity was not strong enough.

## Why the Final Model Was Selected

The final model was not chosen by assumption. It was selected through empirical validation.

Several more complex alternatives were tested, including ensemble averaging and probability-based position sizing. However, these methods did not improve the overall portfolio result.

```text
ExtraTrees + Full Features + 5D_1pct target + Binary sizing
```

remained the strongest overall configuration because it provided the best balance between return, volatility, Sharpe ratio, drawdown control, and Calmar ratio.

## Key Portfolio Insight

```text
The final ML strategy sacrificed some upside return, but it significantly improved downside protection and risk-adjusted performance.
```

This makes the project suitable as a portfolio allocation and risk management case study rather than a simple directional trading model.

## Future Extensions

Future versions of this project could extend the current daily-data model into a more production-style trading workflow:

```text
1. Implement the strategy in QuantConnect / LEAN
2. Test intraday data such as hourly or 15-minute bars
3. Add a more detailed transaction cost and slippage model
4. Build an Alpaca paper-trading signal pipeline
5. Create a Streamlit dashboard for portfolio monitoring
```

These extensions would help test whether the current research model can be transferred into a more realistic algorithmic trading environment.
