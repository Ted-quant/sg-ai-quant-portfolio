# Day 35 — QuantConnect / LEAN Implementation Plan

## Objective

The goal of Day 35 is to plan how the final local Python research model can be translated into a QuantConnect / LEAN-style algorithmic trading workflow.

The local project focused on data collection, feature engineering, model validation, and risk-adjusted backtesting.

The QuantConnect / LEAN extension focuses on converting the validated research strategy into a production-style event-driven trading structure.

## Why QuantConnect / LEAN?

QuantConnect / LEAN is useful because it provides a more realistic algorithmic trading framework than a simple local backtest.

In the local Python workflow, I manually handled data downloads, feature calculations, signal generation, portfolio returns, and performance metrics.

In a LEAN-style workflow, the trading engine handles market data updates, portfolio accounting, order execution simulation, and backtest reporting.

```text
Local Python Backtest:
Download data
Calculate features
Train model
Generate signals
Manually calculate returns and risk metrics

QuantConnect / LEAN:
Initialize algorithm
Receive market data through event-driven functions
Update indicators
Generate trading signals
Use SetHoldings or orders to manage portfolio exposure
Let the engine track portfolio performance
```

## Current Final Model

```text
Model: ExtraTreesClassifier
Feature Set: Full Features
Target: 5-day forward return > +1%
Position Sizing: Binary
Portfolio: Robust Assets
Assets: D05_SI, NVDA, SPY
Data Frequency: Daily
Transaction Cost + Slippage: 10 bps per position change
```

## What Can Be Transferred

The following parts of the local model can be transferred into a QuantConnect-style implementation.

```text
Component                  Transfer Plan
Asset universe             Use D05_SI, NVDA, and SPY
Data frequency             Use daily resolution
Feature logic              Recreate returns, moving-average ratios, volatility, and RSI
Signal concept             Use model probability or simplified signal logic
Portfolio construction     Equal-weight allocation across active robust assets
Risk-control idea          Reduce exposure when signal is not favorable
```

## What Needs to Be Simplified

The full local ML workflow may be too complex to move directly into QuantConnect in the first version.

The first QuantConnect version should focus on strategy structure rather than perfect ML replication.

```text
Challenge                         Initial Simplification
Model training inside LEAN         Start with skeleton or simplified signal proxy
Feature history management         Use RollingWindow or historical data requests
Transaction cost modeling          Use simple brokerage/slippage settings first
D05_SI data availability           Confirm data support or start with US-listed assets
Model serialization                Defer pickle/joblib model loading to later version
```

## Proposed QuantConnect Strategy Structure

The initial LEAN strategy should follow this structure.

```text
1. Initialize()
   - Set backtest start and end dates
   - Set initial capital
   - Add selected assets
   - Set daily resolution
   - Set warm-up period

2. Data Preparation
   - Maintain rolling price history
   - Calculate return_5d and return_20d
   - Calculate moving averages
   - Calculate ma10_ratio, ma20_ratio, ma60_ratio
   - Calculate volatility_20d
   - Calculate RSI

3. Signal Generation
   - Generate a binary signal for each asset
   - Signal = 1 means asset is eligible for allocation
   - Signal = 0 means asset should not be held

4. Portfolio Allocation
   - Allocate equal weight across active assets
   - If no assets have active signals, hold cash
   - Use SetHoldings() to update positions

5. Risk Management
   - Avoid trading before indicators are ready
   - Control turnover by rebalancing on a scheduled daily basis
   - Track exposure and drawdown
```

## Local Model vs QuantConnect Version

```text
Area              Local Python Version                  QuantConnect / LEAN Version
Data              yfinance                               QuantConnect data feed
Execution         Manual return calculation              Engine-managed orders and portfolio
Signal timing     End-of-day backtest logic              Event-driven scheduled execution
Portfolio         Manual equal-weight returns            SetHoldings-based allocation
Metrics           Manually calculated                    Engine-generated backtest report
Purpose           Research and model validation          Platform-style strategy implementation
```

## Implementation Roadmap

```text
Step 1:
Create a QuantConnect strategy skeleton using daily data and robust assets.

Step 2:
Implement daily indicator calculations using rolling price history.

Step 3:
Create a simplified signal logic that follows the same research idea as the final model.

Step 4:
Apply equal-weight allocation across active signals.

Step 5:
Compare the QuantConnect backtest result with the local Python final model.

Step 6:
Later extension: load a pre-trained sklearn model or retrain the model inside the algorithm.
```

## Limitations

The first QuantConnect version will not perfectly reproduce the local ExtraTrees model.

It is better understood as a platform translation prototype.

```text
Current limitation:
The local Python model is the main research model.

QuantConnect extension:
A production-style implementation plan and skeleton showing how the strategy could be transferred into an event-driven trading engine.
```

## Key Takeaway

The purpose of the QuantConnect / LEAN extension is not to replace the local research model.

The purpose is to show that the strategy can be thought of beyond a notebook-style backtest and translated into a more realistic algorithmic trading architecture.

## Interview Summary

After completing the local Python research workflow, I planned a QuantConnect / LEAN implementation to translate the strategy into a production-style algorithmic trading environment. The local model focused on feature engineering, model validation, and risk-adjusted backtesting, while the QuantConnect version would focus on event-driven execution, portfolio allocation, and realistic strategy deployment structure.
