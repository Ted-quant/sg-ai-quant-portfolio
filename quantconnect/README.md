# QuantConnect / LEAN Prototype

This folder contains a QuantConnect / LEAN-style strategy skeleton for the final ML portfolio allocation model.

## Purpose

The main research model was built and validated locally using Python, yfinance, pandas, scikit-learn, and custom backtesting logic.

The QuantConnect / LEAN prototype shows how the validated strategy could be translated into an event-driven algorithmic trading framework.

## Important Note

This is a skeleton implementation, not a perfect reproduction of the local ExtraTrees model.

The local model remains the main research model.

This QuantConnect version focuses on:

```text
1. Event-driven structure
2. Daily data updates
3. Rolling feature calculation
4. Signal generation logic
5. Equal-weight portfolio allocation
6. SetHoldings-based execution structure
```

## Final Local Model

```text
Model: ExtraTreesClassifier
Feature Set: Full Features
Target: 5-day forward return > +1%
Position Sizing: Binary
Portfolio: Robust Assets
Assets: D05_SI, NVDA, SPY
Data Frequency: Daily
```

## Skeleton Strategy Logic

The skeleton calculates local-model-style features:

```text
return_5d
return_20d
ma10_ratio
ma20_ratio
ma60_ratio
volatility_20d
rsi_14
```

Because the trained sklearn model is not yet loaded inside LEAN, the skeleton uses a simplified proxy signal based on the strongest feature-importance insights:

```text
Positive medium-term trend
Reasonable RSI range
Controlled recent volatility
```

## Future Extensions

```text
1. Confirm data availability for D05.SI inside QuantConnect
2. Load a serialized sklearn model into the LEAN algorithm
3. Retrain the model on a scheduled basis inside QuantConnect
4. Add more realistic brokerage, slippage, and fee models
5. Compare QuantConnect backtest results with the local Python final model
```

## Portfolio Interpretation

This extension is intended to show that the local research model can be translated into a production-style algorithmic trading architecture.

The goal is not to replace the local backtest, but to demonstrate awareness of the difference between research backtesting and platform-based algorithmic execution.
