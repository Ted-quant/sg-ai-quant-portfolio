# Day 39 — QuantConnect Web Backtest Verification

## Objective

The objective of this step was to verify that the local Python quant research workflow could be translated into a QuantConnect event-driven backtest prototype.

This step does not represent a full deployment of the final trained ExtraTrees model. Instead, it verifies that the strategy logic can run inside QuantConnect's algorithmic trading environment.

## Prototype Setup

```text
Platform: QuantConnect Web IDE
Language: Python
Assets: SPY, NVDA
Data Frequency: Daily
Initial Cash: $100,000
Backtest Period: 2020-01-01 to 2026-07-01
Strategy Type: Risk-aware trend-following prototype
```

## Strategy Logic

The QuantConnect prototype used a simplified version of the local research model's core idea:

```text
1. Use daily equity data
2. Calculate trend indicators using moving averages
3. Calculate RSI as a market-condition filter
4. Estimate recent volatility using rolling daily returns
5. Allocate only when trend, RSI, and volatility conditions are favorable
6. Use SetHoldings() to manage portfolio exposure
```

## Backtest Result

The QuantConnect web backtest executed successfully.

```text
Final Equity: $474,203.31
Net Profit: $351,380.43
Return: 374.20%
Fees: -$1,921.14
Assets Tested: SPY, NVDA
```

The platform generated a strategy equity curve and portfolio statistics, confirming that the algorithm ran successfully inside QuantConnect.

## Important Interpretation

This result should be interpreted as a platform implementation test, not as the final validated ML model result.

The strong return was heavily influenced by the historical performance of NVDA during the test period. Therefore, this QuantConnect prototype should not be directly compared with the final local ExtraTrees ML backtest.

The main value of this step is that the project was successfully extended from a local Python research environment into a professional event-driven backtesting platform.

## Editor Warnings

QuantConnect displayed several editor-level type warnings in the Problems tab. These warnings did not prevent the backtest from running successfully.

The warnings were related to the editor's static type inference and were not runtime errors.

## Portfolio Significance

This step strengthens the project because it shows that the research workflow is not limited to local Python scripts.

It demonstrates the ability to:

```text
1. Build a local quantitative research model
2. Translate the strategy idea into QuantConnect's event-driven structure
3. Run a cloud-based algorithmic trading backtest
4. Interpret the difference between research validation and platform implementation
```

## Summary

```text
I successfully translated my local Python research workflow into a QuantConnect event-driven backtest prototype and verified that the strategy logic could run on a professional algorithmic trading platform.
```
