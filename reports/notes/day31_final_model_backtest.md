# Day 31 — Final Model Backtest

## Objective

The goal of Day 31 was to consolidate the final model configuration selected from the previous experiments and run a clean final backtest.

After testing model types, feature sets, target definitions, ensemble methods, and position sizing methods, the final model configuration was selected based on empirical validation.

## Final Model Configuration

```text
Model: ExtraTreesClassifier
Feature Set: Full Features
Target: 5-day forward return > +1%
Position Sizing: Binary
Portfolio: Robust Assets
Assets: D05_SI, NVDA, SPY
Transaction Cost + Slippage: 10 bps per position change
```

## Final Portfolio-Level Results

```text
Strategy               Return    Volatility   Sharpe   MDD       Calmar   VaR 95   ES 95
Buy & Hold              54.25%     20.63%      1.46   -22.17%     2.45    -1.94%   -2.97%
Final ExtraTrees ML     38.17%      9.57%      2.24   -10.77%     3.55    -0.71%   -1.30%
```

## Asset-Level Results

```text
Asset    Model Return   Benchmark Return   Model Sharpe   Model MDD   Market Exposure
D05_SI      65.91%          66.99%             1.94        -19.26%       100.00%
NVDA        22.00%          45.30%             1.09         -6.50%         5.07%
SPY         26.56%          29.25%             0.98        -18.76%        91.47%
```

## Interpretation

The final ML strategy did not maximize raw return. Buy & Hold produced a higher total return.

However, the Final ExtraTrees ML model significantly improved risk-adjusted performance.

Compared with Buy & Hold, the final model reduced portfolio volatility from 20.63% to 9.57%, reduced maximum drawdown from -22.17% to -10.77%, improved Sharpe ratio from 1.46 to 2.24, and improved Calmar ratio from 2.45 to 3.55.

This confirms that the final model is best understood as a risk-controlled portfolio allocation system rather than a pure return-maximization strategy.

## Key Insight

```text
Buy & Hold produced higher raw return.
Final ExtraTrees ML produced better risk-adjusted performance.
```

The strongest risk-management example was NVDA. The model reduced NVDA exposure to 5.07%, which lowered maximum drawdown from -36.88% under Buy & Hold to -6.50% under the ML strategy.

## Research Decision

I selected the Final ExtraTrees ML strategy because it provided the best overall balance between return, volatility, Sharpe ratio, maximum drawdown, and Calmar ratio.

The final model was not selected by assumption. It was selected after comparing multiple model designs and validating that additional complexity did not always improve performance.

## Summary

I built and validated an ExtraTrees-based ML portfolio allocation model using daily market data. The final model did not beat Buy & Hold on raw return, but it significantly improved risk-adjusted performance by reducing volatility and maximum drawdown while increasing Sharpe and Calmar ratios. This shows that the model is best interpreted as a risk-controlled allocation system rather than a return-maximizing trading strategy.
