# Day 22 — Transaction Costs and Slippage Test

## Objective

The objective of this experiment was to test whether the ML strategy remained robust after adding transaction costs and slippage.

Until Day 21, the backtests were based on gross returns without trading frictions. However, realistic trading strategies must account for transaction costs and slippage.

## Methodology

The strategy used:

- Asset-specific probability thresholds from Day 19
- Asset-specific best regime filters from Day 21
- 2022–2024 training period
- 2025–2026 out-of-sample test period

Trading friction assumptions:

- Transaction cost: 5 basis points per trade
- Slippage: 5 basis points per trade
- Total cost: 10 basis points per position change

The net strategy return was calculated as:

gross_strategy_return - trading_cost

## Results

```text
Asset    Gross Return   Net Return   Benchmark Return   Net Sharpe   Net MDD    Total Turnover   Cost Drag
D05_SI      49.22%        42.94%          66.62%            1.84      -8.28%          43           4.30%
MSFT        -7.61%       -12.38%          -7.75%           -0.24     -27.83%          53           5.30%
NVDA        30.22%        22.28%          47.18%            0.84      -8.92%          63           6.30%
QQQ         10.80%         4.77%          42.90%            0.46      -5.21%          56           5.60%
SPY         19.00%        14.45%          29.42%            0.98      -7.20%          39           3.90%
```

## Interpretation

Transaction costs reduced performance across all assets.

However, the strategy did not completely collapse after costs.

D05_SI, NVDA, and SPY remained reasonably robust after transaction costs and slippage.

QQQ became much weaker after costs, and MSFT became worse than Buy and Hold.

This suggests that the current ML signals are not equally strong across all assets.

## Asset-Level Analysis

### D05_SI

D05_SI remained strong after costs.

Net return was 42.94%, and maximum drawdown remained low at -8.28%.

This suggests that the D05_SI strategy was not overly dependent on unrealistic zero-cost assumptions.

### MSFT

MSFT became weaker after costs.

The net return was -12.38%, compared with the benchmark return of -7.75%.

This suggests that the MSFT signal is not strong enough under realistic trading assumptions.

### NVDA

NVDA remained positive after costs, with a net return of 22.28%.

Although it underperformed Buy and Hold in total return, it maintained a much lower drawdown.

This supports the interpretation that the NVDA strategy is more useful as a drawdown-controlled exposure strategy than a pure return maximization strategy.

### QQQ

QQQ was highly affected by costs.

The net return fell to 4.77%, suggesting that the signal may not be strong enough after trading frictions.

### SPY

SPY remained positive after costs, with a net return of 14.45%.

Although it underperformed Buy and Hold, it maintained a much lower drawdown.

This suggests that the SPY strategy may be useful as a defensive allocation approach.

## Main Insight

The transaction cost test showed that the strategy's robustness differs by asset.

The strongest post-cost candidates are D05_SI, NVDA, and SPY.

MSFT and QQQ require further improvement before they can be considered robust after realistic trading frictions.

The strategy should be framed as a risk-controlled allocation system rather than a universal alpha generator.

## Limitations

- The cost assumption is simplified.
- Real bid-ask spreads may differ across assets.
- SGX and US assets may have different cost structures.
- Tax, borrow costs, and liquidity constraints are not included.
- The model still uses only technical indicators.
- The strategy does not yet include portfolio-level position sizing.

## Summary

The Day 22 experiment showed that transaction costs and slippage reduced returns, but the strategy remained reasonably robust for D05_SI, NVDA, and SPY. QQQ and MSFT weakened after costs, suggesting that those signals require further improvement. Overall, the model is better interpreted as a selective, risk-controlled allocation system rather than a universal return-maximizing strategy.
