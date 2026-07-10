# Day 23 — Portfolio-Level Allocation Test

## Objective

The objective of this experiment was to test whether the individual ML trading signals could work better when combined into a portfolio-level allocation system.

Until Day 22, the analysis focused on individual assets. However, in real portfolio management, the key question is whether a strategy improves overall portfolio risk and return.

## Methodology

The strategy used:

- Asset-specific probability thresholds from Day 19
- Asset-specific regime filters from Day 21
- Transaction costs and slippage from Day 22
- 2025–2026 out-of-sample test period

Two ML portfolios were tested:

1. All Assets ML Portfolio
   - D05_SI
   - MSFT
   - NVDA
   - QQQ
   - SPY

2. Robust Assets ML Portfolio
   - D05_SI
   - NVDA
   - SPY

Each portfolio used equal-weight daily returns across the selected assets.

The ML portfolio was compared against an equal-weight Buy and Hold portfolio using the same asset universe.

## Results

| Portfolio | Assets | Total Return | Volatility | Sharpe | MDD | Calmar | Daily VaR 95% | Daily ES 95% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| All Assets ML Net | All | 15.54% | 8.26% | 1.25 | -6.66% | 2.33 | -0.84% | -1.23% |
| All Assets Buy & Hold | All | 28.77% | 19.87% | 0.98 | -20.32% | 1.42 | -1.86% | -2.79% |
| Robust Assets ML Net | D05_SI, NVDA, SPY | 28.96% | 8.95% | 2.01 | -5.91% | 4.90 | -0.72% | -1.19% |
| Robust Assets Buy & Hold | D05_SI, NVDA, SPY | 43.88% | 20.64% | 1.32 | -21.68% | 2.02 | -1.96% | -3.00% |

## Interpretation

The ML portfolios did not maximize total return compared with Buy and Hold.

However, they significantly improved risk-adjusted performance.

The All Assets ML Portfolio reduced maximum drawdown from -20.32% to -6.66% and improved Sharpe ratio from 0.98 to 1.25.

The Robust Assets ML Portfolio produced the strongest risk-adjusted result.

Compared with the Robust Assets Buy and Hold Portfolio, it reduced maximum drawdown from -21.68% to -5.91% and improved Sharpe ratio from 1.32 to 2.01.

## Main Insight

The ML system is more effective as a risk-controlled portfolio allocation framework than as a pure return-maximizing strategy.

Removing weaker assets such as MSFT and QQQ improved the portfolio's risk-adjusted performance.

The strongest interpretation is that the model can selectively reduce market exposure and improve downside protection while still maintaining meaningful positive returns.

## Limitations

- The portfolio uses equal weighting only.
- Position sizing is not yet volatility-adjusted.
- Correlation between assets is not explicitly optimized.
- Transaction costs are simplified.
- The portfolio still relies only on technical features.
- The result is based on the 2025–2026 out-of-sample period.

## Future Improvements

- Test inverse-volatility weighting.
- Add portfolio-level volatility targeting.
- Compare equal weight vs risk-adjusted allocation.
- Add correlation-aware portfolio construction.
- Add macro and sentiment features.
- Convert final results into an English portfolio report.

## Summary

The Day 23 experiment showed that the ML strategy works better as a portfolio-level risk management system than as a pure return maximization model. The Robust Assets ML Portfolio achieved a Sharpe ratio of 2.01 and reduced maximum drawdown to -5.91%, compared with -21.68% for the corresponding Buy and Hold portfolio.
