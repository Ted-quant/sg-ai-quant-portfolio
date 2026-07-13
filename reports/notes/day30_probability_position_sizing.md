# Day 30 — Probability-Based Position Sizing

## Objective

The goal of Day 30 was to test whether model probability could be used not only as a buy-or-cash signal, but also as a position sizing signal.

Instead of always investing 100% when the signal passed the threshold, this experiment tested whether higher model confidence should lead to larger portfolio exposure.

## Sizing Methods Tested

```text
Sizing Method   Description
Binary          Existing method: 0% or 100% position
ThreeStep       0%, 50%, or 100% position
FourStep        0%, 30%, 60%, or 100% position
```

## Portfolio-Level Results: Robust Assets

```text
Sizing      Return   Volatility   Sharpe   MDD       Calmar
Buy & Hold  54.25%    20.63%       1.46   -22.17%     2.45
Binary      38.17%     9.57%       2.24   -10.77%     3.55
ThreeStep   30.23%     8.57%       2.04   -11.54%     2.62
FourStep    19.47%     7.28%       1.62   -11.20%     1.74
```

## Interpretation

The Binary method remained the strongest main strategy.

ThreeStep and FourStep reduced volatility, but they also reduced return and Sharpe ratio. FourStep was the most conservative version, but it sacrificed too much upside.

This suggests that the model's strongest signals worked better when expressed as a clear buy-or-cash decision rather than being diluted into smaller fractional positions.

## Key Takeaway

```text
Binary = best main strategy
ThreeStep = slightly more defensive but weaker
FourStep = too conservative
```

Probability-based position sizing improved defensiveness, but it did not improve the overall risk-adjusted performance of the portfolio.

## Research Decision

I decided to keep the Binary position sizing method as the main strategy because it produced the best balance between return, Sharpe ratio, and drawdown control.

## Summary

Instead of using the model output only as a binary buy-or-cash signal, I tested probability-based position sizing to convert model confidence into portfolio exposure. The results showed that fractional sizing reduced volatility, but it also reduced return and Sharpe ratio too much. Therefore, I kept the Binary method as the main strategy because it delivered the strongest overall balance.
