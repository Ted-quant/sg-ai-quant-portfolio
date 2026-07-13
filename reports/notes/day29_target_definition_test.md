# Day 29 — Target Definition Test

## Objective

The goal of Day 29 was to test whether the original prediction target was the most appropriate target for the ML trading model.

The original target was:

```text
forward_5d_return > +1%
```

This means the model was trained to predict whether the asset would rise by more than 1% over the next five trading days.

## Target Definitions Tested

```text
Target      Meaning
5D_0pct     5-day forward return > 0%
5D_1pct     5-day forward return > +1%
10D_1pct    10-day forward return > +1%
10D_2pct    10-day forward return > +2%
```

## Portfolio-Level Results: Robust Assets

```text
Target       Return    Volatility   Sharpe   MDD       Calmar
Buy & Hold   62.23%     20.58%       1.61   -22.17%     2.81
5D_0pct       2.01%      4.76%       0.30    -4.25%     0.47
5D_1pct      38.17%      9.57%       2.24   -10.77%     3.55
10D_1pct     28.71%     19.76%       0.94   -22.00%     1.31
10D_2pct     29.72%     19.48%       0.97   -21.54%     1.38
```

## Interpretation

The original 5-day +1% target remained the strongest target definition.

The 5D_0pct target was too conservative and produced very low return. Although it reduced drawdown, it did not generate enough exposure to capture meaningful upside.

The 10-day targets generated higher returns than 5D_0pct, but their drawdowns were close to Buy & Hold. This means the 10-day targets weakened the model's risk-control benefit.

## Key Takeaway

```text
5D_1pct = best balance
5D_0pct = too defensive and noisy
10D targets = higher exposure but weaker drawdown control
```

The 5-day +1% target worked best because it defined a meaningful short-term upside opportunity without becoming too noisy or too close to Buy & Hold exposure.

## Research Decision

I decided to keep the original 5D_1pct target because it delivered the best balance between return, Sharpe ratio, and drawdown control.

## Interview Summary

I tested multiple target definitions because the prediction horizon and return threshold can strongly affect an ML trading model. The original 5-day +1% target remained the strongest choice because it delivered the best balance between return, Sharpe ratio, and drawdown control. Lower-return targets were too noisy or defensive, while 10-day targets increased exposure and weakened drawdown control.
