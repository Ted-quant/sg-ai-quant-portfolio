# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Simple backtest with performance metrics

import pandas as pd
import os
import numpy as np


# =========================
# 1. Load signal data
# =========================

def load_signal_data(file_path):
    """
    Load final signal data.
    """

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    return df


# =========================
# 2. Run simple backtest
# =========================

def run_backtest(df):
    """
    Run a simple backtest.

    Rule:
    - Invest when final_signal is Buy or Strong Buy
    - Stay in cash when final_signal is Hold
    """

    # Calculate daily stock return
    df["daily_return"] = df["Close"].pct_change()

    # 1 = invested, 0 = cash
    df["position"] = df["final_signal"].apply(
        lambda signal: 1 if signal in ["Buy", "Strong Buy"] else 0
    )

    # Use yesterday's position for today's return
    df["strategy_return"] = df["position"].shift(1) * df["daily_return"]

    # Replace missing values with 0
    df["daily_return"] = df["daily_return"].fillna(0)
    df["strategy_return"] = df["strategy_return"].fillna(0)

    # Calculate cumulative returns
    df["buy_and_hold_cumulative"] = (1 + df["daily_return"]).cumprod()
    df["strategy_cumulative"] = (1 + df["strategy_return"]).cumprod()

    return df


# =========================
# 3. Calculate max drawdown
# =========================

def calculate_max_drawdown(cumulative_return_series):
    """
    Calculate maximum drawdown.

    Drawdown means how much the portfolio falls from its previous peak.
    """

    running_max = cumulative_return_series.cummax()

    drawdown = cumulative_return_series / running_max - 1

    max_drawdown = drawdown.min()

    return max_drawdown


# =========================
# 4. Calculate Sharpe ratio
# =========================

def calculate_sharpe_ratio(daily_returns):
    """
    Calculate annualized Sharpe Ratio.

    Risk-free rate is assumed to be 0 for simplicity.
    """

    if daily_returns.std() == 0:
        return 0

    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    return sharpe


# =========================
# 5. Print and save performance summary
# =========================

def create_performance_summary(df):
    """
    Create performance summary table.
    """

    buy_and_hold_return = df["buy_and_hold_cumulative"].iloc[-1] - 1
    strategy_return = df["strategy_cumulative"].iloc[-1] - 1

    buy_and_hold_mdd = calculate_max_drawdown(df["buy_and_hold_cumulative"])
    strategy_mdd = calculate_max_drawdown(df["strategy_cumulative"])

    buy_and_hold_sharpe = calculate_sharpe_ratio(df["daily_return"])
    strategy_sharpe = calculate_sharpe_ratio(df["strategy_return"])

    summary = pd.DataFrame({
        "Metric": [
            "Total Return",
            "Maximum Drawdown",
            "Sharpe Ratio"
        ],
        "Buy and Hold": [
            buy_and_hold_return,
            buy_and_hold_mdd,
            buy_and_hold_sharpe
        ],
        "Strategy": [
            strategy_return,
            strategy_mdd,
            strategy_sharpe
        ]
    })

    return summary


# =========================
# 6. Main execution
# =========================

if __name__ == "__main__":

    print("Loading final signal data...")

    input_path = "data/signals/NVDA_final_signals.csv"

    df = load_signal_data(input_path)

    print("Running simple backtest...")

    result_df = run_backtest(df)

    print("Calculating performance metrics...")

    summary_df = create_performance_summary(result_df)

    os.makedirs("backtest/results", exist_ok=True)

    result_output_path = "backtest/results/NVDA_simple_backtest.csv"
    summary_output_path = "backtest/results/NVDA_performance_summary.csv"

    result_df.to_csv(result_output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)

    print("\nPerformance Summary")
    print("===================")
    print(summary_df)

    print(f"\nSaved backtest result to: {result_output_path}")
    print(f"Saved performance summary to: {summary_output_path}")
    print("Day 7 extended backtest metrics completed!")