# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Plot backtest performance comparison

import pandas as pd
import matplotlib.pyplot as plt
import os


# =========================
# 1. Load backtest result
# =========================

def load_backtest_result(file_path):
    """
    Load backtest result CSV file.
    """

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    return df


# =========================
# 2. Plot cumulative returns
# =========================

def plot_cumulative_returns(df):
    """
    Plot Buy & Hold vs Strategy cumulative returns.
    """

    os.makedirs("reports/charts", exist_ok=True)

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["Date"],
        df["buy_and_hold_cumulative"],
        label="Buy & Hold"
    )

    plt.plot(
        df["Date"],
        df["strategy_cumulative"],
        label="Strategy"
    )

    plt.title("NVDA Backtest: Buy & Hold vs Strategy")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)

    output_path = "reports/charts/NVDA_backtest_comparison.png"

    plt.savefig(output_path)
    plt.show()

    print(f"Chart saved to: {output_path}")


# =========================
# 3. Main execution
# =========================

if __name__ == "__main__":

    print("Loading backtest result...")

    input_path = "backtest/results/NVDA_simple_backtest.csv"

    df = load_backtest_result(input_path)

    print("Creating cumulative return chart...")

    plot_cumulative_returns(df)

    print("Day 7-2 backtest chart completed!")
    