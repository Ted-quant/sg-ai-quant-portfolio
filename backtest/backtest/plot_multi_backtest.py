# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Plot multi-asset backtest summary

import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. Load summary data
# =========================

def load_summary(file_path):
    """
    Load multi-asset backtest summary CSV.
    """

    df = pd.read_csv(file_path)

    return df


# =========================
# 2. Plot return comparison
# =========================

def plot_return_comparison(df):
    """
    Plot Buy & Hold Return vs Strategy Return.
    """

    os.makedirs("reports/charts", exist_ok=True)

    tickers = df["Ticker"]

    x = range(len(tickers))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(
        [i - width / 2 for i in x],
        df["Buy and Hold Return"] * 100,
        width,
        label="Buy & Hold Return"
    )

    plt.bar(
        [i + width / 2 for i in x],
        df["Strategy Return"] * 100,
        width,
        label="Strategy Return"
    )

    plt.title("Multi-Asset Backtest: Total Return Comparison")
    plt.xlabel("Ticker")
    plt.ylabel("Total Return (%)")
    plt.xticks(x, tickers)
    plt.legend()
    plt.grid(axis="y")

    output_path = "reports/charts/multi_asset_return_comparison.png"

    plt.savefig(output_path)
    plt.show()

    print(f"Saved return comparison chart to: {output_path}")


# =========================
# 3. Plot MDD comparison
# =========================

def plot_mdd_comparison(df):
    """
    Plot Buy & Hold MDD vs Strategy MDD.
    """

    os.makedirs("reports/charts", exist_ok=True)

    tickers = df["Ticker"]

    x = range(len(tickers))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(
        [i - width / 2 for i in x],
        df["Buy and Hold MDD"] * 100,
        width,
        label="Buy & Hold MDD"
    )

    plt.bar(
        [i + width / 2 for i in x],
        df["Strategy MDD"] * 100,
        width,
        label="Strategy MDD"
    )

    plt.title("Multi-Asset Backtest: Maximum Drawdown Comparison")
    plt.xlabel("Ticker")
    plt.ylabel("Maximum Drawdown (%)")
    plt.xticks(x, tickers)
    plt.legend()
    plt.grid(axis="y")

    output_path = "reports/charts/multi_asset_mdd_comparison.png"

    plt.savefig(output_path)
    plt.show()

    print(f"Saved MDD comparison chart to: {output_path}")


# =========================
# 4. Main execution
# =========================

if __name__ == "__main__":

    input_path = "backtest/results/multi_backtest_summary.csv"

    print("Loading multi-asset backtest summary...")

    df = load_summary(input_path)

    print("Creating return comparison chart...")
    plot_return_comparison(df)

    print("Creating MDD comparison chart...")
    plot_mdd_comparison(df)

    print("Multi-asset backtest visualization completed!")
    