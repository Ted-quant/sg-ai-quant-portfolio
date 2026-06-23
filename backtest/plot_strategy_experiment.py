# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Plot strategy experiment results

import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. Load experiment result
# =========================

def load_experiment_result(file_path):
    """
    Load strategy experiment summary CSV.
    """

    df = pd.read_csv(file_path)

    return df


# =========================
# 2. Plot Sharpe Ratio comparison
# =========================

def plot_sharpe_comparison(df):
    """
    Plot Sharpe Ratio by strategy for each ticker.
    """

    os.makedirs("reports/charts", exist_ok=True)

    pivot_df = df.pivot(
        index="Ticker",
        columns="Strategy",
        values="Sharpe Ratio"
    )

    pivot_df.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title("Strategy Experiment: Sharpe Ratio Comparison")
    plt.xlabel("Ticker")
    plt.ylabel("Sharpe Ratio")
    plt.xticks(rotation=0)
    plt.legend(title="Strategy")
    plt.grid(axis="y")

    output_path = "reports/charts/strategy_sharpe_comparison.png"

    plt.savefig(output_path)
    plt.show()

    print(f"Saved Sharpe comparison chart to: {output_path}")


# =========================
# 3. Plot Total Return comparison
# =========================

def plot_return_comparison(df):
    """
    Plot Total Return by strategy for each ticker.
    """

    os.makedirs("reports/charts", exist_ok=True)

    pivot_df = df.pivot(
        index="Ticker",
        columns="Strategy",
        values="Total Return"
    )

    pivot_df = pivot_df * 100

    pivot_df.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title("Strategy Experiment: Total Return Comparison")
    plt.xlabel("Ticker")
    plt.ylabel("Total Return (%)")
    plt.xticks(rotation=0)
    plt.legend(title="Strategy")
    plt.grid(axis="y")

    output_path = "reports/charts/strategy_return_comparison.png"

    plt.savefig(output_path)
    plt.show()

    print(f"Saved return comparison chart to: {output_path}")


# =========================
# 4. Plot Maximum Drawdown comparison
# =========================

def plot_mdd_comparison(df):
    """
    Plot Maximum Drawdown by strategy for each ticker.
    """

    os.makedirs("reports/charts", exist_ok=True)

    pivot_df = df.pivot(
        index="Ticker",
        columns="Strategy",
        values="Maximum Drawdown"
    )

    pivot_df = pivot_df * 100

    pivot_df.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title("Strategy Experiment: Maximum Drawdown Comparison")
    plt.xlabel("Ticker")
    plt.ylabel("Maximum Drawdown (%)")
    plt.xticks(rotation=0)
    plt.legend(title="Strategy")
    plt.grid(axis="y")

    output_path = "reports/charts/strategy_mdd_comparison.png"

    plt.savefig(output_path)
    plt.show()

    print(f"Saved MDD comparison chart to: {output_path}")


# =========================
# 5. Main execution
# =========================

if __name__ == "__main__":

    input_path = "backtest/results/strategy_experiment_summary.csv"

    print("Loading strategy experiment summary...")

    df = load_experiment_result(input_path)

    print("Creating Sharpe Ratio comparison chart...")
    plot_sharpe_comparison(df)

    print("Creating Total Return comparison chart...")
    plot_return_comparison(df)

    print("Creating Maximum Drawdown comparison chart...")
    plot_mdd_comparison(df)

    print("Strategy experiment visualization completed!")