# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Create professional Day 11 risk management charts
# Output folder: reports/charts/

import os
import platform
import subprocess
import pandas as pd
import matplotlib.pyplot as plt


def open_file(file_path):
    """
    Open saved chart automatically after creation.
    Works on macOS, Windows, and Linux.
    """

    system_name = platform.system()

    try:
        if system_name == "Darwin":
            subprocess.run(["open", file_path])
        elif system_name == "Windows":
            os.startfile(file_path)
        else:
            subprocess.run(["xdg-open", file_path])
    except Exception as error:
        print(f"Chart saved, but could not open automatically: {error}")


def load_risk_summary(file_path):
    """
    Load risk management summary CSV file.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


def plot_metric(df, metric, output_path, title, ylabel, convert_to_percent=True, use_absolute=False):
    """
    Plot one risk metric by ticker and strategy.
    """

    plot_df = df.copy()

    if use_absolute:
        plot_df[metric] = plot_df[metric].abs()

    if convert_to_percent:
        plot_df[metric] = plot_df[metric] * 100

    pivot_df = plot_df.pivot(
        index="Ticker",
        columns="Strategy",
        values=metric
    )

    ax = pivot_df.plot(
        kind="bar",
        figsize=(14, 7)
    )

    plt.title(title, fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Asset", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add value labels to make chart portfolio-ready
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
            fontsize=7,
            rotation=90,
            padding=3
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved chart: {output_path}")
    open_file(output_path)


if __name__ == "__main__":

    input_path = "backtest/results/risk_management_summary.csv"
    output_folder = "reports/charts"

    os.makedirs(output_folder, exist_ok=True)

    risk_df = load_risk_summary(input_path)

    plot_metric(
        df=risk_df,
        metric="Market Exposure",
        output_path=f"{output_folder}/risk_market_exposure_comparison.png",
        title="Market Exposure by Strategy",
        ylabel="Market Exposure (%)",
        convert_to_percent=True,
        use_absolute=False
    )

    plot_metric(
        df=risk_df,
        metric="Daily VaR 95",
        output_path=f"{output_folder}/risk_var_95_comparison.png",
        title="Daily VaR 95% Loss by Strategy",
        ylabel="Daily VaR 95% Loss (%)",
        convert_to_percent=True,
        use_absolute=True
    )

    plot_metric(
        df=risk_df,
        metric="Daily Expected Shortfall 95",
        output_path=f"{output_folder}/risk_expected_shortfall_comparison.png",
        title="Daily Expected Shortfall 95% Loss by Strategy",
        ylabel="Daily Expected Shortfall 95% Loss (%)",
        convert_to_percent=True,
        use_absolute=True
    )

    plot_metric(
        df=risk_df,
        metric="Annualized Volatility",
        output_path=f"{output_folder}/risk_annualized_volatility_comparison.png",
        title="Annualized Volatility by Strategy",
        ylabel="Annualized Volatility (%)",
        convert_to_percent=True,
        use_absolute=False
    )

    print("\nDay 11 risk management charts created successfully.")