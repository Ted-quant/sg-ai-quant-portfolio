# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Create full and zoomed risk-return scatter plots for portfolio presentation

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


def prepare_plot_data(df):
    """
    Convert return and volatility values into percentage units.
    """

    plot_df = df.copy()

    plot_df["Total Return (%)"] = plot_df["Total Return"] * 100
    plot_df["Annualized Volatility (%)"] = plot_df["Annualized Volatility"] * 100

    return plot_df


def plot_risk_return_scatter(df, output_path, title, note):
    """
    Plot total return against annualized volatility.
    Each point represents one ticker-strategy combination.
    """

    fig, ax = plt.subplots(figsize=(13, 8))

    strategies = df["Strategy"].unique()

    for strategy in strategies:
        strategy_df = df[df["Strategy"] == strategy]

        ax.scatter(
            strategy_df["Annualized Volatility (%)"],
            strategy_df["Total Return (%)"],
            label=strategy,
            s=90,
            alpha=0.8
        )

        # Add ticker labels to each point
        for _, row in strategy_df.iterrows():
            ax.text(
                row["Annualized Volatility (%)"],
                row["Total Return (%)"],
                row["Ticker"],
                fontsize=8,
                ha="left",
                va="bottom"
            )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=18)
    ax.set_xlabel("Annualized Volatility (%)", fontsize=11)
    ax.set_ylabel("Total Return (%)", fontsize=11)

    ax.grid(True, alpha=0.25)
    ax.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add a small note below the chart instead of a subtitle
    fig.text(
        0.01,
        0.01,
        note,
        fontsize=9,
        ha="left",
        va="bottom"
    )

    plt.tight_layout(rect=[0, 0.04, 0.85, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved chart: {output_path}")
    open_file(output_path)


if __name__ == "__main__":

    input_path = "backtest/results/risk_management_summary.csv"
    output_folder = "reports/charts"

    os.makedirs(output_folder, exist_ok=True)

    risk_df = load_risk_summary(input_path)
    plot_df = prepare_plot_data(risk_df)

    # Chart 1: Full universe including NVDA
    full_output_path = f"{output_folder}/risk_return_scatter_full.png"

    plot_risk_return_scatter(
        df=plot_df,
        output_path=full_output_path,
        title="Risk-Return Profile: Full Asset Universe",
        note="Note: NVDA is included and acts as a high-return, high-volatility outlier."
    )

    # Chart 2: Zoomed view excluding NVDA
    zoomed_df = plot_df[plot_df["Ticker"] != "NVDA"].copy()
    zoomed_output_path = f"{output_folder}/risk_return_scatter_zoomed_ex_nvda.png"

    plot_risk_return_scatter(
        df=zoomed_df,
        output_path=zoomed_output_path,
        title="Risk-Return Profile: Zoomed View Excluding NVDA",
        note="Note: NVDA is excluded only for visualization clarity; it remains included in the full analysis."
    )

    print("\nProfessional risk-return scatter charts created successfully.")