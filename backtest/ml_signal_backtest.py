# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Day 16 backtest ML-based buy signals
# Strategy: Buy when RandomForest predicts target = 1

import os
import platform
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


PREDICTION_PATH = "models/results/random_forest_predictions.csv"
OUTPUT_FOLDER = "backtest/results"
CHART_FOLDER = "reports/charts"


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


def calculate_max_drawdown(cumulative_return_series):
    """
    Calculate maximum drawdown from cumulative return series.
    """

    running_max = cumulative_return_series.cummax()
    drawdown = cumulative_return_series / running_max - 1

    return drawdown.min()


def calculate_sharpe_ratio(daily_returns):
    """
    Calculate annualized Sharpe ratio.
    Risk-free rate is assumed to be 0 for simplicity.
    """

    daily_returns = daily_returns.dropna()

    if daily_returns.std() == 0:
        return 0

    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)


def load_predictions(file_path):
    """
    Load RandomForest prediction results.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prediction file not found: {file_path}")

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["predicted_target"] = pd.to_numeric(df["predicted_target"], errors="coerce")
    df["predicted_probability"] = pd.to_numeric(df["predicted_probability"], errors="coerce")

    df = df.dropna(subset=["Date", "Ticker", "Close", "predicted_target", "predicted_probability"])
    df = df.sort_values(["Ticker", "Date"])

    return df


def run_ml_backtest_for_ticker(df, ticker):
    """
    Run ML signal backtest for one ticker.
    """

    ticker_df = df[df["Ticker"] == ticker].copy()
    ticker_df = ticker_df.sort_values("Date")

    ticker_df["daily_return"] = ticker_df["Close"].pct_change()

    # Position is 1 when the ML model predicts target = 1
    ticker_df["position"] = np.where(ticker_df["predicted_target"] == 1, 1, 0)

    # Use yesterday's prediction for today's return to avoid look-ahead bias
    ticker_df["ml_strategy_return"] = ticker_df["position"].shift(1) * ticker_df["daily_return"]

    # Benchmark is buy and hold
    ticker_df["benchmark_return"] = ticker_df["daily_return"]

    ticker_df = ticker_df.dropna(subset=["ml_strategy_return", "benchmark_return"])

    ticker_df["ml_cumulative_return"] = (1 + ticker_df["ml_strategy_return"]).cumprod()
    ticker_df["benchmark_cumulative_return"] = (1 + ticker_df["benchmark_return"]).cumprod()

    ml_total_return = ticker_df["ml_cumulative_return"].iloc[-1] - 1
    benchmark_total_return = ticker_df["benchmark_cumulative_return"].iloc[-1] - 1

    ml_mdd = calculate_max_drawdown(ticker_df["ml_cumulative_return"])
    benchmark_mdd = calculate_max_drawdown(ticker_df["benchmark_cumulative_return"])

    ml_sharpe = calculate_sharpe_ratio(ticker_df["ml_strategy_return"])
    benchmark_sharpe = calculate_sharpe_ratio(ticker_df["benchmark_return"])

    market_exposure = ticker_df["position"].mean()

    result = {
        "Ticker": ticker,
        "ML Total Return": ml_total_return,
        "Benchmark Total Return": benchmark_total_return,
        "ML Maximum Drawdown": ml_mdd,
        "Benchmark Maximum Drawdown": benchmark_mdd,
        "ML Sharpe Ratio": ml_sharpe,
        "Benchmark Sharpe Ratio": benchmark_sharpe,
        "Market Exposure": market_exposure,
        "Average Predicted Probability": ticker_df["predicted_probability"].mean()
    }

    return result, ticker_df


def plot_equity_curve(ticker_df, ticker):
    """
    Plot ML strategy equity curve against buy and hold benchmark.
    """

    output_path = f"{CHART_FOLDER}/{ticker}_ml_signal_equity_curve.png"

    plt.figure(figsize=(12, 6))

    plt.plot(
        ticker_df["Date"],
        ticker_df["ml_cumulative_return"],
        label="ML Signal Strategy"
    )

    plt.plot(
        ticker_df["Date"],
        ticker_df["benchmark_cumulative_return"],
        label="Buy and Hold Benchmark"
    )

    plt.title(f"{ticker}: ML Signal Strategy vs Buy and Hold", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Cumulative Return", fontsize=11)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved chart: {output_path}")
    open_file(output_path)


def plot_summary_chart(summary_df, metric, output_path, title, ylabel):
    """
    Plot summary comparison chart.
    """

    plot_df = summary_df.copy()

    ax = plot_df.plot(
        kind="bar",
        x="Ticker",
        y=metric,
        figsize=(10, 6),
        legend=False
    )

    plt.title(title, fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Asset", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.25)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=9, padding=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved chart: {output_path}")
    open_file(output_path)


if __name__ == "__main__":

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(CHART_FOLDER, exist_ok=True)

    prediction_df = load_predictions(PREDICTION_PATH)

    tickers = prediction_df["Ticker"].unique()

    all_results = []
    all_backtest_rows = []

    for ticker in tickers:
        print(f"Running ML signal backtest for {ticker}...")

        result, ticker_backtest_df = run_ml_backtest_for_ticker(prediction_df, ticker)

        all_results.append(result)
        all_backtest_rows.append(ticker_backtest_df)

        plot_equity_curve(ticker_backtest_df, ticker)

    summary_df = pd.DataFrame(all_results)
    detail_df = pd.concat(all_backtest_rows, ignore_index=True)

    summary_output_path = f"{OUTPUT_FOLDER}/ml_signal_backtest_summary.csv"
    detail_output_path = f"{OUTPUT_FOLDER}/ml_signal_backtest_details.csv"

    summary_df.to_csv(summary_output_path, index=False)
    detail_df.to_csv(detail_output_path, index=False)

    plot_summary_chart(
        summary_df=summary_df,
        metric="ML Total Return",
        output_path=f"{CHART_FOLDER}/ml_signal_total_return_comparison.png",
        title="ML Signal Strategy: Total Return by Asset",
        ylabel="Total Return"
    )

    plot_summary_chart(
        summary_df=summary_df,
        metric="ML Sharpe Ratio",
        output_path=f"{CHART_FOLDER}/ml_signal_sharpe_comparison.png",
        title="ML Signal Strategy: Sharpe Ratio by Asset",
        ylabel="Sharpe Ratio"
    )

    plot_summary_chart(
        summary_df=summary_df,
        metric="Market Exposure",
        output_path=f"{CHART_FOLDER}/ml_signal_market_exposure_comparison.png",
        title="ML Signal Strategy: Market Exposure by Asset",
        ylabel="Market Exposure"
    )

    print("\nDay 16 ML signal backtest completed.")
    print("===================================")
    print(summary_df)

    print(f"\nSaved summary to: {summary_output_path}")
    print(f"Saved details to: {detail_output_path}")