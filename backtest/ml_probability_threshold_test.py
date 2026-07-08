# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Day 17 test ML probability thresholds for buy signals
# Strategy: Buy when predicted_probability is above different thresholds

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


PREDICTION_PATH = "models/results/random_forest_predictions.csv"
OUTPUT_FOLDER = "backtest/results"
CHART_FOLDER = "reports/charts"

THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]


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
    Risk-free rate is assumed to be zero for simplicity.
    """

    daily_returns = daily_returns.dropna()

    if daily_returns.std() == 0:
        return 0

    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)


def load_predictions():
    """
    Load RandomForest prediction results.
    """

    if not os.path.exists(PREDICTION_PATH):
        raise FileNotFoundError(f"Prediction file not found: {PREDICTION_PATH}")

    df = pd.read_csv(PREDICTION_PATH)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["predicted_probability"] = pd.to_numeric(df["predicted_probability"], errors="coerce")

    df = df.dropna(subset=["Date", "Ticker", "Close", "predicted_probability"])
    df = df.sort_values(["Ticker", "Date"])

    return df


def run_threshold_backtest(df, ticker, threshold):
    """
    Run one threshold-based ML signal backtest.
    """

    ticker_df = df[df["Ticker"] == ticker].copy()
    ticker_df = ticker_df.sort_values("Date")

    ticker_df["daily_return"] = ticker_df["Close"].pct_change()

    # Buy only when predicted probability is above the selected threshold
    ticker_df["position"] = np.where(ticker_df["predicted_probability"] >= threshold, 1, 0)

    # Shift position by one day to avoid look-ahead bias
    ticker_df["strategy_return"] = ticker_df["position"].shift(1) * ticker_df["daily_return"]
    ticker_df["benchmark_return"] = ticker_df["daily_return"]

    ticker_df = ticker_df.dropna(subset=["strategy_return", "benchmark_return"])

    ticker_df["strategy_cumulative_return"] = (1 + ticker_df["strategy_return"]).cumprod()
    ticker_df["benchmark_cumulative_return"] = (1 + ticker_df["benchmark_return"]).cumprod()

    strategy_total_return = ticker_df["strategy_cumulative_return"].iloc[-1] - 1
    benchmark_total_return = ticker_df["benchmark_cumulative_return"].iloc[-1] - 1

    strategy_mdd = calculate_max_drawdown(ticker_df["strategy_cumulative_return"])
    benchmark_mdd = calculate_max_drawdown(ticker_df["benchmark_cumulative_return"])

    strategy_sharpe = calculate_sharpe_ratio(ticker_df["strategy_return"])
    benchmark_sharpe = calculate_sharpe_ratio(ticker_df["benchmark_return"])

    market_exposure = ticker_df["position"].mean()

    result = {
        "Ticker": ticker,
        "Threshold": threshold,
        "Strategy Total Return": strategy_total_return,
        "Benchmark Total Return": benchmark_total_return,
        "Strategy Maximum Drawdown": strategy_mdd,
        "Benchmark Maximum Drawdown": benchmark_mdd,
        "Strategy Sharpe Ratio": strategy_sharpe,
        "Benchmark Sharpe Ratio": benchmark_sharpe,
        "Market Exposure": market_exposure
    }

    return result


def plot_threshold_results(summary_df):
    """
    Save threshold comparison charts.
    """

    for ticker in summary_df["Ticker"].unique():
        ticker_df = summary_df[summary_df["Ticker"] == ticker].copy()

        plt.figure(figsize=(10, 6))
        plt.plot(
            ticker_df["Threshold"],
            ticker_df["Strategy Total Return"],
            marker="o",
            label="ML Strategy"
        )
        plt.axhline(
            ticker_df["Benchmark Total Return"].iloc[0],
            linestyle="--",
            label="Buy and Hold"
        )

        plt.title(f"{ticker}: Return by ML Probability Threshold", fontsize=15, fontweight="bold", pad=15)
        plt.xlabel("Probability Threshold", fontsize=11)
        plt.ylabel("Total Return", fontsize=11)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()

        output_path = f"{CHART_FOLDER}/{ticker}_ml_threshold_return_comparison.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Saved chart: {output_path}")


if __name__ == "__main__":

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(CHART_FOLDER, exist_ok=True)

    prediction_df = load_predictions()

    tickers = prediction_df["Ticker"].unique()

    all_results = []

    for ticker in tickers:
        for threshold in THRESHOLDS:
            print(f"Testing {ticker} with threshold {threshold}...")

            result = run_threshold_backtest(
                df=prediction_df,
                ticker=ticker,
                threshold=threshold
            )

            all_results.append(result)

    summary_df = pd.DataFrame(all_results)

    output_path = f"{OUTPUT_FOLDER}/ml_probability_threshold_summary.csv"
    summary_df.to_csv(output_path, index=False)

    plot_threshold_results(summary_df)

    print("\nDay 17 ML probability threshold test completed.")
    print("==============================================")
    print(summary_df)

    print(f"\nSaved summary to: {output_path}")