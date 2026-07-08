# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Day 18 walk-forward validation for ML probability thresholds
# Goal: Check whether ML threshold tuning works across different out-of-sample years

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


DATASET_PATH = "models/datasets/ml_signal_dataset.csv"
OUTPUT_FOLDER = "backtest/results"
CHART_FOLDER = "reports/charts"

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "ma20_ma60_ratio",
    "close_ma20_ratio",
    "close_ma60_ratio",
    "volatility_5d",
    "volatility_20d",
    "rsi"
]

TARGET_COLUMN = "target"

THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]

WALK_FORWARD_WINDOWS = [
    {
        "Train Period": "2022",
        "Train Start": "2022-01-01",
        "Train End": "2022-12-31",
        "Test Year": "2023",
        "Test Start": "2023-01-01",
        "Test End": "2023-12-31"
    },
    {
        "Train Period": "2022-2023",
        "Train Start": "2022-01-01",
        "Train End": "2023-12-31",
        "Test Year": "2024",
        "Test Start": "2024-01-01",
        "Test End": "2024-12-31"
    }
]


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


def load_dataset():
    """
    Load ML dataset.
    """

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    df = df.dropna(subset=["Date", "Ticker", "Close", TARGET_COLUMN])
    df = df.sort_values(["Ticker", "Date"])

    return df


def train_model(train_df):
    """
    Train RandomForest model on the selected train period.
    """

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=20,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(x_train, y_train)

    return model


def run_backtest(test_df, ticker, threshold):
    """
    Run threshold-based strategy backtest for one ticker.
    """

    ticker_df = test_df[test_df["Ticker"] == ticker].copy()
    ticker_df = ticker_df.sort_values("Date")

    ticker_df["daily_return"] = ticker_df["Close"].pct_change()

    # Buy when predicted probability is above the selected threshold
    ticker_df["position"] = np.where(
        ticker_df["predicted_probability"] >= threshold,
        1,
        0
    )

    # Shift the position to avoid look-ahead bias
    ticker_df["strategy_return"] = ticker_df["position"].shift(1) * ticker_df["daily_return"]
    ticker_df["benchmark_return"] = ticker_df["daily_return"]

    ticker_df = ticker_df.dropna(subset=["strategy_return", "benchmark_return"])

    if ticker_df.empty:
        return None

    ticker_df["strategy_cumulative_return"] = (1 + ticker_df["strategy_return"]).cumprod()
    ticker_df["benchmark_cumulative_return"] = (1 + ticker_df["benchmark_return"]).cumprod()

    result = {
        "Ticker": ticker,
        "Threshold": threshold,
        "Strategy Total Return": ticker_df["strategy_cumulative_return"].iloc[-1] - 1,
        "Benchmark Total Return": ticker_df["benchmark_cumulative_return"].iloc[-1] - 1,
        "Strategy Maximum Drawdown": calculate_max_drawdown(ticker_df["strategy_cumulative_return"]),
        "Benchmark Maximum Drawdown": calculate_max_drawdown(ticker_df["benchmark_cumulative_return"]),
        "Strategy Sharpe Ratio": calculate_sharpe_ratio(ticker_df["strategy_return"]),
        "Benchmark Sharpe Ratio": calculate_sharpe_ratio(ticker_df["benchmark_return"]),
        "Market Exposure": ticker_df["position"].mean(),
        "Average Predicted Probability": ticker_df["predicted_probability"].mean()
    }

    return result


def plot_walk_forward_summary(summary_df):
    """
    Save chart showing strategy return by threshold and test year.
    """

    for ticker in summary_df["Ticker"].unique():
        ticker_df = summary_df[summary_df["Ticker"] == ticker].copy()

        plt.figure(figsize=(10, 6))

        for test_year in ticker_df["Test Year"].unique():
            year_df = ticker_df[ticker_df["Test Year"] == test_year].copy()

            plt.plot(
                year_df["Threshold"],
                year_df["Strategy Total Return"],
                marker="o",
                label=f"Strategy {test_year}"
            )

            plt.axhline(
                year_df["Benchmark Total Return"].iloc[0],
                linestyle="--",
                label=f"Benchmark {test_year}"
            )

        plt.title(f"{ticker}: Walk-Forward Threshold Validation", fontsize=15, fontweight="bold", pad=15)
        plt.xlabel("Probability Threshold", fontsize=11)
        plt.ylabel("Total Return", fontsize=11)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()

        output_path = f"{CHART_FOLDER}/{ticker}_walk_forward_threshold_validation.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Saved chart: {output_path}")


if __name__ == "__main__":

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(CHART_FOLDER, exist_ok=True)

    dataset = load_dataset()

    all_results = []

    for window in WALK_FORWARD_WINDOWS:
        print(f"\nTraining on {window['Train Period']} and testing on {window['Test Year']}...")

        train_df = dataset[
            (dataset["Date"] >= window["Train Start"]) &
            (dataset["Date"] <= window["Train End"])
        ].copy()

        test_df = dataset[
            (dataset["Date"] >= window["Test Start"]) &
            (dataset["Date"] <= window["Test End"])
        ].copy()

        model = train_model(train_df)

        x_test = test_df[FEATURE_COLUMNS]
        test_df["predicted_probability"] = model.predict_proba(x_test)[:, 1]

        tickers = test_df["Ticker"].unique()

        for ticker in tickers:
            for threshold in THRESHOLDS:
                print(f"Testing {ticker}, year {window['Test Year']}, threshold {threshold}...")

                result = run_backtest(
                    test_df=test_df,
                    ticker=ticker,
                    threshold=threshold
                )

                if result is not None:
                    result["Train Period"] = window["Train Period"]
                    result["Test Year"] = window["Test Year"]
                    all_results.append(result)

    summary_df = pd.DataFrame(all_results)

    output_path = f"{OUTPUT_FOLDER}/ml_walk_forward_threshold_summary.csv"
    summary_df.to_csv(output_path, index=False)

    plot_walk_forward_summary(summary_df)

    print("\nDay 18 walk-forward threshold validation completed.")
    print("===================================================")
    print(summary_df)

    print(f"\nSaved summary to: {output_path}")