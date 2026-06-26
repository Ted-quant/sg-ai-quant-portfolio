# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Standardized strategy experiment across multiple assets
# Backtest period: 2022-01-01 to 2024-12-31

import os
import numpy as np
import pandas as pd


START_DATE = "2022-01-01"
END_DATE = "2024-12-31"

TICKERS = [
    "NVDA",
    "MSFT",
    "SPY",
    "QQQ",
    "D05_SI"
]

STRATEGY_NAMES = [
    "Buy_and_Hold",
    "Current_Defensive",
    "Trend_Only",
    "Loose_RSI",
    "Breakout_60D"
]


def calculate_max_drawdown(cumulative_return_series):
    """
    Calculate maximum drawdown from cumulative return series.
    """

    running_max = cumulative_return_series.cummax()
    drawdown = cumulative_return_series / running_max - 1
    max_drawdown = drawdown.min()

    return max_drawdown


def calculate_sharpe_ratio(daily_returns):
    """
    Calculate annualized Sharpe ratio.
    Risk-free rate is assumed to be 0 for simplicity.
    """

    daily_returns = daily_returns.dropna()

    if daily_returns.std() == 0:
        return 0

    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)


def apply_strategy_rule(df, strategy_name):
    """
    Apply selected strategy rule and create position column.
    Position 1 means invested, 0 means cash.
    """

    df = df.copy()

    if strategy_name == "Buy_and_Hold":
        df["position"] = 1

    elif strategy_name == "Current_Defensive":
        df["position"] = np.where(
            (df["MA20"] > df["MA60"]) & (df["RSI"] < 70),
            1,
            0
        )

    elif strategy_name == "Trend_Only":
        df["position"] = np.where(
            df["MA20"] > df["MA60"],
            1,
            0
        )

    elif strategy_name == "Loose_RSI":
        df["position"] = np.where(
            (df["MA20"] > df["MA60"]) & (df["RSI"] < 80),
            1,
            0
        )

    elif strategy_name == "Breakout_60D":
        # Use yesterday's 60-day high to avoid look-ahead bias
        df["rolling_60d_high"] = df["Close"].rolling(window=60).max().shift(1)

        df["position"] = np.where(
            df["Close"] > df["rolling_60d_high"],
            1,
            0
        )

    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")

    return df


def load_indicator_data(ticker):
    """
    Load indicator data for one ticker and apply standardized date filter.
    """

    input_path = f"data/raw/{ticker}_with_indicators.csv"

    if not os.path.exists(input_path):
        print(f"Skipped {ticker}: indicator file not found at {input_path}")
        return None

    df = pd.read_csv(input_path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["MA20"] = pd.to_numeric(df["MA20"], errors="coerce")
    df["MA60"] = pd.to_numeric(df["MA60"], errors="coerce")
    df["RSI"] = pd.to_numeric(df["RSI"], errors="coerce")

    df = df.dropna(subset=["Date", "Close", "MA20", "MA60", "RSI"])
    df = df.sort_values("Date")

    df = df[
        (df["Date"] >= START_DATE) &
        (df["Date"] <= END_DATE)
    ].copy()

    if df.empty:
        print(f"Skipped {ticker}: no data available in selected period")
        return None

    return df


def run_single_strategy_experiment(ticker, strategy_name):
    """
    Run one strategy experiment for one ticker.
    """

    df = load_indicator_data(ticker)

    if df is None:
        return None

    df["daily_return"] = df["Close"].pct_change()

    df = apply_strategy_rule(df, strategy_name)

    # Use yesterday's position for today's return to avoid look-ahead bias
    df["strategy_return"] = df["position"].shift(1) * df["daily_return"]

    df = df.dropna(subset=["strategy_return"])

    if df.empty:
        print(f"Skipped {ticker} - {strategy_name}: not enough return data")
        return None

    df["cumulative_return"] = (1 + df["strategy_return"]).cumprod()

    total_return = df["cumulative_return"].iloc[-1] - 1
    max_drawdown = calculate_max_drawdown(df["cumulative_return"])
    sharpe_ratio = calculate_sharpe_ratio(df["strategy_return"])

    result = {
        "Ticker": ticker,
        "Strategy": strategy_name,
        "Total Return": total_return,
        "Maximum Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio
    }

    return result


def run_experiment(tickers, strategies):
    """
    Run strategy experiments for all tickers and strategies.
    """

    results = []

    for ticker in tickers:
        for strategy_name in strategies:
            print(f"Running experiment for {ticker} - {strategy_name}...")

            result = run_single_strategy_experiment(ticker, strategy_name)

            if result is not None:
                results.append(result)

    return pd.DataFrame(results)


if __name__ == "__main__":

    os.makedirs("backtest/results", exist_ok=True)

    experiment_df = run_experiment(TICKERS, STRATEGY_NAMES)

    output_path = "backtest/results/strategy_experiment_summary.csv"
    experiment_df.to_csv(output_path, index=False)

    print("\nStrategy Experiment Summary")
    print("===========================")
    print(experiment_df)

    print(f"\nSaved to: {output_path}")