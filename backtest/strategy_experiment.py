# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Compare multiple trading strategy rules across multiple assets

import os
import numpy as np
import pandas as pd


# =========================
# 1. Performance functions
# =========================

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

    if daily_returns.std() == 0:
        return 0

    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    return sharpe_ratio


# =========================
# 2. Strategy rules
# =========================

def apply_strategy_rule(df, strategy_name):
    """
    Apply selected strategy rule and create position column.
    Position 1 means invested, 0 means cash.
    """

    if strategy_name == "Current_Defensive":
        # Buy only when trend is positive and RSI is not overbought
        df["position"] = np.where(
            (df["MA20"] > df["MA60"]) & (df["RSI"] < 70),
            1,
            0
        )

    elif strategy_name == "Trend_Only":
        # Buy whenever short-term trend is above long-term trend
        df["position"] = np.where(
            df["MA20"] > df["MA60"],
            1,
            0
        )

    elif strategy_name == "Loose_RSI":
        # Similar to current strategy, but allows higher RSI
        df["position"] = np.where(
            (df["MA20"] > df["MA60"]) & (df["RSI"] < 80),
            1,
            0
        )

    elif strategy_name == "Buy_and_Hold":
        # Always invested
        df["position"] = 1

    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")

    return df


# =========================
# 3. Backtest function
# =========================

def run_strategy_backtest(ticker, strategy_name):
    """
    Run backtest for one ticker and one strategy.
    """

    input_path = f"data/raw/{ticker}_with_indicators.csv"

    if not os.path.exists(input_path):
        print(f"Skipped {ticker}: indicator file not found")
        return None

    df = pd.read_csv(input_path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna(subset=["Date", "Close", "MA20", "MA60", "RSI"])
    df = df.sort_values("Date")

    df["daily_return"] = df["Close"].pct_change()

    df = apply_strategy_rule(df, strategy_name)

    # Use yesterday's position for today's return to avoid look-ahead bias
    df["strategy_return"] = df["position"].shift(1) * df["daily_return"]

    df["cumulative_return"] = (1 + df["strategy_return"]).cumprod()

    total_return = df["cumulative_return"].iloc[-1] - 1
    max_drawdown = calculate_max_drawdown(df["cumulative_return"])
    sharpe_ratio = calculate_sharpe_ratio(df["strategy_return"].dropna())

    result = {
        "Ticker": ticker,
        "Strategy": strategy_name,
        "Total Return": total_return,
        "Maximum Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio
    }

    return result


# =========================
# 4. Run experiment
# =========================

def run_experiment(tickers, strategies):
    """
    Run all strategy experiments.
    """

    results = []

    for ticker in tickers:
        for strategy_name in strategies:
            print(f"Running {strategy_name} for {ticker}...")

            result = run_strategy_backtest(ticker, strategy_name)

            if result is not None:
                results.append(result)

    return pd.DataFrame(results)


# =========================
# 5. Main execution
# =========================

if __name__ == "__main__":

    tickers = [
        "NVDA",
        "MSFT",
        "SPY",
        "QQQ",
        "D05_SI"
    ]

    strategies = [
        "Buy_and_Hold",
        "Current_Defensive",
        "Trend_Only",
        "Loose_RSI"
    ]

    os.makedirs("backtest/results", exist_ok=True)

    experiment_df = run_experiment(tickers, strategies)

    output_path = "backtest/results/strategy_experiment_summary.csv"

    experiment_df.to_csv(output_path, index=False)

    print("\nStrategy Experiment Summary")
    print("===========================")
    print(experiment_df)

    print(f"\nSaved to: {output_path}")