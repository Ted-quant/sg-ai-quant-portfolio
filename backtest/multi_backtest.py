# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Run simple backtest for multiple tickers

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
# 2. Single ticker backtest
# =========================

def run_single_backtest(ticker):
    """
    Run backtest for one ticker using final signal CSV.
    """

    input_path = f"data/signals/{ticker}_final_signals.csv"

    if not os.path.exists(input_path):
        print(f"Skipped {ticker}: signal file not found")
        return None

    df = pd.read_csv(input_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    df["daily_return"] = df["Close"].pct_change()

    df["position"] = df["final_signal"].apply(
        lambda x: 1 if x in ["Buy", "Strong Buy"] else 0
    )

    # Use yesterday's signal for today's return to avoid look-ahead bias
    df["strategy_return"] = df["position"].shift(1) * df["daily_return"]

    df["buy_and_hold_cumulative"] = (1 + df["daily_return"]).cumprod()
    df["strategy_cumulative"] = (1 + df["strategy_return"]).cumprod()

    buy_hold_total_return = df["buy_and_hold_cumulative"].iloc[-1] - 1
    strategy_total_return = df["strategy_cumulative"].iloc[-1] - 1

    buy_hold_mdd = calculate_max_drawdown(df["buy_and_hold_cumulative"])
    strategy_mdd = calculate_max_drawdown(df["strategy_cumulative"])

    buy_hold_sharpe = calculate_sharpe_ratio(df["daily_return"].dropna())
    strategy_sharpe = calculate_sharpe_ratio(df["strategy_return"].dropna())

    result = {
        "Ticker": ticker,
        "Buy and Hold Return": buy_hold_total_return,
        "Strategy Return": strategy_total_return,
        "Buy and Hold MDD": buy_hold_mdd,
        "Strategy MDD": strategy_mdd,
        "Buy and Hold Sharpe": buy_hold_sharpe,
        "Strategy Sharpe": strategy_sharpe
    }

    return result


# =========================
# 3. Multi ticker backtest
# =========================

def run_multi_backtest(tickers):
    """
    Run backtest for multiple tickers.
    """

    results = []

    for ticker in tickers:
        print(f"Running backtest for {ticker}...")

        result = run_single_backtest(ticker)

        if result is not None:
            results.append(result)

    return pd.DataFrame(results)


# =========================
# 4. Main execution
# =========================

if __name__ == "__main__":

    tickers = [
        "NVDA",
        "MSFT",
        "SPY",
        "QQQ",
        "D05_SI"
    ]

    os.makedirs("backtest/results", exist_ok=True)

    summary_df = run_multi_backtest(tickers)

    output_path = "backtest/results/multi_backtest_summary.csv"

    summary_df.to_csv(output_path, index=False)

    print("\nMulti Backtest Summary")
    print("======================")
    print(summary_df)

    print(f"\nSaved to: {output_path}")