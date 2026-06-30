# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Day 12 vectorbt moving average backtest
# Strategy: MA20 > MA60 long-only strategy on SPY

import os
import pandas as pd
import vectorbt as vbt


START_DATE = "2022-01-01"
END_DATE = "2024-12-31"
TICKER = "SPY"

INPUT_PATH = f"data/raw/{TICKER}_with_indicators.csv"
OUTPUT_FOLDER = "backtest/results"
CHART_FOLDER = "reports/charts"


def load_price_data(file_path):
    """
    Load local indicator data and apply the standardized date filter.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["MA20"] = pd.to_numeric(df["MA20"], errors="coerce")
    df["MA60"] = pd.to_numeric(df["MA60"], errors="coerce")

    df = df.dropna(subset=["Date", "Close", "MA20", "MA60"])
    df = df.sort_values("Date")

    df = df[
        (df["Date"] >= START_DATE) &
        (df["Date"] <= END_DATE)
    ].copy()

    if df.empty:
        raise ValueError("No data available after applying the date filter.")

    df = df.set_index("Date")

    return df


def run_vectorbt_backtest(df):
    """
    Run a simple long-only MA strategy using vectorbt.
    """

    close = df["Close"]

    entries = df["MA20"] > df["MA60"]
    exits = df["MA20"] <= df["MA60"]

    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=0.001,
        freq="1D"
    )

    return portfolio


if __name__ == "__main__":

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(CHART_FOLDER, exist_ok=True)

    price_df = load_price_data(INPUT_PATH)
    portfolio = run_vectorbt_backtest(price_df)

    stats = portfolio.stats()

    stats_output_path = f"{OUTPUT_FOLDER}/{TICKER}_vectorbt_ma_stats.csv"
    stats.to_csv(stats_output_path)

    html_output_path = f"{CHART_FOLDER}/{TICKER}_vectorbt_ma_equity_curve.html"
    portfolio.plot().write_html(html_output_path)

    print("\nDay 12 vectorbt backtest completed.")
    print("===================================")
    print(stats)

    print(f"\nSaved stats to: {stats_output_path}")
    print(f"Saved HTML chart to: {html_output_path}")
