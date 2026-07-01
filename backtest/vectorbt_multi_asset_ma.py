# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Day 13 vectorbt multi-asset MA20/MA60 backtest
# Strategy: Long when MA20 > MA60, exit when MA20 <= MA60

import os
import platform
import subprocess
import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt


START_DATE = "2022-01-01"
END_DATE = "2024-12-31"

TICKERS = [
    "NVDA",
    "MSFT",
    "SPY",
    "QQQ",
    "D05_SI"
]

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


def load_price_data(ticker):
    """
    Load local indicator data for one ticker.
    """

    input_path = f"data/raw/{ticker}_with_indicators.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

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
        raise ValueError(f"No data available for {ticker} after applying the date filter.")

    df = df.set_index("Date")

    return df


def run_vectorbt_backtest(df):
    """
    Run vectorbt MA20/MA60 long-only strategy.
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


def extract_key_stats(ticker, portfolio):
    """
    Extract key portfolio statistics for comparison.
    """

    stats = portfolio.stats()

    result = {
        "Ticker": ticker,
        "Start": stats.get("Start"),
        "End": stats.get("End"),
        "Total Return [%]": stats.get("Total Return [%]"),
        "Benchmark Return [%]": stats.get("Benchmark Return [%]"),
        "Max Drawdown [%]": stats.get("Max Drawdown [%]"),
        "Sharpe Ratio": stats.get("Sharpe Ratio"),
        "Sortino Ratio": stats.get("Sortino Ratio"),
        "Calmar Ratio": stats.get("Calmar Ratio"),
        "Total Trades": stats.get("Total Trades"),
        "Win Rate [%]": stats.get("Win Rate [%]"),
        "Total Fees Paid": stats.get("Total Fees Paid")
    }

    return result


def plot_bar_chart(df, metric, output_path, title, ylabel):
    """
    Create a bar chart for one metric.
    """

    plot_df = df.copy()
    plot_df = plot_df.sort_values(metric, ascending=False)

    ax = plot_df.plot(
        kind="bar",
        x="Ticker",
        y=metric,
        legend=False,
        figsize=(10, 6)
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

    all_results = []

    for ticker in TICKERS:
        print(f"Running vectorbt backtest for {ticker}...")

        price_df = load_price_data(ticker)
        portfolio = run_vectorbt_backtest(price_df)

        result = extract_key_stats(ticker, portfolio)
        all_results.append(result)

    summary_df = pd.DataFrame(all_results)

    output_path = f"{OUTPUT_FOLDER}/vectorbt_multi_asset_ma_summary.csv"
    summary_df.to_csv(output_path, index=False)

    print("\nDay 13 vectorbt multi-asset backtest completed.")
    print("==============================================")
    print(summary_df)

    print(f"\nSaved summary to: {output_path}")

    plot_bar_chart(
        df=summary_df,
        metric="Total Return [%]",
        output_path=f"{CHART_FOLDER}/vectorbt_multi_asset_return_comparison.png",
        title="Vectorbt MA20/MA60 Strategy: Total Return by Asset",
        ylabel="Total Return (%)"
    )

    plot_bar_chart(
        df=summary_df,
        metric="Benchmark Return [%]",
        output_path=f"{CHART_FOLDER}/vectorbt_multi_asset_benchmark_return_comparison.png",
        title="Buy and Hold Benchmark Return by Asset",
        ylabel="Benchmark Return (%)"
    )

    plot_bar_chart(
        df=summary_df,
        metric="Max Drawdown [%]",
        output_path=f"{CHART_FOLDER}/vectorbt_multi_asset_mdd_comparison.png",
        title="Vectorbt MA20/MA60 Strategy: Maximum Drawdown by Asset",
        ylabel="Maximum Drawdown (%)"
    )

    plot_bar_chart(
        df=summary_df,
        metric="Sharpe Ratio",
        output_path=f"{CHART_FOLDER}/vectorbt_multi_asset_sharpe_comparison.png",
        title="Vectorbt MA20/MA60 Strategy: Sharpe Ratio by Asset",
        ylabel="Sharpe Ratio"
    )
