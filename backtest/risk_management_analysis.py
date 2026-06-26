# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Risk management analysis for multiple strategies and assets

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

    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)


def calculate_var_95(daily_returns):
    """
    Calculate historical 95% Value at Risk.
    This is the 5th percentile of daily returns.
    """

    return daily_returns.quantile(0.05)


def calculate_expected_shortfall_95(daily_returns):
    """
    Calculate Expected Shortfall at 95%.
    This is the average return when returns are worse than VaR.
    """

    var_95 = calculate_var_95(daily_returns)
    tail_losses = daily_returns[daily_returns <= var_95]

    if tail_losses.empty:
        return 0

    return tail_losses.mean()


# =========================
# 2. Strategy rules
# =========================

def apply_strategy_rule(df, strategy_name):
    """
    Apply selected strategy rule and create position column.
    Position 1 means invested, 0 means cash.
    """

    if strategy_name == "Current_Defensive":
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

    elif strategy_name == "Buy_and_Hold":
        df["position"] = 1

    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")

    return df


# =========================
# 3. Risk analysis function
# =========================

def run_risk_analysis(ticker, strategy_name):
    """
    Run risk analysis for one ticker and one strategy.
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

    df = df.dropna(subset=["strategy_return"])

    df["cumulative_return"] = (1 + df["strategy_return"]).cumprod()

    total_return = df["cumulative_return"].iloc[-1] - 1
    max_drawdown = calculate_max_drawdown(df["cumulative_return"])
    sharpe_ratio = calculate_sharpe_ratio(df["strategy_return"])

    annualized_volatility = df["strategy_return"].std() * np.sqrt(252)
    var_95 = calculate_var_95(df["strategy_return"])
    expected_shortfall_95 = calculate_expected_shortfall_95(df["strategy_return"])

    exposure = df["position"].mean()

    result = {
        "Ticker": ticker,
        "Strategy": strategy_name,
        "Total Return": total_return,
        "Maximum Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio,
        "Annualized Volatility": annualized_volatility,
        "Daily VaR 95": var_95,
        "Daily Expected Shortfall 95": expected_shortfall_95,
        "Market Exposure": exposure
    }

    return result


# =========================
# 4. Run all risk analysis
# =========================

def run_all_risk_analysis(tickers, strategies):
    """
    Run risk analysis for all tickers and strategies.
    """

    results = []

    for ticker in tickers:
        for strategy_name in strategies:
            print(f"Running risk analysis for {ticker} - {strategy_name}...")

            result = run_risk_analysis(ticker, strategy_name)

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
        "Loose_RSI",
        "Breakout_60D"
    ]

    os.makedirs("backtest/results", exist_ok=True)

    risk_df = run_all_risk_analysis(tickers, strategies)

    output_path = "backtest/results/risk_management_summary.csv"

    risk_df.to_csv(output_path, index=False)

    print("\nRisk Management Summary")
    print("=======================")
    print(risk_df)

    print(f"\nSaved to: {output_path}")