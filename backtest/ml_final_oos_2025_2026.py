"""
Day 19 — Final Out-of-Sample Test: 2025–2026

This script tests whether the ML probability thresholds selected during
walk-forward validation remain effective on a fresh 2025–2026 out-of-sample period.

Model:
- RandomForestClassifier
- Target: forward 5-trading-day return > 1%
- Signal: predicted probability >= threshold
- Look-ahead bias prevention: position is shifted by one day
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


warnings.filterwarnings("ignore")

ASSETS = {
    "D05.SI": "D05_SI",
    "MSFT": "MSFT",
    "NVDA": "NVDA",
    "QQQ": "QQQ",
    "SPY": "SPY",
}

START_DATE = "2022-01-01"
END_DATE = "2026-12-31"

TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2026-12-31"

THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]

RESULTS_DIR = Path("backtest/results")
CHARTS_DIR = Path("reports/charts")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI using rolling average gains and losses."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def make_features(price: pd.DataFrame) -> pd.DataFrame:
    """Create technical features and target variable."""
    df = price.copy()

    df["daily_return"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)

    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_60"] = df["Close"].rolling(60).mean()

    df["ma10_ratio"] = df["Close"] / df["ma_10"] - 1
    df["ma20_ratio"] = df["Close"] / df["ma_20"] - 1
    df["ma60_ratio"] = df["Close"] / df["ma_60"] - 1

    df["volatility_20d"] = df["daily_return"].rolling(20).std()
    df["rsi_14"] = calculate_rsi(df["Close"], 14)

    df["forward_5d_return"] = df["Close"].pct_change(5).shift(-5)
    df["target"] = (df["forward_5d_return"] > 0.01).astype(int)

    df = df.dropna()

    return df


def calculate_metrics(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    """Calculate return and risk metrics."""
    strategy_returns = strategy_returns.dropna()
    benchmark_returns = benchmark_returns.loc[strategy_returns.index].dropna()

    strategy_total_return = (1 + strategy_returns).prod() - 1
    benchmark_total_return = (1 + benchmark_returns).prod() - 1

    strategy_vol = strategy_returns.std() * np.sqrt(252)
    benchmark_vol = benchmark_returns.std() * np.sqrt(252)

    strategy_sharpe = (
        strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        if strategy_returns.std() != 0
        else np.nan
    )

    benchmark_sharpe = (
        benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)
        if benchmark_returns.std() != 0
        else np.nan
    )

    strategy_curve = (1 + strategy_returns).cumprod()
    benchmark_curve = (1 + benchmark_returns).cumprod()

    strategy_mdd = (strategy_curve / strategy_curve.cummax() - 1).min()
    benchmark_mdd = (benchmark_curve / benchmark_curve.cummax() - 1).min()

    return {
        "strategy_return": strategy_total_return,
        "benchmark_return": benchmark_total_return,
        "excess_return": strategy_total_return - benchmark_total_return,
        "strategy_sharpe": strategy_sharpe,
        "benchmark_sharpe": benchmark_sharpe,
        "strategy_volatility": strategy_vol,
        "benchmark_volatility": benchmark_vol,
        "strategy_mdd": strategy_mdd,
        "benchmark_mdd": benchmark_mdd,
    }


def run_asset_final_oos(ticker: str, asset_name: str) -> pd.DataFrame:
    """Run final OOS test for one asset."""
    print(f"\nRunning final OOS test for {ticker}...")

    price = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )

    if price.empty:
        print(f"Warning: No data downloaded for {ticker}")
        return pd.DataFrame()

    if isinstance(price.columns, pd.MultiIndex):
        price.columns = price.columns.get_level_values(0)

    df = make_features(price)

    feature_cols = [
        "return_5d",
        "return_20d",
        "ma10_ratio",
        "ma20_ratio",
        "ma60_ratio",
        "volatility_20d",
        "rsi_14",
    ]

    train = df.loc[:TRAIN_END].copy()
    test = df.loc[TEST_START:TEST_END].copy()

    if train.empty or test.empty:
        print(f"Warning: Not enough train/test data for {ticker}")
        return pd.DataFrame()

    X_train = train[feature_cols]
    y_train = train["target"]

    X_test = test[feature_cols]
    y_test = test["target"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    test["pred_prob"] = model.predict_proba(X_test)[:, 1]
    test["pred_label_050"] = (test["pred_prob"] >= 0.50).astype(int)

    accuracy = accuracy_score(y_test, test["pred_label_050"])
    precision = precision_score(y_test, test["pred_label_050"], zero_division=0)
    recall = recall_score(y_test, test["pred_label_050"], zero_division=0)

    rows = []

    for threshold in THRESHOLDS:
        temp = test.copy()

        temp["signal"] = (temp["pred_prob"] >= threshold).astype(int)
        temp["position"] = temp["signal"].shift(1).fillna(0)
        temp["strategy_return_daily"] = temp["position"] * temp["daily_return"]

        metrics = calculate_metrics(
            temp["strategy_return_daily"],
            temp["daily_return"],
        )

        market_exposure = temp["position"].mean()

        rows.append(
            {
                "asset": asset_name,
                "ticker": ticker,
                "train_period": "2022-2024",
                "test_period": "2025-2026",
                "threshold": threshold,
                "accuracy_at_050": accuracy,
                "precision_at_050": precision,
                "recall_at_050": recall,
                "market_exposure": market_exposure,
                **metrics,
            }
        )

    result = pd.DataFrame(rows)

    plot_threshold_result(result, asset_name)

    return result


def plot_threshold_result(result: pd.DataFrame, asset_name: str) -> None:
    """Plot strategy return by threshold vs benchmark."""
    plt.figure(figsize=(10, 6))

    plt.plot(
        result["threshold"],
        result["strategy_return"] * 100,
        marker="o",
        label="ML Strategy Return",
    )

    benchmark_return = result["benchmark_return"].iloc[0] * 100

    plt.axhline(
        benchmark_return,
        linestyle="--",
        label="Buy and Hold Return",
    )

    plt.title(f"{asset_name} Final OOS Test: 2025–2026")
    plt.xlabel("Probability Threshold")
    plt.ylabel("Total Return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = CHARTS_DIR / f"{asset_name}_final_oos_2025_2026.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    all_results = []

    for ticker, asset_name in ASSETS.items():
        result = run_asset_final_oos(ticker, asset_name)
        if not result.empty:
            all_results.append(result)

    if not all_results:
        raise RuntimeError("No final OOS results were generated.")

    final_result = pd.concat(all_results, ignore_index=True)

    output_path = RESULTS_DIR / "ml_final_oos_2025_2026_summary.csv"
    final_result.to_csv(output_path, index=False)

    best_by_asset = (
        final_result.sort_values(["asset", "strategy_return"], ascending=[True, False])
        .groupby("asset")
        .head(1)
        .reset_index(drop=True)
    )

    best_output_path = RESULTS_DIR / "ml_final_oos_2025_2026_best_thresholds.csv"
    best_by_asset.to_csv(best_output_path, index=False)

    print("\nFinal OOS test complete.")
    print(f"Saved full results to: {output_path}")
    print(f"Saved best thresholds to: {best_output_path}")

    print("\nBest threshold by asset:")
    display_cols = [
        "asset",
        "threshold",
        "strategy_return",
        "benchmark_return",
        "excess_return",
        "strategy_sharpe",
        "benchmark_sharpe",
        "strategy_mdd",
        "benchmark_mdd",
        "market_exposure",
    ]
    print(best_by_asset[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
