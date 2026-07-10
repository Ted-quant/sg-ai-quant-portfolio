"""
Day 20 — Asset-Specific Threshold + Regime Filter Test

This script compares:
1. Asset-specific ML threshold strategy
2. Asset-specific ML threshold + 200-day moving average regime filter

Objective:
Test whether adding a market regime filter improves drawdown and risk-adjusted performance
during the 2025–2026 final out-of-sample period.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


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

RESULTS_DIR = Path("backtest/results")
CHARTS_DIR = Path("reports/charts")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_THRESHOLD_PATH = RESULTS_DIR / "ml_final_oos_2025_2026_best_thresholds.csv"


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
    """Create model features, target, and regime filter variable."""
    df = price.copy()

    df["daily_return"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)

    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_60"] = df["Close"].rolling(60).mean()
    df["ma_200"] = df["Close"].rolling(200).mean()

    df["ma10_ratio"] = df["Close"] / df["ma_10"] - 1
    df["ma20_ratio"] = df["Close"] / df["ma_20"] - 1
    df["ma60_ratio"] = df["Close"] / df["ma_60"] - 1

    df["volatility_20d"] = df["daily_return"].rolling(20).std()
    df["rsi_14"] = calculate_rsi(df["Close"], 14)

    df["regime_filter"] = (df["Close"] > df["ma_200"]).astype(int)

    df["forward_5d_return"] = df["Close"].pct_change(5).shift(-5)
    df["target"] = (df["forward_5d_return"] > 0.01).astype(int)

    df = df.dropna()

    return df


def calculate_metrics(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    """Calculate performance and risk metrics."""
    strategy_returns = strategy_returns.dropna()
    benchmark_returns = benchmark_returns.loc[strategy_returns.index].dropna()

    total_return = (1 + strategy_returns).prod() - 1
    benchmark_return = (1 + benchmark_returns).prod() - 1

    volatility = strategy_returns.std() * np.sqrt(252)
    benchmark_volatility = benchmark_returns.std() * np.sqrt(252)

    sharpe = (
        strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        if strategy_returns.std() != 0
        else np.nan
    )

    benchmark_sharpe = (
        benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)
        if benchmark_returns.std() != 0
        else np.nan
    )

    equity_curve = (1 + strategy_returns).cumprod()
    benchmark_curve = (1 + benchmark_returns).cumprod()

    mdd = (equity_curve / equity_curve.cummax() - 1).min()
    benchmark_mdd = (benchmark_curve / benchmark_curve.cummax() - 1).min()

    return {
        "strategy_return": total_return,
        "benchmark_return": benchmark_return,
        "excess_return": total_return - benchmark_return,
        "strategy_sharpe": sharpe,
        "benchmark_sharpe": benchmark_sharpe,
        "strategy_volatility": volatility,
        "benchmark_volatility": benchmark_volatility,
        "strategy_mdd": mdd,
        "benchmark_mdd": benchmark_mdd,
    }


def load_asset_thresholds() -> dict:
    """Load asset-specific thresholds from Day 19 final OOS result."""
    if not BEST_THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Missing {BEST_THRESHOLD_PATH}. Run Day 19 final OOS test first."
        )

    threshold_df = pd.read_csv(BEST_THRESHOLD_PATH)

    thresholds = {
        row["asset"]: float(row["threshold"])
        for _, row in threshold_df.iterrows()
    }

    return thresholds


def run_asset_test(ticker: str, asset_name: str, threshold: float) -> pd.DataFrame:
    """Run threshold-only and threshold + regime filter tests for one asset."""
    print(f"\nRunning Day 20 test for {asset_name} using threshold {threshold:.2f}...")

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

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(train[feature_cols], train["target"])

    test["pred_prob"] = model.predict_proba(test[feature_cols])[:, 1]

    test["threshold_signal"] = (test["pred_prob"] >= threshold).astype(int)
    test["threshold_position"] = test["threshold_signal"].shift(1).fillna(0)
    test["threshold_strategy_return"] = (
        test["threshold_position"] * test["daily_return"]
    )

    test["regime_signal"] = (
        (test["pred_prob"] >= threshold) & (test["regime_filter"] == 1)
    ).astype(int)
    test["regime_position"] = test["regime_signal"].shift(1).fillna(0)
    test["regime_strategy_return"] = (
        test["regime_position"] * test["daily_return"]
    )

    threshold_metrics = calculate_metrics(
        test["threshold_strategy_return"],
        test["daily_return"],
    )

    regime_metrics = calculate_metrics(
        test["regime_strategy_return"],
        test["daily_return"],
    )

    rows = [
        {
            "asset": asset_name,
            "ticker": ticker,
            "test_period": "2025-2026",
            "strategy_type": "threshold_only",
            "threshold": threshold,
            "market_exposure": test["threshold_position"].mean(),
            **threshold_metrics,
        },
        {
            "asset": asset_name,
            "ticker": ticker,
            "test_period": "2025-2026",
            "strategy_type": "threshold_plus_regime_filter",
            "threshold": threshold,
            "market_exposure": test["regime_position"].mean(),
            **regime_metrics,
        },
    ]

    result = pd.DataFrame(rows)

    plot_equity_curves(test, asset_name)

    return result


def plot_equity_curves(test: pd.DataFrame, asset_name: str) -> None:
    """Plot benchmark, threshold-only, and regime-filter equity curves."""
    benchmark_curve = (1 + test["daily_return"]).cumprod()
    threshold_curve = (1 + test["threshold_strategy_return"]).cumprod()
    regime_curve = (1 + test["regime_strategy_return"]).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(benchmark_curve.index, benchmark_curve, label="Buy and Hold")
    plt.plot(threshold_curve.index, threshold_curve, label="Threshold Only")
    plt.plot(regime_curve.index, regime_curve, label="Threshold + Regime Filter")

    plt.title(f"{asset_name} — Threshold vs Regime Filter OOS Test")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = CHARTS_DIR / f"{asset_name}_threshold_vs_regime_filter.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    thresholds = load_asset_thresholds()

    all_results = []

    for ticker, asset_name in ASSETS.items():
        if asset_name not in thresholds:
            print(f"Warning: Missing threshold for {asset_name}")
            continue

        result = run_asset_test(ticker, asset_name, thresholds[asset_name])

        if not result.empty:
            all_results.append(result)

    if not all_results:
        raise RuntimeError("No Day 20 results were generated.")

    final_result = pd.concat(all_results, ignore_index=True)

    output_path = RESULTS_DIR / "ml_asset_threshold_regime_filter_summary.csv"
    final_result.to_csv(output_path, index=False)

    comparison = final_result[
        [
            "asset",
            "strategy_type",
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
    ].copy()

    print("\nDay 20 complete.")
    print(f"Saved result to: {output_path}")
    print("\nStrategy comparison:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
