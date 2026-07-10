"""
Day 21 — Regime Filter Sensitivity Test

This script compares different moving average regime filters:

1. No Filter
2. MA100 Filter
3. MA150 Filter
4. MA200 Filter

Objective:
Find which regime filter provides the best balance between return,
maximum drawdown, Sharpe ratio, and market exposure during the 2025–2026 OOS period.
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

REGIME_WINDOWS = {
    "no_filter": None,
    "ma100_filter": 100,
    "ma150_filter": 150,
    "ma200_filter": 200,
}

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
    """Create technical features, target, and multiple regime filters."""
    df = price.copy()

    df["daily_return"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)

    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_60"] = df["Close"].rolling(60).mean()
    df["ma_100"] = df["Close"].rolling(100).mean()
    df["ma_150"] = df["Close"].rolling(150).mean()
    df["ma_200"] = df["Close"].rolling(200).mean()

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

    strategy_return = (1 + strategy_returns).prod() - 1
    benchmark_return = (1 + benchmark_returns).prod() - 1

    strategy_volatility = strategy_returns.std() * np.sqrt(252)
    benchmark_volatility = benchmark_returns.std() * np.sqrt(252)

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

    calmar = (
        strategy_return / abs(strategy_mdd)
        if strategy_mdd != 0
        else np.nan
    )

    return {
        "strategy_return": strategy_return,
        "benchmark_return": benchmark_return,
        "excess_return": strategy_return - benchmark_return,
        "strategy_sharpe": strategy_sharpe,
        "benchmark_sharpe": benchmark_sharpe,
        "strategy_volatility": strategy_volatility,
        "benchmark_volatility": benchmark_volatility,
        "strategy_mdd": strategy_mdd,
        "benchmark_mdd": benchmark_mdd,
        "calmar_ratio": calmar,
    }


def load_asset_thresholds() -> dict:
    """Load asset-specific thresholds from Day 19 final OOS results."""
    if not BEST_THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Missing {BEST_THRESHOLD_PATH}. Run Day 19 final OOS test first."
        )

    threshold_df = pd.read_csv(BEST_THRESHOLD_PATH)

    return {
        row["asset"]: float(row["threshold"])
        for _, row in threshold_df.iterrows()
    }


def run_asset_test(ticker: str, asset_name: str, threshold: float) -> pd.DataFrame:
    """Run regime filter sensitivity test for one asset."""
    print(f"\nRunning Day 21 test for {asset_name} using threshold {threshold:.2f}...")

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
    test["base_signal"] = (test["pred_prob"] >= threshold).astype(int)

    rows = []
    equity_curves = {}

    benchmark_curve = (1 + test["daily_return"]).cumprod()
    equity_curves["Buy and Hold"] = benchmark_curve

    for filter_name, window in REGIME_WINDOWS.items():
        temp = test.copy()

        if window is None:
            temp["final_signal"] = temp["base_signal"]
        else:
            ma_col = f"ma_{window}"
            temp["final_signal"] = (
                (temp["base_signal"] == 1) & (temp["Close"] > temp[ma_col])
            ).astype(int)

        temp["position"] = temp["final_signal"].shift(1).fillna(0)
        temp["strategy_return_daily"] = temp["position"] * temp["daily_return"]

        metrics = calculate_metrics(
            temp["strategy_return_daily"],
            temp["daily_return"],
        )

        equity_curves[filter_name] = (1 + temp["strategy_return_daily"]).cumprod()

        rows.append(
            {
                "asset": asset_name,
                "ticker": ticker,
                "test_period": "2025-2026",
                "filter_type": filter_name,
                "ma_window": window if window is not None else 0,
                "threshold": threshold,
                "market_exposure": temp["position"].mean(),
                **metrics,
            }
        )

    plot_equity_curves(equity_curves, asset_name)

    return pd.DataFrame(rows)


def plot_equity_curves(equity_curves: dict, asset_name: str) -> None:
    """Plot benchmark and regime filter equity curves."""
    plt.figure(figsize=(10, 6))

    for label, curve in equity_curves.items():
        plt.plot(curve.index, curve, label=label)

    plt.title(f"{asset_name} — Regime Filter Sensitivity Test")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = CHARTS_DIR / f"{asset_name}_regime_filter_sensitivity.png"
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
        raise RuntimeError("No Day 21 results were generated.")

    final_result = pd.concat(all_results, ignore_index=True)

    output_path = RESULTS_DIR / "ml_regime_filter_sensitivity_summary.csv"
    final_result.to_csv(output_path, index=False)

    best_by_asset = (
        final_result.sort_values(
            ["asset", "calmar_ratio", "strategy_sharpe", "strategy_return"],
            ascending=[True, False, False, False],
        )
        .groupby("asset")
        .head(1)
        .reset_index(drop=True)
    )

    best_output_path = RESULTS_DIR / "ml_regime_filter_sensitivity_best.csv"
    best_by_asset.to_csv(best_output_path, index=False)

    display_cols = [
        "asset",
        "filter_type",
        "ma_window",
        "threshold",
        "strategy_return",
        "benchmark_return",
        "excess_return",
        "strategy_sharpe",
        "strategy_mdd",
        "calmar_ratio",
        "market_exposure",
    ]

    print("\nDay 21 complete.")
    print(f"Saved full result to: {output_path}")
    print(f"Saved best filter result to: {best_output_path}")

    print("\nBest regime filter by asset using Calmar ratio:")
    print(best_by_asset[display_cols].to_string(index=False))

    print("\nFull strategy comparison:")
    print(final_result[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
