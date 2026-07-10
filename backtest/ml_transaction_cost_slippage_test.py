"""
Day 22 — Transaction Costs and Slippage Test

This script tests whether the ML + regime filter strategy remains robust
after adding transaction costs and slippage assumptions.

Objective:
Compare gross returns vs net returns after trading frictions.

Trading friction assumptions:
- Transaction cost: 5 bps per trade
- Slippage: 5 bps per trade
- Total cost: 10 bps per position change

Strategy:
- Use asset-specific probability thresholds from Day 19
- Use asset-specific best regime filters from Day 21
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

TRANSACTION_COST_BPS = 5
SLIPPAGE_BPS = 5
TOTAL_COST_RATE = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000

RESULTS_DIR = Path("backtest/results")
CHARTS_DIR = Path("reports/charts")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_THRESHOLD_PATH = RESULTS_DIR / "ml_final_oos_2025_2026_best_thresholds.csv"
BEST_REGIME_PATH = RESULTS_DIR / "ml_regime_filter_sensitivity_best.csv"


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
    """Create model features, targets, and regime filter variables."""
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


def calculate_metrics(returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    """Calculate performance and risk metrics."""
    returns = returns.dropna()
    benchmark_returns = benchmark_returns.loc[returns.index].dropna()

    total_return = (1 + returns).prod() - 1
    benchmark_return = (1 + benchmark_returns).prod() - 1

    volatility = returns.std() * np.sqrt(252)
    benchmark_volatility = benchmark_returns.std() * np.sqrt(252)

    sharpe = (
        returns.mean() / returns.std() * np.sqrt(252)
        if returns.std() != 0
        else np.nan
    )

    benchmark_sharpe = (
        benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)
        if benchmark_returns.std() != 0
        else np.nan
    )

    equity_curve = (1 + returns).cumprod()
    benchmark_curve = (1 + benchmark_returns).cumprod()

    mdd = (equity_curve / equity_curve.cummax() - 1).min()
    benchmark_mdd = (benchmark_curve / benchmark_curve.cummax() - 1).min()

    calmar = total_return / abs(mdd) if mdd != 0 else np.nan

    return {
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "excess_return": total_return - benchmark_return,
        "sharpe": sharpe,
        "benchmark_sharpe": benchmark_sharpe,
        "volatility": volatility,
        "benchmark_volatility": benchmark_volatility,
        "mdd": mdd,
        "benchmark_mdd": benchmark_mdd,
        "calmar_ratio": calmar,
    }


def load_thresholds() -> dict:
    """Load asset-specific probability thresholds."""
    if not BEST_THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Missing {BEST_THRESHOLD_PATH}. Run Day 19 final OOS test first."
        )

    df = pd.read_csv(BEST_THRESHOLD_PATH)
    return {row["asset"]: float(row["threshold"]) for _, row in df.iterrows()}


def load_regime_filters() -> dict:
    """Load asset-specific best regime filters from Day 21."""
    if not BEST_REGIME_PATH.exists():
        raise FileNotFoundError(
            f"Missing {BEST_REGIME_PATH}. Run Day 21 regime sensitivity test first."
        )

    df = pd.read_csv(BEST_REGIME_PATH)

    filters = {}
    for _, row in df.iterrows():
        asset = row["asset"]
        ma_window = int(row["ma_window"])
        filter_type = row["filter_type"]
        filters[asset] = {
            "filter_type": filter_type,
            "ma_window": ma_window,
        }

    return filters


def apply_regime_filter(test: pd.DataFrame, threshold: float, ma_window: int) -> pd.Series:
    """Create final signal using threshold and selected regime filter."""
    base_signal = test["pred_prob"] >= threshold

    if ma_window == 0:
        return base_signal.astype(int)

    ma_col = f"ma_{ma_window}"
    regime_signal = base_signal & (test["Close"] > test[ma_col])

    return regime_signal.astype(int)


def run_asset_test(
    ticker: str,
    asset_name: str,
    threshold: float,
    regime_config: dict,
) -> pd.DataFrame:
    """Run gross vs net return test for one asset."""
    print(f"\nRunning Day 22 cost test for {asset_name}...")

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

    ma_window = regime_config["ma_window"]
    filter_type = regime_config["filter_type"]

    test["signal"] = apply_regime_filter(test, threshold, ma_window)
    test["position"] = test["signal"].shift(1).fillna(0)

    test["trade"] = test["position"].diff().abs().fillna(test["position"].abs())

    test["gross_strategy_return"] = test["position"] * test["daily_return"]
    test["trading_cost"] = test["trade"] * TOTAL_COST_RATE
    test["net_strategy_return"] = (
        test["gross_strategy_return"] - test["trading_cost"]
    )

    gross_metrics = calculate_metrics(
        test["gross_strategy_return"],
        test["daily_return"],
    )

    net_metrics = calculate_metrics(
        test["net_strategy_return"],
        test["daily_return"],
    )

    turnover = test["trade"].sum()
    avg_daily_turnover = test["trade"].mean()
    total_cost = test["trading_cost"].sum()

    rows = [
        {
            "asset": asset_name,
            "ticker": ticker,
            "test_period": "2025-2026",
            "cost_model": "gross_no_cost",
            "threshold": threshold,
            "filter_type": filter_type,
            "ma_window": ma_window,
            "market_exposure": test["position"].mean(),
            "total_turnover": turnover,
            "avg_daily_turnover": avg_daily_turnover,
            "total_cost_drag": 0.0,
            **gross_metrics,
        },
        {
            "asset": asset_name,
            "ticker": ticker,
            "test_period": "2025-2026",
            "cost_model": "net_after_cost_slippage",
            "threshold": threshold,
            "filter_type": filter_type,
            "ma_window": ma_window,
            "market_exposure": test["position"].mean(),
            "total_turnover": turnover,
            "avg_daily_turnover": avg_daily_turnover,
            "total_cost_drag": total_cost,
            **net_metrics,
        },
    ]

    plot_gross_vs_net(test, asset_name)

    return pd.DataFrame(rows)


def plot_gross_vs_net(test: pd.DataFrame, asset_name: str) -> None:
    """Plot gross vs net equity curves."""
    benchmark_curve = (1 + test["daily_return"]).cumprod()
    gross_curve = (1 + test["gross_strategy_return"]).cumprod()
    net_curve = (1 + test["net_strategy_return"]).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(benchmark_curve.index, benchmark_curve, label="Buy and Hold")
    plt.plot(gross_curve.index, gross_curve, label="Gross Strategy")
    plt.plot(net_curve.index, net_curve, label="Net Strategy After Costs")

    plt.title(f"{asset_name} — Gross vs Net Performance After Costs")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = CHARTS_DIR / f"{asset_name}_transaction_cost_slippage_test.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    thresholds = load_thresholds()
    regime_filters = load_regime_filters()

    all_results = []

    for ticker, asset_name in ASSETS.items():
        if asset_name not in thresholds:
            print(f"Warning: Missing threshold for {asset_name}")
            continue

        if asset_name not in regime_filters:
            print(f"Warning: Missing regime filter for {asset_name}")
            continue

        result = run_asset_test(
            ticker=ticker,
            asset_name=asset_name,
            threshold=thresholds[asset_name],
            regime_config=regime_filters[asset_name],
        )

        if not result.empty:
            all_results.append(result)

    if not all_results:
        raise RuntimeError("No Day 22 results were generated.")

    final_result = pd.concat(all_results, ignore_index=True)

    output_path = RESULTS_DIR / "ml_transaction_cost_slippage_summary.csv"
    final_result.to_csv(output_path, index=False)

    display_cols = [
        "asset",
        "cost_model",
        "filter_type",
        "ma_window",
        "threshold",
        "total_return",
        "benchmark_return",
        "excess_return",
        "sharpe",
        "mdd",
        "calmar_ratio",
        "market_exposure",
        "total_turnover",
        "total_cost_drag",
    ]

    print("\nDay 22 complete.")
    print(f"Saved result to: {output_path}")
    print("\nGross vs net comparison:")
    print(final_result[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
