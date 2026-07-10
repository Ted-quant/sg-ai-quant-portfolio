"""
Day 23 — Portfolio-Level Allocation Test

This script combines individual ML strategy returns into portfolio-level returns.

Objective:
Test whether the ML strategy works better as a portfolio-level allocation system
than as isolated single-asset strategies.

Portfolios:
1. All Assets ML Portfolio: D05_SI, MSFT, NVDA, QQQ, SPY
2. Robust Assets ML Portfolio: D05_SI, NVDA, SPY
3. All Assets Buy and Hold Portfolio
4. Robust Assets Buy and Hold Portfolio

The ML strategy uses:
- Asset-specific thresholds from Day 19
- Asset-specific regime filters from Day 21
- Transaction costs and slippage from Day 22
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

ROBUST_ASSETS = ["D05_SI", "NVDA", "SPY"]

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
    """Create technical features, targets, and regime filter variables."""
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


def calculate_metrics(returns: pd.Series) -> dict:
    """Calculate portfolio performance metrics."""
    returns = returns.dropna()

    total_return = (1 + returns).prod() - 1
    annualized_volatility = returns.std() * np.sqrt(252)

    sharpe = (
        returns.mean() / returns.std() * np.sqrt(252)
        if returns.std() != 0
        else np.nan
    )

    equity_curve = (1 + returns).cumprod()
    mdd = (equity_curve / equity_curve.cummax() - 1).min()

    calmar = total_return / abs(mdd) if mdd != 0 else np.nan

    var_95 = returns.quantile(0.05)
    expected_shortfall_95 = returns[returns <= var_95].mean()

    return {
        "total_return": total_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar_ratio": calmar,
        "daily_var_95": var_95,
        "daily_expected_shortfall_95": expected_shortfall_95,
    }


def load_thresholds() -> dict:
    """Load asset-specific thresholds."""
    if not BEST_THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Missing {BEST_THRESHOLD_PATH}. Run Day 19 first."
        )

    df = pd.read_csv(BEST_THRESHOLD_PATH)
    return {row["asset"]: float(row["threshold"]) for _, row in df.iterrows()}


def load_regime_filters() -> dict:
    """Load asset-specific regime filters."""
    if not BEST_REGIME_PATH.exists():
        raise FileNotFoundError(
            f"Missing {BEST_REGIME_PATH}. Run Day 21 first."
        )

    df = pd.read_csv(BEST_REGIME_PATH)

    filters = {}
    for _, row in df.iterrows():
        filters[row["asset"]] = {
            "filter_type": row["filter_type"],
            "ma_window": int(row["ma_window"]),
        }

    return filters


def apply_regime_filter(test: pd.DataFrame, threshold: float, ma_window: int) -> pd.Series:
    """Apply asset-specific ML threshold and regime filter."""
    base_signal = test["pred_prob"] >= threshold

    if ma_window == 0:
        return base_signal.astype(int)

    ma_col = f"ma_{ma_window}"
    return (base_signal & (test["Close"] > test[ma_col])).astype(int)


def run_asset_strategy(
    ticker: str,
    asset_name: str,
    threshold: float,
    regime_config: dict,
) -> pd.DataFrame:
    """Generate daily ML net returns and benchmark returns for one asset."""
    print(f"Running asset strategy for {asset_name}...")

    price = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )

    if price.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")

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
        raise RuntimeError(f"Not enough train/test data for {ticker}")

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

    test["signal"] = apply_regime_filter(test, threshold, ma_window)
    test["position"] = test["signal"].shift(1).fillna(0)

    test["trade"] = test["position"].diff().abs().fillna(test["position"].abs())
    test["gross_ml_return"] = test["position"] * test["daily_return"]
    test["trading_cost"] = test["trade"] * TOTAL_COST_RATE
    test["net_ml_return"] = test["gross_ml_return"] - test["trading_cost"]

    result = pd.DataFrame(
        {
            f"{asset_name}_ml_net_return": test["net_ml_return"],
            f"{asset_name}_benchmark_return": test["daily_return"],
            f"{asset_name}_position": test["position"],
        },
        index=test.index,
    )

    return result


def build_portfolio_returns(asset_results: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    """Build equal-weight ML and benchmark portfolio returns."""
    ml_cols = [f"{asset}_ml_net_return" for asset in assets]
    benchmark_cols = [f"{asset}_benchmark_return" for asset in assets]
    position_cols = [f"{asset}_position" for asset in assets]

    portfolio = pd.DataFrame(index=asset_results.index)

    portfolio["ml_portfolio_return"] = asset_results[ml_cols].mean(axis=1)
    portfolio["benchmark_portfolio_return"] = asset_results[benchmark_cols].mean(axis=1)
    portfolio["average_market_exposure"] = asset_results[position_cols].mean(axis=1)

    return portfolio.dropna()


def plot_portfolio_curves(portfolio_results: dict) -> None:
    """Plot portfolio equity curves."""
    plt.figure(figsize=(11, 7))

    for label, returns in portfolio_results.items():
        curve = (1 + returns).cumprod()
        plt.plot(curve.index, curve, label=label)

    plt.title("Portfolio-Level Allocation Test: ML Strategy vs Buy and Hold")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = CHARTS_DIR / "portfolio_allocation_test_equity_curves.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    thresholds = load_thresholds()
    regime_filters = load_regime_filters()

    all_asset_results = []

    for ticker, asset_name in ASSETS.items():
        result = run_asset_strategy(
            ticker=ticker,
            asset_name=asset_name,
            threshold=thresholds[asset_name],
            regime_config=regime_filters[asset_name],
        )
        all_asset_results.append(result)

    asset_results = pd.concat(all_asset_results, axis=1).dropna()

    all_assets = list(ASSETS.values())

    all_portfolio = build_portfolio_returns(asset_results, all_assets)
    robust_portfolio = build_portfolio_returns(asset_results, ROBUST_ASSETS)

    portfolio_summary_rows = []

    portfolios = {
        "all_assets_ml_net": all_portfolio["ml_portfolio_return"],
        "all_assets_buy_hold": all_portfolio["benchmark_portfolio_return"],
        "robust_assets_ml_net": robust_portfolio["ml_portfolio_return"],
        "robust_assets_buy_hold": robust_portfolio["benchmark_portfolio_return"],
    }

    for name, returns in portfolios.items():
        metrics = calculate_metrics(returns)

        if "all_assets" in name:
            exposure = all_portfolio["average_market_exposure"].mean()
        else:
            exposure = robust_portfolio["average_market_exposure"].mean()

        portfolio_summary_rows.append(
            {
                "portfolio": name,
                "assets": "all" if "all_assets" in name else "D05_SI,NVDA,SPY",
                "average_market_exposure": exposure,
                **metrics,
            }
        )

    summary = pd.DataFrame(portfolio_summary_rows)

    summary_path = RESULTS_DIR / "ml_portfolio_allocation_summary.csv"
    summary.to_csv(summary_path, index=False)

    daily_returns = pd.DataFrame(
        {
            "all_assets_ml_net": all_portfolio["ml_portfolio_return"],
            "all_assets_buy_hold": all_portfolio["benchmark_portfolio_return"],
            "robust_assets_ml_net": robust_portfolio["ml_portfolio_return"],
            "robust_assets_buy_hold": robust_portfolio["benchmark_portfolio_return"],
        }
    ).dropna()

    daily_path = RESULTS_DIR / "ml_portfolio_allocation_daily_returns.csv"
    daily_returns.to_csv(daily_path)

    plot_portfolio_curves(daily_returns.to_dict(orient="series"))

    display_cols = [
        "portfolio",
        "assets",
        "total_return",
        "annualized_volatility",
        "sharpe",
        "mdd",
        "calmar_ratio",
        "daily_var_95",
        "daily_expected_shortfall_95",
        "average_market_exposure",
    ]

    print("\nDay 23 complete.")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved daily returns to: {daily_path}")
    print("\nPortfolio allocation summary:")
    print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
