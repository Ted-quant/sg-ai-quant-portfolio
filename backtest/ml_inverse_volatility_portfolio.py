"""
Day 24 — Inverse Volatility Portfolio Weighting

This script compares equal-weight portfolio allocation with inverse-volatility weighting.

Objective:
Test whether risk-adjusted portfolio construction improves Sharpe ratio,
maximum drawdown, and Calmar ratio during the 2025–2026 OOS period.

Portfolios:
1. Robust Assets Equal Weight ML Portfolio
2. Robust Assets Inverse Volatility ML Portfolio
3. Robust Assets Equal Weight Buy and Hold Portfolio
4. Robust Assets Inverse Volatility Buy and Hold Portfolio

Assets:
- D05_SI
- NVDA
- SPY

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

VOL_LOOKBACK = 60

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
    """Load asset-specific ML probability thresholds."""
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
    """Apply asset-specific threshold and regime filter."""
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
    """Generate daily ML net returns, benchmark returns, and volatility for one asset."""
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

    test["rolling_volatility"] = test["daily_return"].rolling(VOL_LOOKBACK).std() * np.sqrt(252)

    result = pd.DataFrame(
        {
            f"{asset_name}_ml_net_return": test["net_ml_return"],
            f"{asset_name}_benchmark_return": test["daily_return"],
            f"{asset_name}_rolling_volatility": test["rolling_volatility"],
            f"{asset_name}_position": test["position"],
        },
        index=test.index,
    )

    return result


def calculate_inverse_vol_weights(
    asset_results: pd.DataFrame,
    assets: list[str],
) -> pd.DataFrame:
    """Calculate dynamic inverse-volatility weights using rolling volatility."""
    vol_cols = [f"{asset}_rolling_volatility" for asset in assets]

    vol = asset_results[vol_cols].copy()
    vol.columns = assets

    inverse_vol = 1 / vol.replace(0, np.nan)
    weights = inverse_vol.div(inverse_vol.sum(axis=1), axis=0)

    return weights.dropna()


def build_equal_weight_portfolio(
    asset_results: pd.DataFrame,
    assets: list[str],
) -> pd.DataFrame:
    """Build equal-weight ML and benchmark portfolio returns."""
    ml_cols = [f"{asset}_ml_net_return" for asset in assets]
    benchmark_cols = [f"{asset}_benchmark_return" for asset in assets]
    position_cols = [f"{asset}_position" for asset in assets]

    portfolio = pd.DataFrame(index=asset_results.index)

    portfolio["ml_return"] = asset_results[ml_cols].mean(axis=1)
    portfolio["benchmark_return"] = asset_results[benchmark_cols].mean(axis=1)
    portfolio["average_market_exposure"] = asset_results[position_cols].mean(axis=1)

    return portfolio.dropna()


def build_inverse_vol_portfolio(
    asset_results: pd.DataFrame,
    assets: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build inverse-volatility weighted ML and benchmark portfolio returns."""
    weights = calculate_inverse_vol_weights(asset_results, assets)

    ml_returns = pd.DataFrame(
        {
            asset: asset_results[f"{asset}_ml_net_return"]
            for asset in assets
        }
    ).loc[weights.index]

    benchmark_returns = pd.DataFrame(
        {
            asset: asset_results[f"{asset}_benchmark_return"]
            for asset in assets
        }
    ).loc[weights.index]

    positions = pd.DataFrame(
        {
            asset: asset_results[f"{asset}_position"]
            for asset in assets
        }
    ).loc[weights.index]

    portfolio = pd.DataFrame(index=weights.index)

    portfolio["ml_return"] = (ml_returns * weights).sum(axis=1)
    portfolio["benchmark_return"] = (benchmark_returns * weights).sum(axis=1)
    portfolio["average_market_exposure"] = (positions * weights).sum(axis=1)

    return portfolio.dropna(), weights


def plot_portfolio_curves(daily_returns: pd.DataFrame) -> None:
    """Plot portfolio equity curves."""
    plt.figure(figsize=(11, 7))

    for col in daily_returns.columns:
        curve = (1 + daily_returns[col]).cumprod()
        plt.plot(curve.index, curve, label=col)

    plt.title("Day 24 — Equal Weight vs Inverse Volatility Portfolio")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = CHARTS_DIR / "inverse_volatility_portfolio_equity_curves.png"
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

    equal_portfolio = build_equal_weight_portfolio(asset_results, ROBUST_ASSETS)
    inverse_portfolio, weights = build_inverse_vol_portfolio(asset_results, ROBUST_ASSETS)

    common_index = equal_portfolio.index.intersection(inverse_portfolio.index)
    equal_portfolio = equal_portfolio.loc[common_index]
    inverse_portfolio = inverse_portfolio.loc[common_index]

    portfolios = {
        "equal_weight_ml_net": equal_portfolio["ml_return"],
        "equal_weight_buy_hold": equal_portfolio["benchmark_return"],
        "inverse_vol_ml_net": inverse_portfolio["ml_return"],
        "inverse_vol_buy_hold": inverse_portfolio["benchmark_return"],
    }

    summary_rows = []

    for name, returns in portfolios.items():
        metrics = calculate_metrics(returns)

        if "equal_weight" in name:
            exposure = equal_portfolio["average_market_exposure"].mean()
        else:
            exposure = inverse_portfolio["average_market_exposure"].mean()

        summary_rows.append(
            {
                "portfolio": name,
                "assets": ",".join(ROBUST_ASSETS),
                "average_market_exposure": exposure,
                **metrics,
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary_path = RESULTS_DIR / "ml_inverse_volatility_portfolio_summary.csv"
    summary.to_csv(summary_path, index=False)

    daily_returns = pd.DataFrame(
        {
            "equal_weight_ml_net": equal_portfolio["ml_return"],
            "equal_weight_buy_hold": equal_portfolio["benchmark_return"],
            "inverse_vol_ml_net": inverse_portfolio["ml_return"],
            "inverse_vol_buy_hold": inverse_portfolio["benchmark_return"],
        },
        index=common_index,
    )

    daily_path = RESULTS_DIR / "ml_inverse_volatility_portfolio_daily_returns.csv"
    daily_returns.to_csv(daily_path)

    weights_path = RESULTS_DIR / "ml_inverse_volatility_portfolio_weights.csv"
    weights.loc[common_index].to_csv(weights_path)

    plot_portfolio_curves(daily_returns)

    avg_weights = weights.loc[common_index].mean().reset_index()
    avg_weights.columns = ["asset", "average_inverse_vol_weight"]

    avg_weights_path = RESULTS_DIR / "ml_inverse_volatility_average_weights.csv"
    avg_weights.to_csv(avg_weights_path, index=False)

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

    print("\nDay 24 complete.")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved daily returns to: {daily_path}")
    print(f"Saved weights to: {weights_path}")
    print(f"Saved average weights to: {avg_weights_path}")

    print("\nPortfolio weighting summary:")
    print(summary[display_cols].to_string(index=False))

    print("\nAverage inverse-volatility weights:")
    print(avg_weights.to_string(index=False))


if __name__ == "__main__":
    main()
