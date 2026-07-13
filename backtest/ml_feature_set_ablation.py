import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier


# =========================
# Configuration
# =========================

ASSETS = {
    "D05_SI": "D05.SI",
    "MSFT": "MSFT",
    "NVDA": "NVDA",
    "QQQ": "QQQ",
    "SPY": "SPY",
}

ROBUST_ASSETS = ["D05_SI", "NVDA", "SPY"]

START_DATE = "2020-01-01"
TRAIN_END = "2023-12-31"
VALID_START = "2024-01-01"
VALID_END = "2024-12-31"
TEST_START = "2025-01-01"

THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]
REGIME_FILTERS = ["NO_FILTER", "MA100", "MA150", "MA200"]

TOTAL_COST_PER_POSITION_CHANGE = 0.001

RESULT_DIR = Path("backtest/results")
CHART_DIR = Path("reports/charts")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_SETS = {
    "Full": [
        "return_5d",
        "return_20d",
        "ma10_ratio",
        "ma20_ratio",
        "ma60_ratio",
        "volatility_20d",
        "rsi_14",
    ],
    "Top3": [
        "ma60_ratio",
        "volatility_20d",
        "rsi_14",
    ],
    "RiskAware": [
        "ma60_ratio",
        "volatility_20d",
        "rsi_14",
        "ma20_ratio",
    ],
    "Trend": [
        "ma10_ratio",
        "ma20_ratio",
        "ma60_ratio",
        "return_20d",
    ],
    "Momentum": [
        "return_5d",
        "return_20d",
        "ma10_ratio",
        "ma20_ratio",
    ],
}


# =========================
# Helper Functions
# =========================

def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI using rolling average gains and losses."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def download_price_data(ticker: str) -> pd.DataFrame:
    """Download adjusted close data from yfinance."""
    df = yf.download(
        ticker,
        start=START_DATE,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].copy()
    df.columns = ["close"]
    return df


def create_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Create technical features and forward-return target."""
    df = price_df.copy()

    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)

    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["ma100"] = df["close"].rolling(100).mean()
    df["ma150"] = df["close"].rolling(150).mean()
    df["ma200"] = df["close"].rolling(200).mean()

    df["ma10_ratio"] = df["close"] / df["ma10"] - 1
    df["ma20_ratio"] = df["close"] / df["ma20"] - 1
    df["ma60_ratio"] = df["close"] / df["ma60"] - 1

    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    df["rsi_14"] = calculate_rsi(df["close"], 14)

    df["forward_5d_return"] = df["close"].shift(-5) / df["close"] - 1
    df["target"] = (df["forward_5d_return"] > 0.01).astype(int)

    return df.dropna()


def build_model() -> ExtraTreesClassifier:
    """Build the ExtraTrees model used in the ablation test."""
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )


def build_signal(df: pd.DataFrame, probability: pd.Series, threshold: float, regime_filter: str) -> pd.Series:
    """Build binary trading signal from probability threshold and regime filter."""
    signal = (probability >= threshold).astype(int)

    if regime_filter == "NO_FILTER":
        return signal
    if regime_filter == "MA100":
        return signal.where(df["close"] > df["ma100"], 0)
    if regime_filter == "MA150":
        return signal.where(df["close"] > df["ma150"], 0)
    if regime_filter == "MA200":
        return signal.where(df["close"] > df["ma200"], 0)

    raise ValueError(f"Unknown regime filter: {regime_filter}")


def backtest_signal(df: pd.DataFrame, signal: pd.Series):
    """Backtest signal with one-day execution delay and transaction costs."""
    position = signal.astype(float)
    gross_return = position.shift(1).fillna(0) * df["return_1d"]

    position_change = position.diff().abs()
    position_change.iloc[0] = abs(position.iloc[0])

    cost = position_change * TOTAL_COST_PER_POSITION_CHANGE
    net_return = gross_return - cost

    turnover = position_change.sum()
    cost_drag = cost.sum()
    exposure = position.mean()

    return net_return, turnover, cost_drag, exposure


def calculate_metrics(returns: pd.Series) -> dict:
    """Calculate performance metrics from daily returns."""
    returns = returns.dropna()

    equity = (1 + returns).cumprod()
    total_return = equity.iloc[-1] - 1

    volatility = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else np.nan

    peak = equity.cummax()
    drawdown = equity / peak - 1
    mdd = drawdown.min()

    calmar = total_return / abs(mdd) if mdd < 0 else np.nan

    var_95 = returns.quantile(0.05)
    es_95 = returns[returns <= var_95].mean()

    return {
        "total_return": total_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
        "var_95": var_95,
        "es_95": es_95,
    }


def format_pct(x):
    return f"{x * 100:.2f}%" if pd.notna(x) else "NA"


def format_num(x):
    return f"{x:.2f}" if pd.notna(x) else "NA"


# =========================
# Main Experiment
# =========================

def main():
    asset_rows = []
    benchmark_returns = {}
    strategy_returns = {feature_set_name: {} for feature_set_name in FEATURE_SETS.keys()}

    for asset_name, ticker in ASSETS.items():
        print(f"\nProcessing {asset_name} ({ticker})...")

        price_df = download_price_data(ticker)
        df = create_features(price_df)

        train_df = df.loc[:TRAIN_END].copy()
        valid_df = df.loc[VALID_START:VALID_END].copy()
        test_df = df.loc[TEST_START:].copy()

        y_train = train_df["target"]

        benchmark_returns[asset_name] = test_df["return_1d"]
        benchmark_metrics = calculate_metrics(test_df["return_1d"])

        for feature_set_name, feature_cols in FEATURE_SETS.items():
            print(f"Testing feature set: {feature_set_name}")

            X_train = train_df[feature_cols]
            X_valid = valid_df[feature_cols]
            X_test = test_df[feature_cols]

            model = build_model()
            model.fit(X_train, y_train)

            valid_prob = pd.Series(
                model.predict_proba(X_valid)[:, 1],
                index=valid_df.index,
            )

            test_prob = pd.Series(
                model.predict_proba(X_test)[:, 1],
                index=test_df.index,
            )

            best_score = -np.inf
            best_config = None

            for threshold in THRESHOLDS:
                for regime_filter in REGIME_FILTERS:
                    valid_signal = build_signal(valid_df, valid_prob, threshold, regime_filter)
                    valid_returns, _, _, _ = backtest_signal(valid_df, valid_signal)
                    valid_metrics = calculate_metrics(valid_returns)

                    score = valid_metrics["calmar"]
                    if pd.isna(score):
                        score = -np.inf

                    if score > best_score:
                        best_score = score
                        best_config = {
                            "threshold": threshold,
                            "regime_filter": regime_filter,
                        }

            test_signal = build_signal(
                test_df,
                test_prob,
                best_config["threshold"],
                best_config["regime_filter"],
            )

            test_returns, turnover, cost_drag, exposure = backtest_signal(test_df, test_signal)
            test_metrics = calculate_metrics(test_returns)

            strategy_returns[feature_set_name][asset_name] = test_returns

            asset_rows.append({
                "asset": asset_name,
                "feature_set": feature_set_name,
                "selected_threshold": best_config["threshold"],
                "selected_filter": best_config["regime_filter"],
                "test_return": test_metrics["total_return"],
                "benchmark_return": benchmark_metrics["total_return"],
                "test_sharpe": test_metrics["sharpe"],
                "test_mdd": test_metrics["mdd"],
                "test_calmar": test_metrics["calmar"],
                "market_exposure": exposure,
                "turnover": turnover,
                "cost_drag": cost_drag,
            })

    asset_summary = pd.DataFrame(asset_rows)
    asset_summary.to_csv(RESULT_DIR / "ml_feature_set_ablation_asset_summary.csv", index=False)

    portfolio_rows = []
    equity_curves = {}

    portfolio_sets = {
        "All Assets": list(ASSETS.keys()),
        "Robust Assets": ROBUST_ASSETS,
    }

    for portfolio_name, asset_list in portfolio_sets.items():
        benchmark_df = pd.concat(
            [benchmark_returns[asset] for asset in asset_list],
            axis=1,
        )
        benchmark_df.columns = asset_list
        benchmark_portfolio_returns = benchmark_df.mean(axis=1)

        benchmark_metrics = calculate_metrics(benchmark_portfolio_returns)

        portfolio_rows.append({
            "feature_set": "Buy & Hold",
            "portfolio": portfolio_name,
            "assets": ",".join(asset_list),
            "total_return": benchmark_metrics["total_return"],
            "volatility": benchmark_metrics["volatility"],
            "sharpe": benchmark_metrics["sharpe"],
            "mdd": benchmark_metrics["mdd"],
            "calmar": benchmark_metrics["calmar"],
            "var_95": benchmark_metrics["var_95"],
            "es_95": benchmark_metrics["es_95"],
        })

        equity_curves[f"{portfolio_name} Buy & Hold"] = (1 + benchmark_portfolio_returns).cumprod()

        for feature_set_name, returns_by_asset in strategy_returns.items():
            model_df = pd.concat(
                [returns_by_asset[asset] for asset in asset_list],
                axis=1,
            )
            model_df.columns = asset_list
            model_portfolio_returns = model_df.mean(axis=1)

            model_metrics = calculate_metrics(model_portfolio_returns)

            portfolio_rows.append({
                "feature_set": feature_set_name,
                "portfolio": portfolio_name,
                "assets": ",".join(asset_list),
                "total_return": model_metrics["total_return"],
                "volatility": model_metrics["volatility"],
                "sharpe": model_metrics["sharpe"],
                "mdd": model_metrics["mdd"],
                "calmar": model_metrics["calmar"],
                "var_95": model_metrics["var_95"],
                "es_95": model_metrics["es_95"],
            })

            equity_curves[f"{portfolio_name} {feature_set_name}"] = (1 + model_portfolio_returns).cumprod()

    portfolio_summary = pd.DataFrame(portfolio_rows)
    portfolio_summary.to_csv(RESULT_DIR / "ml_feature_set_ablation_portfolio_summary.csv", index=False)

    plt.figure(figsize=(12, 7))

    for label, curve in equity_curves.items():
        if "Robust Assets" in label:
            plt.plot(curve.index, curve.values, label=label)

    plt.title("Day 28 - Feature Set Ablation: Robust Assets Portfolio")
    plt.xlabel("Date")
    plt.ylabel("Equity Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "feature_set_ablation_robust_assets_equity_curves.png", dpi=150)
    plt.close()

    print("\n==============================")
    print("Asset-Level Feature Set Ablation Test")
    print("==============================")

    printable_asset = asset_summary.copy()

    for col in ["test_return", "benchmark_return", "test_mdd", "market_exposure", "cost_drag"]:
        printable_asset[col] = printable_asset[col].apply(format_pct)

    for col in ["test_sharpe", "test_calmar"]:
        printable_asset[col] = printable_asset[col].apply(format_num)

    print(printable_asset.to_string(index=False))

    print("\n==============================")
    print("Portfolio-Level Feature Set Ablation Test")
    print("==============================")

    printable_portfolio = portfolio_summary.copy()

    for col in ["total_return", "volatility", "mdd", "var_95", "es_95"]:
        printable_portfolio[col] = printable_portfolio[col].apply(format_pct)

    for col in ["sharpe", "calmar"]:
        printable_portfolio[col] = printable_portfolio[col].apply(format_num)

    print(printable_portfolio.to_string(index=False))

    print("\nSaved files:")
    print(f"- {RESULT_DIR / 'ml_feature_set_ablation_asset_summary.csv'}")
    print(f"- {RESULT_DIR / 'ml_feature_set_ablation_portfolio_summary.csv'}")
    print(f"- {CHART_DIR / 'feature_set_ablation_robust_assets_equity_curves.png'}")


if __name__ == "__main__":
    main()
