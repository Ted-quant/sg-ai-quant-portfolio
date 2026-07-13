import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier


# =========================
# Final Model Configuration
# =========================

ASSETS = {
    "D05_SI": "D05.SI",
    "NVDA": "NVDA",
    "SPY": "SPY",
}

START_DATE = "2020-01-01"
TRAIN_END = "2023-12-31"
VALID_START = "2024-01-01"
VALID_END = "2024-12-31"
TEST_START = "2025-01-01"

TARGET_HORIZON = 5
TARGET_RETURN_THRESHOLD = 0.01

SIGNAL_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]
REGIME_FILTERS = ["NO_FILTER", "MA100", "MA150", "MA200"]

TOTAL_COST_PER_POSITION_CHANGE = 0.001

FEATURE_COLS = [
    "return_5d",
    "return_20d",
    "ma10_ratio",
    "ma20_ratio",
    "ma60_ratio",
    "volatility_20d",
    "rsi_14",
]

RESULT_DIR = Path("backtest/results")
CHART_DIR = Path("reports/charts")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)


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
    """Download adjusted close data from yfinance with retry logic."""
    last_error = None

    for attempt in range(1, 4):
        try:
            print(f"Downloading {ticker}, attempt {attempt}/3...")

            df = yf.download(
                ticker,
                start=START_DATE,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df[["Close"]].copy()
                df.columns = ["close"]
                return df

            last_error = f"Empty dataframe returned for {ticker}"

        except Exception as e:
            last_error = str(e)

    raise ValueError(f"No data downloaded for {ticker}. Last error: {last_error}")


def create_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Create final model features and target."""
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

    df["forward_return"] = df["close"].shift(-TARGET_HORIZON) / df["close"] - 1
    df["target"] = (df["forward_return"] > TARGET_RETURN_THRESHOLD).astype(int)

    return df.dropna()


def build_model() -> ExtraTreesClassifier:
    """Build final ExtraTrees model."""
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )


def build_signal(df: pd.DataFrame, probability: pd.Series, threshold: float, regime_filter: str) -> pd.Series:
    """Convert model probability into binary trading signal."""
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
    """Backtest binary signal with one-day execution delay and transaction costs."""
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
    """Calculate performance and risk metrics from daily returns."""
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
# Main Final Backtest
# =========================

def main():
    asset_rows = []
    strategy_returns = {}
    benchmark_returns = {}

    for asset_name, ticker in ASSETS.items():
        print(f"\nProcessing final model for {asset_name} ({ticker})...")

        price_df = download_price_data(ticker)
        df = create_features(price_df)

        train_df = df.loc[:TRAIN_END].copy()
        valid_df = df.loc[VALID_START:VALID_END].copy()
        test_df = df.loc[TEST_START:].copy()

        X_train = train_df[FEATURE_COLS]
        y_train = train_df["target"]

        X_valid = valid_df[FEATURE_COLS]
        X_test = test_df[FEATURE_COLS]

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

        for threshold in SIGNAL_THRESHOLDS:
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

        strategy_returns[asset_name] = test_returns
        benchmark_returns[asset_name] = test_df["return_1d"]

        model_metrics = calculate_metrics(test_returns)
        benchmark_metrics = calculate_metrics(test_df["return_1d"])

        asset_rows.append({
            "asset": asset_name,
            "selected_threshold": best_config["threshold"],
            "selected_filter": best_config["regime_filter"],
            "model_return": model_metrics["total_return"],
            "benchmark_return": benchmark_metrics["total_return"],
            "model_volatility": model_metrics["volatility"],
            "benchmark_volatility": benchmark_metrics["volatility"],
            "model_sharpe": model_metrics["sharpe"],
            "benchmark_sharpe": benchmark_metrics["sharpe"],
            "model_mdd": model_metrics["mdd"],
            "benchmark_mdd": benchmark_metrics["mdd"],
            "model_calmar": model_metrics["calmar"],
            "benchmark_calmar": benchmark_metrics["calmar"],
            "model_var_95": model_metrics["var_95"],
            "model_es_95": model_metrics["es_95"],
            "market_exposure": exposure,
            "turnover": turnover,
            "cost_drag": cost_drag,
        })

    asset_summary = pd.DataFrame(asset_rows)
    asset_summary.to_csv(RESULT_DIR / "final_model_asset_summary.csv", index=False)

    strategy_df = pd.concat(strategy_returns.values(), axis=1)
    strategy_df.columns = strategy_returns.keys()
    strategy_portfolio_returns = strategy_df.mean(axis=1)

    benchmark_df = pd.concat(benchmark_returns.values(), axis=1)
    benchmark_df.columns = benchmark_returns.keys()
    benchmark_portfolio_returns = benchmark_df.mean(axis=1)

    strategy_metrics = calculate_metrics(strategy_portfolio_returns)
    benchmark_metrics = calculate_metrics(benchmark_portfolio_returns)

    portfolio_summary = pd.DataFrame([
        {
            "strategy": "Buy & Hold",
            "assets": ",".join(ASSETS.keys()),
            **benchmark_metrics,
        },
        {
            "strategy": "Final ExtraTrees ML",
            "assets": ",".join(ASSETS.keys()),
            **strategy_metrics,
        },
    ])

    portfolio_summary.to_csv(RESULT_DIR / "final_model_portfolio_summary.csv", index=False)

    equity_curves = pd.DataFrame({
        "Buy & Hold": (1 + benchmark_portfolio_returns).cumprod(),
        "Final ExtraTrees ML": (1 + strategy_portfolio_returns).cumprod(),
    })

    equity_curves.to_csv(RESULT_DIR / "final_model_equity_curves.csv")

    plt.figure(figsize=(12, 7))
    plt.plot(equity_curves.index, equity_curves["Buy & Hold"], label="Buy & Hold")
    plt.plot(equity_curves.index, equity_curves["Final ExtraTrees ML"], label="Final ExtraTrees ML")
    plt.title("Final Model Backtest - Robust Assets Portfolio")
    plt.xlabel("Date")
    plt.ylabel("Equity Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "final_model_robust_assets_equity_curve.png", dpi=150)
    plt.close()

    print("\n==============================")
    print("Final Model Asset-Level Summary")
    print("==============================")

    printable_asset = asset_summary.copy()

    pct_cols = [
        "model_return",
        "benchmark_return",
        "model_volatility",
        "benchmark_volatility",
        "model_mdd",
        "benchmark_mdd",
        "model_var_95",
        "model_es_95",
        "market_exposure",
        "cost_drag",
    ]

    for col in pct_cols:
        printable_asset[col] = printable_asset[col].apply(format_pct)

    num_cols = [
        "model_sharpe",
        "benchmark_sharpe",
        "model_calmar",
        "benchmark_calmar",
    ]

    for col in num_cols:
        printable_asset[col] = printable_asset[col].apply(format_num)

    print(printable_asset.to_string(index=False))

    print("\n==============================")
    print("Final Model Portfolio-Level Summary")
    print("==============================")

    printable_portfolio = portfolio_summary.copy()

    for col in ["total_return", "volatility", "mdd", "var_95", "es_95"]:
        printable_portfolio[col] = printable_portfolio[col].apply(format_pct)

    for col in ["sharpe", "calmar"]:
        printable_portfolio[col] = printable_portfolio[col].apply(format_num)

    print(printable_portfolio.to_string(index=False))

    print("\nSaved files:")
    print(f"- {RESULT_DIR / 'final_model_asset_summary.csv'}")
    print(f"- {RESULT_DIR / 'final_model_portfolio_summary.csv'}")
    print(f"- {RESULT_DIR / 'final_model_equity_curves.csv'}")
    print(f"- {CHART_DIR / 'final_model_robust_assets_equity_curve.png'}")


if __name__ == "__main__":
    main()
