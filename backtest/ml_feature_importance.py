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

START_DATE = "2020-01-01"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"

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
    """Create technical features and binary forward-return target."""
    df = price_df.copy()

    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)

    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    df["ma10_ratio"] = df["close"] / df["ma10"] - 1
    df["ma20_ratio"] = df["close"] / df["ma20"] - 1
    df["ma60_ratio"] = df["close"] / df["ma60"] - 1

    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    df["rsi_14"] = calculate_rsi(df["close"], 14)

    df["forward_5d_return"] = df["close"].shift(-5) / df["close"] - 1
    df["target"] = (df["forward_5d_return"] > 0.01).astype(int)

    df = df.dropna()
    return df


def get_extratrees_model() -> ExtraTreesClassifier:
    """Return the ExtraTrees model used for feature importance analysis."""
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )


# =========================
# Main Experiment
# =========================

def main():
    feature_cols = [
        "return_5d",
        "return_20d",
        "ma10_ratio",
        "ma20_ratio",
        "ma60_ratio",
        "volatility_20d",
        "rsi_14",
    ]

    all_rows = []

    for asset_name, ticker in ASSETS.items():
        print(f"Processing {asset_name} ({ticker})...")

        price_df = download_price_data(ticker)
        df = create_features(price_df)

        train_df = df.loc[:TRAIN_END].copy()
        test_df = df.loc[TEST_START:].copy()

        if len(train_df) < 200 or len(test_df) < 50:
            print(f"Skipping {asset_name}: not enough data.")
            continue

        X_train = train_df[feature_cols]
        y_train = train_df["target"]

        model = get_extratrees_model()
        model.fit(X_train, y_train)

        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "asset": asset_name,
            "feature": feature_cols,
            "importance": importances,
        })

        importance_df["importance_rank"] = (
            importance_df["importance"]
            .rank(ascending=False, method="first")
            .astype(int)
        )

        importance_df = importance_df.sort_values("importance", ascending=False)

        all_rows.append(importance_df)

        # Asset-level chart
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["feature"], importance_df["importance"])
        plt.gca().invert_yaxis()
        plt.title(f"ExtraTrees Feature Importance - {asset_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(CHART_DIR / f"feature_importance_{asset_name}.png", dpi=150)
        plt.close()

    final_df = pd.concat(all_rows, ignore_index=True)
    final_df.to_csv(RESULT_DIR / "extratrees_feature_importance_by_asset.csv", index=False)

    avg_df = (
        final_df
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )

    avg_df["importance_rank"] = (
        avg_df["importance"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    avg_df.to_csv(RESULT_DIR / "extratrees_feature_importance_average.csv", index=False)

    # Average importance chart
    plt.figure(figsize=(10, 6))
    plt.barh(avg_df["feature"], avg_df["importance"])
    plt.gca().invert_yaxis()
    plt.title("Average ExtraTrees Feature Importance Across Assets")
    plt.xlabel("Average Importance")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "feature_importance_average_extratrees.png", dpi=150)
    plt.close()

    print("\n==============================")
    print("Average ExtraTrees Feature Importance")
    print("==============================")
    print(avg_df.to_string(index=False))

    print("\n==============================")
    print("Asset-Level ExtraTrees Feature Importance")
    print("==============================")
    print(final_df.to_string(index=False))

    print("\nSaved files:")
    print(f"- {RESULT_DIR / 'extratrees_feature_importance_by_asset.csv'}")
    print(f"- {RESULT_DIR / 'extratrees_feature_importance_average.csv'}")
    print(f"- {CHART_DIR / 'feature_importance_average_extratrees.png'}")
    print(f"- {CHART_DIR / 'feature_importance_D05_SI.png'}")
    print(f"- {CHART_DIR / 'feature_importance_MSFT.png'}")
    print(f"- {CHART_DIR / 'feature_importance_NVDA.png'}")
    print(f"- {CHART_DIR / 'feature_importance_QQQ.png'}")
    print(f"- {CHART_DIR / 'feature_importance_SPY.png'}")


if __name__ == "__main__":
    main()
