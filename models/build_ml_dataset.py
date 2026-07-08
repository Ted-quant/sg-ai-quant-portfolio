# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Day 14 build machine learning dataset for quant signal modeling
# Target: 1 if forward 5-day return is greater than 1%, otherwise 0

import os
import pandas as pd
import numpy as np


START_DATE = "2022-01-01"
END_DATE = "2024-12-31"

TICKERS = [
    "NVDA",
    "MSFT",
    "SPY",
    "QQQ",
    "D05_SI"
]

INPUT_FOLDER = "data/raw"
OUTPUT_FOLDER = "models/datasets"


def load_indicator_data(ticker):
    """
    Load local price and indicator data for one ticker.
    """

    input_path = f"{INPUT_FOLDER}/{ticker}_with_indicators.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["MA20"] = pd.to_numeric(df["MA20"], errors="coerce")
    df["MA60"] = pd.to_numeric(df["MA60"], errors="coerce")
    df["RSI"] = pd.to_numeric(df["RSI"], errors="coerce")

    df = df.dropna(subset=["Date", "Close", "MA20", "MA60", "RSI"])
    df = df.sort_values("Date")

    df = df[
        (df["Date"] >= START_DATE) &
        (df["Date"] <= END_DATE)
    ].copy()

    return df


def create_features(df, ticker):
    """
    Create machine learning features from price and technical indicators.
    """

    df = df.copy()

    df["Ticker"] = ticker

    # Daily return
    df["return_1d"] = df["Close"].pct_change()

    # Recent momentum features
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)

    # Moving average relationship features
    df["ma20_ma60_ratio"] = df["MA20"] / df["MA60"] - 1
    df["close_ma20_ratio"] = df["Close"] / df["MA20"] - 1
    df["close_ma60_ratio"] = df["Close"] / df["MA60"] - 1

    # Volatility features
    df["volatility_5d"] = df["return_1d"].rolling(window=5).std()
    df["volatility_20d"] = df["return_1d"].rolling(window=20).std()

    # RSI feature
    df["rsi"] = df["RSI"]

    # Forward 5-day return
    df["forward_return_5d"] = df["Close"].shift(-5) / df["Close"] - 1

    # Target: 1 if forward 5-day return is greater than 1%, otherwise 0
    df["target"] = np.where(df["forward_return_5d"] > 0.01, 1, 0)

    return df


def build_dataset():
    """
    Build combined ML dataset for all tickers.
    """

    all_data = []

    for ticker in TICKERS:
        print(f"Building ML features for {ticker}...")

        df = load_indicator_data(ticker)
        feature_df = create_features(df, ticker)

        all_data.append(feature_df)

    dataset = pd.concat(all_data, ignore_index=True)

    feature_columns = [
        "Date",
        "Ticker",
        "Close",
        "return_1d",
        "return_5d",
        "return_20d",
        "ma20_ma60_ratio",
        "close_ma20_ratio",
        "close_ma60_ratio",
        "volatility_5d",
        "volatility_20d",
        "rsi",
        "forward_return_5d",
        "target"
    ]

    dataset = dataset[feature_columns]
    dataset = dataset.dropna()

    return dataset


if __name__ == "__main__":

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    ml_dataset = build_dataset()

    output_path = f"{OUTPUT_FOLDER}/ml_signal_dataset.csv"
    ml_dataset.to_csv(output_path, index=False)

    print("\nDay 14 ML dataset created.")
    print("==========================")
    print(ml_dataset.head())
    print("\nDataset shape:")
    print(ml_dataset.shape)

    print("\nTarget distribution:")
    print(ml_dataset["target"].value_counts(normalize=True))

    print(f"\nSaved to: {output_path}")
