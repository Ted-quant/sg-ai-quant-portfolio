# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Calculate technical indicators for multiple tickers

import os
import pandas as pd


# =========================
# 1. Indicator functions
# =========================

def calculate_rsi(series, period=14):
    """
    Calculate Relative Strength Index.
    """

    delta = series.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_indicators(df):
    """
    Calculate MA20, MA60, and RSI.
    """

    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA60"] = df["Close"].rolling(window=60).mean()
    df["RSI"] = calculate_rsi(df["Close"], period=14)

    return df


# =========================
# 2. Load and clean data
# =========================

def load_price_data(file_path):
    """
    Load price data and clean Date and Close columns.
    """

    df = pd.read_csv(file_path)

    # If Date column does not exist, rename the first column to Date
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Convert Close column to numeric
    if "Close" in df.columns:
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Remove rows where Date or Close is missing
    df = df.dropna(subset=["Date", "Close"])

    # Sort by date
    df = df.sort_values("Date")

    return df


# =========================
# 3. Process one ticker
# =========================

def process_ticker(ticker):
    """
    Calculate indicators for one ticker and save result.
    """

    input_path = f"data/raw/{ticker}.csv"
    output_path = f"data/raw/{ticker}_with_indicators.csv"

    if not os.path.exists(input_path):
        print(f"Skipped {ticker}: raw price file not found")
        return

    df = load_price_data(input_path)

    if "Close" not in df.columns:
        print(f"Skipped {ticker}: Close column not found")
        return

    df = calculate_indicators(df)

    df.to_csv(output_path, index=False)

    print(f"Saved indicators for {ticker}: {output_path}")


# =========================
# 4. Main execution
# =========================

if __name__ == "__main__":

    tickers = [
        "NVDA",
        "MSFT",
        "SPY",
        "QQQ",
        "D05_SI"
    ]

    for ticker in tickers:
        process_ticker(ticker)

    print("Multi ticker indicator calculation completed!")