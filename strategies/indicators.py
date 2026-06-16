# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Calculate technical indicators for SGX and US stocks

import pandas as pd
import matplotlib.pyplot as plt
import os


# =========================
# 1. Load stock data
# =========================

def load_stock_data(file_path):
    """
    Load stock price data from CSV.
    This version handles yfinance CSV files safely.
    """

    df = pd.read_csv(file_path)

    # Rename the first column to Date if Date column does not exist
    if "Date" not in df.columns:
        first_column = df.columns[0]
        df = df.rename(columns={first_column: "Date"})

    # Convert Date column to real date format
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Remove rows where Date is not valid
    df = df.dropna(subset=["Date"])

    # Set Date as index
    df = df.set_index("Date")

    # Convert price columns to numbers
    price_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    for col in price_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows where Close price is missing
    df = df.dropna(subset=["Close"])

    return df


# =========================
# 2. Calculate indicators
# =========================

def add_indicators(df):
    """
    Add MA20, MA60, and RSI indicators.
    """

    # MA20 = 20-day average closing price
    df["MA20"] = df["Close"].rolling(window=20).mean()

    # MA60 = 60-day average closing price
    df["MA60"] = df["Close"].rolling(window=60).mean()

    # Daily price change
    delta = df["Close"].diff()

    # Positive changes
    gain = delta.where(delta > 0, 0)

    # Negative changes
    loss = -delta.where(delta < 0, 0)

    # Average gain and loss over 14 days
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    # RSI formula
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


# =========================
# 3. Save charts
# =========================

def save_price_chart(df, stock_name):
    """
    Save price chart with moving averages.
    """

    os.makedirs("reports/charts", exist_ok=True)

    plt.figure(figsize=(12, 6))

    plt.plot(df.index, df["Close"], label="Close Price")
    plt.plot(df.index, df["MA20"], label="MA20")
    plt.plot(df.index, df["MA60"], label="MA60")

    plt.title(f"{stock_name} Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    file_name = f"reports/charts/{stock_name}_ma_chart.png"
    plt.savefig(file_name)
    plt.show()

    print(f"Chart saved: {file_name}")


def save_rsi_chart(df, stock_name):
    """
    Save RSI chart.
    """

    os.makedirs("reports/charts", exist_ok=True)

    plt.figure(figsize=(12, 4))

    plt.plot(df.index, df["RSI"], label="RSI")
    plt.axhline(70, linestyle="--", label="Overbought 70")
    plt.axhline(30, linestyle="--", label="Oversold 30")

    plt.title(f"{stock_name} RSI Indicator")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(True)

    file_name = f"reports/charts/{stock_name}_rsi_chart.png"
    plt.savefig(file_name)
    plt.show()

    print(f"Chart saved: {file_name}")


# =========================
# 4. Main execution
# =========================

if __name__ == "__main__":

    print("Loading data...")

    dbs = load_stock_data("data/raw/D05_SI.csv")
    nvda = load_stock_data("data/raw/NVDA.csv")

    print("Calculating indicators...")

    dbs = add_indicators(dbs)
    nvda = add_indicators(nvda)

    dbs.to_csv("data/raw/D05_SI_with_indicators.csv")
    nvda.to_csv("data/raw/NVDA_with_indicators.csv")

    print("Indicator CSV files saved!")

    save_price_chart(dbs, "DBS")
    save_rsi_chart(dbs, "DBS")

    save_price_chart(nvda, "NVDA")
    save_rsi_chart(nvda, "NVDA")

    print("Day 4 completed successfully!")