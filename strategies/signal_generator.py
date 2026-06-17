# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Generate simple trading signals using technical indicators

import pandas as pd
import os


# =========================
# 1. Load indicator data
# =========================

def load_indicator_data(file_path):
    """
    Load stock data with technical indicators.
    """

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    return df


# =========================
# 2. Generate trading signal
# =========================

def generate_signal(row):
    """
    Generate Buy or Hold signal based on simple rules.

    Rule:
    - Buy if MA20 > MA60 and RSI < 70
    - Otherwise Hold
    """

    if row["MA20"] > row["MA60"] and row["RSI"] < 70:
        return "Buy"
    else:
        return "Hold"


# =========================
# 3. Main execution
# =========================

if __name__ == "__main__":

    print("Loading NVDA indicator data...")

    input_path = "data/raw/NVDA_with_indicators.csv"

    df = load_indicator_data(input_path)

    print("Generating trading signals...")

    df["signal"] = df.apply(generate_signal, axis=1)

    os.makedirs("data/signals", exist_ok=True)

    output_path = "data/signals/NVDA_signals.csv"

    df.to_csv(output_path, index=False)

    print(df[["Date", "Close", "MA20", "MA60", "RSI", "signal"]].tail(20))

    print(f"\nSaved trading signals to: {output_path}")
    print("Day 6 signal generation completed!")