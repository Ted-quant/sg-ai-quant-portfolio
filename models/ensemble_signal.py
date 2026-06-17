# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Combine technical trading signals with news sentiment signals

import pandas as pd
import os


# =========================
# 1. Load data
# =========================

def load_data():
    """
    Load technical signals and sentiment summary.
    """

    technical_df = pd.read_csv("data/signals/NVDA_signals.csv")
    sentiment_df = pd.read_csv("data/news/ticker_sentiment_summary.csv")

    return technical_df, sentiment_df


# =========================
# 2. Get sentiment score
# =========================

def get_ticker_sentiment(sentiment_df, ticker):
    """
    Get average sentiment score for a ticker.
    """

    ticker_row = sentiment_df[sentiment_df["ticker"] == ticker]

    if ticker_row.empty:
        return 0

    return ticker_row["average_sentiment"].iloc[0]


# =========================
# 3. Create final signal
# =========================

def create_final_signal(row, sentiment_score):
    """
    Combine technical signal and sentiment score.

    Rule:
    - Strong Buy: technical signal is Buy and sentiment is positive
    - Buy: technical signal is Buy
    - Hold: otherwise
    """

    if row["signal"] == "Buy" and sentiment_score > 0:
        return "Strong Buy"
    elif row["signal"] == "Buy":
        return "Buy"
    else:
        return "Hold"


# =========================
# 4. Main execution
# =========================

if __name__ == "__main__":

    print("Loading technical and sentiment data...")

    technical_df, sentiment_df = load_data()

    print("Getting NVDA sentiment score...")

    nvda_sentiment = get_ticker_sentiment(sentiment_df, "NVDA")

    print(f"NVDA average sentiment score: {nvda_sentiment}")

    print("Creating final combined signal...")

    technical_df["sentiment_score"] = nvda_sentiment

    technical_df["final_signal"] = technical_df.apply(
        lambda row: create_final_signal(row, nvda_sentiment),
        axis=1
    )

    os.makedirs("data/signals", exist_ok=True)

    output_path = "data/signals/NVDA_final_signals.csv"

    technical_df.to_csv(output_path, index=False)

    print(technical_df[["Date", "Close", "signal", "sentiment_score", "final_signal"]].tail(20))

    print(f"\nSaved final signals to: {output_path}")
    print("Day 6 extended ensemble signal completed!")