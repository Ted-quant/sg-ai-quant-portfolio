# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Generate final trading signals for multiple tickers

import os
import pandas as pd


# =========================
# 1. Technical signal logic
# =========================

def create_technical_signal(row):
    """
    Create Buy/Hold signal using moving averages and RSI.
    """

    if row["MA20"] > row["MA60"] and row["RSI"] < 70:
        return "Buy"
    else:
        return "Hold"


# =========================
# 2. Sentiment lookup
# =========================

def get_sentiment_score(ticker, sentiment_df):
    """
    Get average sentiment score for one ticker.
    If ticker is missing, return 0 as neutral sentiment.
    """

    ticker_sentiment = sentiment_df[sentiment_df["ticker"] == ticker]

    if ticker_sentiment.empty:
        return 0

    return ticker_sentiment["average_sentiment"].iloc[0]


# =========================
# 3. Final signal logic
# =========================

def create_final_signal(technical_signal, sentiment_score):
    """
    Combine technical signal and sentiment score.
    """

    if technical_signal == "Buy" and sentiment_score > 0:
        return "Strong Buy"
    elif technical_signal == "Buy":
        return "Buy"
    else:
        return "Hold"


# =========================
# 4. Process one ticker
# =========================

def process_ticker(ticker):
    """
    Create final signal CSV for one ticker.
    """

    input_path = f"data/raw/{ticker}_with_indicators.csv"
    sentiment_path = "data/news/ticker_sentiment_summary.csv"
    output_path = f"data/signals/{ticker}_final_signals.csv"

    if not os.path.exists(input_path):
        print(f"Skipped {ticker}: indicator file not found")
        return

    if not os.path.exists(sentiment_path):
        print("Sentiment summary not found. Using neutral sentiment.")
        sentiment_df = pd.DataFrame(columns=["ticker", "average_sentiment"])
    else:
        sentiment_df = pd.read_csv(sentiment_path)

    df = pd.read_csv(input_path)

    df["signal"] = df.apply(create_technical_signal, axis=1)

    sentiment_score = get_sentiment_score(ticker, sentiment_df)

    df["sentiment_score"] = sentiment_score

    df["final_signal"] = df["signal"].apply(
        lambda x: create_final_signal(x, sentiment_score)
    )

    os.makedirs("data/signals", exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Saved final signals for {ticker}: {output_path}")


# =========================
# 5. Main execution
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

    print("Multi ticker final signal generation completed!")