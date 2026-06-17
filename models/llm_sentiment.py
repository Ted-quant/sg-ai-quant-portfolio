# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Rule-based news sentiment scoring and ticker-level summary

import pandas as pd
import os


# =========================
# 1. Sample news headlines
# =========================

news_data = [
    {
        "ticker": "NVDA",
        "headline": "NVIDIA shares rise after strong AI chip demand"
    },
    {
        "ticker": "NVDA",
        "headline": "NVIDIA faces pressure from new export restrictions"
    },
    {
        "ticker": "MSFT",
        "headline": "Microsoft reports strong cloud revenue growth"
    },
    {
        "ticker": "MSFT",
        "headline": "Microsoft stock falls after weak guidance"
    },
    {
        "ticker": "DBS",
        "headline": "DBS Bank benefits from resilient Singapore economy"
    },
    {
        "ticker": "DBS",
        "headline": "DBS shares decline as interest rate uncertainty grows"
    },
    {
        "ticker": "SPY",
        "headline": "US stocks rally as inflation cools"
    },
    {
        "ticker": "SPY",
        "headline": "US market drops amid recession fears"
    },
    {
        "ticker": "QQQ",
        "headline": "Tech stocks gain as investors return to growth shares"
    },
    {
        "ticker": "QQQ",
        "headline": "Nasdaq weakens as valuation concerns rise"
    }
]


# =========================
# 2. Simple sentiment function
# =========================

def calculate_sentiment_score(headline):
    """
    Calculate simple sentiment score from a news headline.
    Positive words add points.
    Negative words subtract points.
    """

    positive_words = [
        "rise", "strong", "growth", "benefits", "rally",
        "cools", "gain", "gains", "resilient", "beat",
        "upgrade", "surge", "optimism"
    ]

    negative_words = [
        "falls", "fall", "pressure", "restrictions", "weak",
        "decline", "declines", "uncertainty", "drops",
        "recession", "fears", "weakens", "concerns",
        "downgrade", "miss", "risk", "selloff"
    ]

    headline_lower = headline.lower()

    score = 0

    for word in positive_words:
        if word in headline_lower:
            score += 1

    for word in negative_words:
        if word in headline_lower:
            score -= 1

    return score


# =========================
# 3. Convert score to label
# =========================

def convert_score_to_label(score):
    """
    Convert numeric sentiment score into text label.
    """

    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"


# =========================
# 4. Main execution
# =========================

if __name__ == "__main__":

    print("Creating sample news sentiment data...")

    # Convert list data into table format
    df = pd.DataFrame(news_data)

    # Calculate sentiment score for each headline
    df["sentiment_score"] = df["headline"].apply(calculate_sentiment_score)

    # Convert score into Positive / Negative / Neutral label
    df["sentiment_label"] = df["sentiment_score"].apply(convert_score_to_label)

    # Create output folder
    os.makedirs("data/news", exist_ok=True)

    # Save detailed news sentiment result
    detail_output_path = "data/news/sample_news_sentiment.csv"
    df.to_csv(detail_output_path, index=False)

    print("\nDetailed sentiment result:")
    print(df)

    print(f"\nSaved detailed sentiment data to: {detail_output_path}")

    # =========================
    # 5. Ticker-level sentiment summary
    # =========================

    summary_df = df.groupby("ticker").agg(
        news_count=("headline", "count"),
        average_sentiment=("sentiment_score", "mean"),
        total_sentiment=("sentiment_score", "sum")
    ).reset_index()

    summary_df["overall_label"] = summary_df["average_sentiment"].apply(convert_score_to_label)

    summary_output_path = "data/news/ticker_sentiment_summary.csv"
    summary_df.to_csv(summary_output_path, index=False)

    print("\nTicker-level sentiment summary:")
    print(summary_df)

    print(f"\nSaved ticker sentiment summary to: {summary_output_path}")
    print("Day 5 extended sentiment analysis completed!")