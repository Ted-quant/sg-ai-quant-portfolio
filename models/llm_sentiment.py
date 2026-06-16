# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Description: Simple rule-based news sentiment scoring for quant portfolio

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
        "cools", "gain", "gains", "resilient"
    ]

    negative_words = [
        "falls", "fall", "pressure", "restrictions", "weak",
        "decline", "declines", "uncertainty", "drops",
        "recession", "fears", "weakens", "concerns"
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
# 3. Main execution
# =========================

if __name__ == "__main__":

    print("Creating sample news sentiment data...")

    df = pd.DataFrame(news_data)

    df["sentiment_score"] = df["headline"].apply(calculate_sentiment_score)

    os.makedirs("data/news", exist_ok=True)

    output_path = "data/news/sample_news_sentiment.csv"

    df.to_csv(output_path, index=False)

    print(df)

    print(f"\nSaved sentiment data to: {output_path}")
    print("Day 5 basic sentiment analysis completed!")