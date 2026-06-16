# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Date: 2026-06-16
# Description: Fetch multiple SGX and US stocks automatically

import yfinance as yf
import pandas as pd
import os

# Step 1: Define ticker list (SGX + US large caps)
tickers = {
    "DBS Bank": "D05.SI",
    "Sea Limited": "SE",
    "Singtel": "Z74.SI",
    "OCBC Bank": "O39.SI",
    "NVIDIA": "NVDA",
    "Microsoft": "MSFT",
    "Apple": "AAPL",
    "Alphabet": "GOOGL",
    "SPY ETF": "SPY",
    "QQQ ETF": "QQQ",
}

start_date = "2022-01-01"
end_date = "2024-12-31"

# Step 2: Create folder if not exists
os.makedirs("data/raw", exist_ok=True)

# Step 3: Loop through tickers and download
all_data = {}

for name, ticker in tickers.items():
    print(f"Downloading {name} ({ticker})...")
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        
        # Step 4: Handle missing data
        if df.empty:
            print(f"  Warning: No data for {ticker}")
            continue
        
        df = df.dropna()  # Remove rows with missing values
        
        # Step 5: Save to individual CSV
        filename = f"data/raw/{ticker.replace('.', '_')}.csv"
        df.to_csv(filename)
        
        all_data[name] = df
        print(f"  Saved: {filename} ({len(df)} rows)")
        
    except Exception as e:
        print(f"  Error downloading {ticker}: {e}")

print(f"\nDone! Downloaded {len(all_data)} out of {len(tickers)} tickers.")