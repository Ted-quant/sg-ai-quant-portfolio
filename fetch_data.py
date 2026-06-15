# Project: sg-ai-quant-portfolio
# Author: Chae Youngjun
# Date: 2026-06-15
# Description: Fetch SGX and US stock data

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download stock data
print("Downloading data...")

dbs = yf.download("D05.SI", start="2022-01-01", end="2024-12-31")
nvda = yf.download("NVDA", start="2022-01-01", end="2024-12-31")

print("DBS Bank (SGX):")
print(dbs.tail())

print("\nNVIDIA (US):")
print(nvda.tail())

# Save to CSV
dbs.to_csv("data/dbs_data.csv")
nvda.to_csv("data/nvda_data.csv")
print("\nData saved to data/ folder!")

# Plot closing prices
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

dbs["Close"].plot(ax=ax1, title="DBS Bank (SGX)", color="red")
nvda["Close"].plot(ax=ax2, title="NVIDIA (US)", color="blue")

plt.tight_layout()
plt.savefig("reports/price_chart.png")
plt.show()
print("Chart saved!")