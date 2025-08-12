import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# --------------------------
# Settings
# --------------------------
tickers = ["TSLA", "BND", "SPY"]
start_date = "2015-07-01"
end_date = "2025-07-31"
csv_file = "market_data.csv"

# --------------------------
# 1. Load or fetch data
# --------------------------
if os.path.exists(csv_file):
    print(f"Loading data from {csv_file}...")
    df_all = pd.read_csv(csv_file, parse_dates=["Date"])
else:
    print("Fetching data from Yahoo Finance...")
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        df["Ticker"] = ticker
        data[ticker] = df
    
    # Merge into single DataFrame
    df_all = pd.concat(data.values(), ignore_index=True)
    
    # Save to CSV
    df_all.to_csv(csv_file, index=False)
    print(f"Data saved to {csv_file}")

# --------------------------
# 2. Data quality checks
# --------------------------
print(df_all.info())
print(df_all.isnull().sum())

# --------------------------
# 3. Handle missing values
# --------------------------
df_all = df_all.groupby("Ticker").apply(lambda x: x.ffill().bfill()).reset_index(drop=True)

# --------------------------
# 4. Feature Engineering
# --------------------------
df_all["Daily Return"] = df_all.groupby("Ticker")["Adj Close"].pct_change()

# Rolling 20-day volatility
df_all["Rolling Volatility"] = (
    df_all.groupby("Ticker")["Daily Return"]
    .rolling(window=20)
    .std()
    .reset_index(level=0, drop=True)
)

# Value at Risk (95%)
VaR_95 = df_all.groupby("Ticker")["Daily Return"].quantile(0.05)
print("\nValue at Risk (95%):\n", VaR_95)

# Sharpe Ratio (annualized)
sharpe_ratios = (
    df_all.groupby("Ticker")["Daily Return"].mean()
    / df_all.groupby("Ticker")["Daily Return"].std()
    * np.sqrt(252)
)
print("\nAnnualized Sharpe Ratios:\n", sharpe_ratios)

# --------------------------
# 5. EDA Visualizations
# --------------------------
plt.figure(figsize=(12, 6))
for ticker in tickers:
    subset = df_all[df_all["Ticker"] == ticker]
    plt.plot(subset["Date"], subset["Adj Close"], label=ticker)
plt.title("Adjusted Closing Prices")
plt.legend()
plt.show()

# --------------------------
# 6. Stationarity Check (ADF Test on Returns)
# --------------------------
for ticker in tickers:
    returns = df_all[df_all["Ticker"] == ticker]["Daily Return"].dropna()
    adf_result = adfuller(returns)
    print(f"\nADF Test for {ticker} Returns:")
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    if adf_result[1] < 0.05:
        print("Series is stationary")
    else:
        print("Series is NOT stationary")

