# GMF-Time-Series-Forecasting-Portfolio-Optimization

## Overview
GMF Investments is a forward-thinking financial advisory firm specializing in personalized portfolio management. This project focuses on applying advanced time series forecasting models to historical financial data to optimize portfolio management strategies. The goal is to predict market trends, optimize asset allocation, and enhance portfolio performance by integrating classical statistical models and deep learning techniques.

## Project Description
This repository contains the code and analysis for forecasting Tesla (TSLA) stock prices using ARIMA and LSTM models, combined with portfolio optimization based on Modern Portfolio Theory (MPT). Additionally, the project includes backtesting of portfolio strategies to validate forecasting-based decisions against benchmark portfolios.


## Data
- Historical financial data is sourced via the [Yahoo Finance API](https://pypi.org/project/yfinance/) using the Python `yfinance` library.
- Assets analyzed:
  - **TSLA**: Tesla stock (high growth, high risk)
  - **BND**: Vanguard Total Bond Market ETF (stability, low risk)
  - **SPY**: S&P 500 ETF (diversified market exposure)
- Data period: July 1, 2015 - July 31, 2025

---

## Features
- Data preprocessing and exploratory data analysis (EDA)
- Statistical stationarity testing (Augmented Dickey-Fuller)
- Time series forecasting using:
  - ARIMA model
  - LSTM (Long Short-Term Memory) neural network
- Portfolio optimization with PyPortfolioOpt
- Efficient Frontier visualization and key portfolio identification
- Backtesting strategy performance against benchmark portfolios

---

## Usage

### Requirements
- Python 3.8+
- Key libraries:
  - `yfinance`
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `statsmodels`, `pmdarima`
  - `tensorflow` / `keras`
  - `prophet` (optional alternative forecasting model)
  - `PyPortfolioOpt`


pip install -r requirements.txt
Running the Project
Fetch and preprocess data:
python preprocessing.py
Train forecasting models and evaluate:
python forecasting.py
Generate portfolio optimization and efficient frontier:

python optimization.py
Run backtesting simulation:

python backtesting.py
Project Structure

├── data/                       # Raw and processed data files
├── src/                        # Source code for preprocessing, modeling, optimization
│   ├── preprocessing.py
│   ├── forecasting.py
           
Author
Rahel Sileshi Abdisa
GMF Investments – Financial Analyst

References
YFinance Python API

ARIMA tutorial

LSTM time series forecasting

PyPortfolioOpt documentation

License
This project is licensed under the MIT License.
