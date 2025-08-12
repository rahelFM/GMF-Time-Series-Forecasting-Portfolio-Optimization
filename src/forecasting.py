import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Settings
DATA_FILE = "market_data.csv"
FORECAST_FILE = "forecasts.csv"

# Train/Test split dates
TRAIN_END_DATE = "2023-12-31"
TEST_START_DATE = "2024-01-01"
TEST_END_DATE = "2025-07-31"

# Forecast horizon (days)
FORECAST_DAYS = 365 + 212  # approx. 1.7 years (Jan 2024 to Jul 2025)

# Devices for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTM Dataset for time series
class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len=20):
        self.series = series
        self.seq_len = seq_len

    def __len__(self):
        return len(self.series) - self.seq_len

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.seq_len]
        y = self.series[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last time step
        out = self.linear(out)
        return out

def train_lstm(train_series, seq_len=20, epochs=30, batch_size=16):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.reshape(-1, 1)).flatten()

    dataset = TimeSeriesDataset(train_scaled, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.unsqueeze(-1).to(device)
            y_batch = y_batch.unsqueeze(-1).to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"LSTM Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.6f}")

    return model, scaler

def forecast_lstm(model, scaler, recent_data, forecast_len=FORECAST_DAYS, seq_len=20):
    model.eval()
    preds = []
    data = recent_data[-seq_len:].copy()
    data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()

    input_seq = list(data_scaled)
    for _ in range(forecast_len):
        seq_input = torch.tensor(input_seq[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            pred_scaled = model(seq_input).cpu().numpy().flatten()[0]
        input_seq.append(pred_scaled)
        preds.append(pred_scaled)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds

def evaluate_metrics(true_vals, preds):
    rmse = np.sqrt(mean_squared_error(true_vals, preds))
    mae = mean_absolute_error(true_vals, preds)
    mape = np.mean(np.abs((true_vals - preds) / true_vals)) * 100
    return rmse, mae, mape

def run_arima_forecast(series, train_end_date, forecast_days):
    train_series = series[:train_end_date]
    model = ARIMA(train_series, order=(5,1,0))  # p,d,q fixed for simplicity; you can optimize these later
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)
    return forecast

def run_prophet_forecast(df, train_end_date, forecast_days):
    df_prophet = df.reset_index()[['Date', 'Adj Close']].rename(columns={'Date':'ds', 'Adj Close':'y'})
    train_df = df_prophet[df_prophet['ds'] <= train_end_date]

    model = Prophet(daily_seasonality=True)
    model.fit(train_df)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    forecast_series = forecast.set_index('ds')['yhat'][-forecast_days:]
    return forecast_series

def main():
    # Load data
    df = pd.read_csv(DATA_FILE, parse_dates=['Date'])
    df.set_index('Date', inplace=True)

    results = []

    for ticker in ["TSLA", "BND", "SPY"]:
        print(f"\nProcessing {ticker}...")

        series = df[df['Ticker'] == ticker]['Adj Close']
        train_series = series[:TRAIN_END_DATE]
        test_series = series[TEST_START_DATE:TEST_END_DATE]

        # --- ARIMA forecast ---
        print("Training ARIMA...")
        arima_forecast = run_arima_forecast(series, TRAIN_END_DATE, len(test_series))

        # --- Prophet forecast ---
        print("Training Prophet...")
        df_ticker = df[df['Ticker'] == ticker][['Adj Close']]
        prophet_forecast = run_prophet_forecast(df_ticker, TRAIN_END_DATE, len(test_series))

        # --- LSTM forecast ---
        print("Training LSTM...")
        train_values = train_series.values
        model, scaler = train_lstm(train_values)
        lstm_forecast = forecast_lstm(model, scaler, train_values, forecast_len=len(test_series))

        # Evaluate models
        print("Evaluating models...")
        arima_rmse, arima_mae, arima_mape = evaluate_metrics(test_series.values, arima_forecast.values)
        prophet_rmse, prophet_mae, prophet_mape = evaluate_metrics(test_series.values, prophet_forecast.values)
        lstm_rmse, lstm_mae, lstm_mape = evaluate_metrics(test_series.values, lstm_forecast)

        print(f"ARIMA RMSE: {arima_rmse:.3f}, MAE: {arima_mae:.3f}, MAPE: {arima_mape:.2f}%")
        print(f"Prophet RMSE: {prophet_rmse:.3f}, MAE: {prophet_mae:.3f}, MAPE: {prophet_mape:.2f}%")
        print(f"LSTM RMSE: {lstm_rmse:.3f}, MAE: {lstm_mae:.3f}, MAPE: {lstm_mape:.2f}%")

        # Store results for this ticker
        for i, date in enumerate(test_series.index):
            results.append({
                "Date": date,
                "Ticker": ticker,
                "True": test_series.iloc[i],
                "ARIMA": arima_forecast.iloc[i],
                "Prophet": prophet_forecast.iloc[i],
                "LSTM": lstm_forecast[i],
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv(FORECAST_FILE, index=False)
    print(f"\nForecasts saved to {FORECAST_FILE}")

if __name__ == "__main__":
    main()
