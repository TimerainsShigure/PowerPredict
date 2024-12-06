import pandas as pd
import numpy as np
import joblib


# Time Series
def load_preprocessed_data(file_path):
    data = pd.read_csv(file_path, index_col=0)  # index_col=0 用于保留时间索引
    data.index = pd.to_datetime(data.index)
    return data


def resample_data(data, frequency):

    resampled_data = data.resample(frequency).mean()
    return resampled_data

def create_time_series(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data.iloc[i:i + time_steps].values)
        y.append(data.iloc[i + time_steps].values)
    X, y = np.array(X), np.array(y)
    return X, y


def split_data(X, y, train_ratio=0.8):
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data_file_path_1 = "data/preprocessed_data_hourly.csv"  # File Path of preprocessed data
    data_1 = load_preprocessed_data(data_file_path_1)

    # Resample and produce hourly data
    print("Hourly resampling...")
    resampled_data_hourly = resample_data(data_1, frequency='h')
    X_hourly, y_hourly = create_time_series(resampled_data_hourly, time_steps=24)
    X_train_hourly, X_test_hourly, y_train_hourly, y_test_hourly = split_data(X_hourly, y_hourly)
    joblib.dump((X_train_hourly, X_test_hourly, y_train_hourly, y_test_hourly), "data/hourly_time_series_data.pkl")
    print("Saved as 'hourly_time_series_data.pkl'")

    data_file_path_2 = "data/preprocessed_data_daily.csv"  # File Path of preprocessed data
    data_2 = load_preprocessed_data(data_file_path_2)

    # Resample and produce daily data
    print("Daily resampling...")
    resampled_data_daily = resample_data(data_2, frequency='D')
    X_daily, y_daily = create_time_series(resampled_data_daily, time_steps=7)
    X_train_daily, X_test_daily, y_train_daily, y_test_daily = split_data(X_daily, y_daily)
    joblib.dump((X_train_daily, X_test_daily, y_train_daily, y_test_daily), "data/daily_time_series_data.pkl")
    print("Saved as 'daily_time_series_data.pkl'")

    data_file_path_3 = "data/preprocessed_data_combine.csv"  # File Path of preprocessed data
    data_3 = load_preprocessed_data(data_file_path_3)

    # Resample and produce daily data
    print("Combined resampling...")
    resampled_data_combined = resample_data(data_3, frequency='h')
    X_com, y_com = create_time_series(resampled_data_combined, time_steps=24)
    X_train_com, X_test_com, y_train_com, y_test_com = split_data(X_com, y_com)
    joblib.dump((X_train_com, X_test_com, y_train_com, y_test_com), "data/combined_hourly_series_data.pkl")
    print("Saved as 'combined_hourly_series_data.pkl'")

