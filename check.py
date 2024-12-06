import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model


def load_time_series_data(file_path):
    X_train, X_test, y_train, y_test = joblib.load(file_path)
    return X_train, X_test, y_train, y_test


def load_model_from_path(model_path):
    if model_path.endswith(".h5"):
        model = load_model(model_path)
        model_type = "LSTM"
    elif model_path.endswith(".pkl"):
        model = joblib.load(model_path)
        model_type = "RandomForest"
    else:
        raise ValueError("Unsupported model format. Only '.h5' and '.pkl' are supported.")
    return model, model_type


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    return mse, rmse, mae


def evaluate_model_on_train_test(model, model_type, X_train_list, X_test_list, y_train, y_test):
    # Combine input data for RandomForest if needed
    if model_type == "RandomForest":
        X_train_combined = np.hstack([X.reshape(X.shape[0], -1) for X in X_train_list])
        X_test_combined = np.hstack([X.reshape(X.shape[0], -1) for X in X_test_list])
    elif model_type == "LSTM":
        X_train_combined = X_train_list
        X_test_combined = X_test_list

    # trainset
    train_predictions = model.predict(X_train_combined)
    train_mse, train_rmse, train_mae = calculate_metrics(y_train, train_predictions)
    print("训练集误差：")
    for i, (mse, rmse, mae) in enumerate(zip(train_mse, train_rmse, train_mae)):
        print(f"  Output {i+1} - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")

    # testset
    test_predictions = model.predict(X_test_combined)
    test_mse, test_rmse, test_mae = calculate_metrics(y_test, test_predictions)
    print("\n测试集误差：")
    for i, (mse, rmse, mae) in enumerate(zip(test_mse, test_rmse, test_mae)):
        print(f"  Output {i+1} - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")

if __name__ == "__main__":
    data_file_path = "data/hourly_time_series_data.pkl"
    X_train, X_test, y_train, y_test = load_time_series_data(data_file_path)

    y_train = y_train[:, :3]
    y_test = y_test[:, :3]

    X_train_list = [
        X_train[..., [0, 1, 2, 3]],  # Wine cellar [kW]
        X_train[..., [0, 1, 2, 4]],  # Furnace 1 [kW]
        X_train[..., [0, 1, 2, 5]]   # Fridge [kW]
    ]
    X_test_list = [
        X_test[..., [0, 1, 2, 3]],
        X_test[..., [0, 1, 2, 4]],
        X_test[..., [0, 1, 2, 5]]
    ]

    model_path = "models/rf_model_hourly.pkl"  # 或者 "models/multi_input_lstm_model.h5"
    model, model_type = load_model_from_path(model_path)

    evaluate_model_on_train_test(model, model_type, X_train_list, X_test_list, y_train, y_test)