import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def load_time_series_data(file_path):
    X_train, X_test, y_train, y_test = joblib.load(file_path)
    return X_train, X_test, y_train, y_test

# Load Model
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

#
# def model_prediction(model, X_test_list, model_type):
#     if model_type == "LSTM":
#         predictions = model.predict(X_test_list)
#     elif model_type == "RandomForest":
#         X_test_combined = np.hstack([X.reshape(X.shape[0], -1) for X in X_test_list])
#         predictions = model.predict(X_test_combined)
#     return predictions
def model_prediction(model, X_test, model_type):
    if model_type == "LSTM":
        predictions = model.predict(X_test_list)
    elif model_type == "RandomForest":
        X_test_combined = np.hstack([X.reshape(X.shape[0], -1) for X in X_test_list])
        print("Combined X_test shape for prediction:", X_test_combined.shape)  # 调试信息
        if X_test_combined.shape[1] != 360: #In hourly sampled data, it should be 432,196 in daily,576 in combined
            raise ValueError(f"Expected features got {X_test_combined.shape[1]}.")
        predictions = model.predict(X_test_combined)
    return predictions

# metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    return mse, rmse, mae


def calculate_accuracy(y_true, y_pred, threshold=0.05):

    accuracies = []
    for i in range(y_true.shape[1]):
        error_percentage = np.abs((y_pred[:, i] - y_true[:, i]) / y_true[:, i])
        accuracy = np.mean(error_percentage < threshold) * 100  # 转化为百分比
        accuracies.append(accuracy)
    return accuracies


def plot_predictions(y_true, y_pred, title="Model Predictions vs True Values"):
    for i in range(y_true.shape[1]):
        plt.figure(figsize=(10, 5))
        plt.plot(y_true[:, i], label=f'True Output {i+1}', color='blue')
        plt.plot(y_pred[:, i], label=f'Predicted Output {i+1}', color='orange')
        plt.title(f"{title} - Output {i+1}")
        plt.xlabel("Sample Index")
        plt.ylabel("Output Value")
        plt.legend()
        #plt.show()

        file_path = f"pictures/{title.replace(' ', '_')}_Output_{i+1}.png"
        plt.savefig(file_path)
        print(f"Saved plot for Output {i+1} to {file_path}")
        plt.close()

def evaluate_model(model, model_type, X_test, y_test):
    # prediction and Estimate
    predictions = model_prediction(model, X_test, model_type=model_type)

    print("y_test shape:", y_test.shape)
    print("Predictions shape:", predictions.shape)

    mse, rmse, mae = calculate_metrics(y_test, predictions)
    print(f"{model_type} Estimate Metrics：")
    for i, (mse_val, rmse_val, mae_val) in enumerate(zip(mse, rmse, mae)):
        print(f"  output {i+1} - MSE: {mse_val}, RMSE: {rmse_val}, MAE: {mae_val}")

    # accuracy
    accuracies = calculate_accuracy(y_test, predictions)
    for i, accuracy in enumerate(accuracies):
        print(f"  输出 {i+1} - 预测正确率: {accuracy:.2f}%")

    # Visualization
    plot_predictions(y_test, predictions, title=f"{model_type} Model Predictions vs True Values for mixed")



if __name__ == "__main__":
    #data_file_path = "data/hourly_time_series_data.pkl"  # 或者 "daily_time_series_data.pkl"
    #data_file_path = "data/daily_time_series_data.pkl"
    data_file_path = "data/combined_hourly_series_data.pkl"
    X_train, X_test, y_train, y_test = load_time_series_data(data_file_path)

    # Feature configurations
    selected_features_hourly = [
        'temperature', 'apparentTemperature', 'dewPoint', 'windSpeed', 'windBearing', 'pressure',
        'humidity', 'visibility', 'Wine cellar [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]'
    ]
    interaction_features = [
        'temperature_x_pressure', 'dewPoint_x_apparentTemperature', 'windSpeed_x_dewPoint',
        'temperature_x_windSpeed', 'dewPoint_x_pressure', 'humidity_x_visibility'
    ]
    custom_features = {
        "single": {
            'Wine cellar [kW]': ['temperature', 'apparentTemperature', 'dewPoint', 'pressure', 'windSpeed',
                                 'visibility'],
            'Furnace 1 [kW]': ['temperature', 'apparentTemperature', 'dewPoint', 'pressure', 'windSpeed',
                               'windBearing'],
            'Furnace 2 [kW]': ['temperature', 'apparentTemperature', 'dewPoint', 'windSpeed', 'humidity', 'windBearing']
        },
        "combined": {
            'Wine cellar [kW]': ['temperature_x_pressure', 'dewPoint_x_apparentTemperature', 'humidity_x_visibility'],
            'Furnace 1 [kW]': ['windSpeed_x_dewPoint', 'temperature_x_pressure', 'humidity_x_visibility'],
            'Furnace 2 [kW]': ['temperature_x_windSpeed', 'dewPoint_x_pressure', 'humidity_x_visibility']
        },
        "mixed": {
            'Wine cellar [kW]': ['temperature', 'dewPoint', 'pressure', 'temperature_x_pressure',
                                 'dewPoint_x_apparentTemperature'],
            'Furnace 1 [kW]': ['temperature', 'dewPoint', 'pressure', 'windSpeed_x_dewPoint', 'temperature_x_pressure'],
            'Furnace 2 [kW]': ['temperature', 'dewPoint', 'windSpeed', 'temperature_x_windSpeed', 'dewPoint_x_pressure']
        }
    }

    # Select configuration
    feature_config = custom_features["mixed"]  # Change to "combined" or "mixed" as needed

    # Feature mapping
    feature_index_mapping = {feature: idx for idx, feature in
                             enumerate(selected_features_hourly + interaction_features)}

    X_train_list = [
        X_train[..., [feature_index_mapping[feature] for feature in feature_config[device_feature]]]
        for device_feature in feature_config
    ]
    X_test_list = [
        X_test[..., [feature_index_mapping[feature] for feature in feature_config[device_feature]]]
        for device_feature in feature_config
    ]

    # Output configuration
    y_train = y_train[:, :3]
    y_test = y_test[:, :3]



    # Debug shapes
    print("y_test shape:", y_test.shape)

    #lstm_model_path = "models/multi_input_lstm_model.h5"
    rf_model_path = "models/rf_model_single.pkl"

    # LSTM
    # lstm_model, lstm_type = load_model_from_path(lstm_model_path)
    # evaluate_model(lstm_model, lstm_type, X_test, y_test)

    # Random Forest
    rf_model, rf_type = load_model_from_path(rf_model_path)
    evaluate_model(rf_model, rf_type, X_test_list, y_test)
