import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load data
def load_time_series_data(file_path):
    X_train, X_test, y_train, y_test = joblib.load(file_path)
    return X_train, X_test, y_train, y_test

# Build MIMO RF Model
def build_multi_input_rf_model(n_estimators=100, max_depth=None, random_state=42):
    print("Building Random Forest Model...")
    rf_model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    )
    return rf_model

# Train with tracking
def train_multi_input_rf_model_with_tracking(model, X_train_list, y_train, X_test_list, y_test, max_trees=100, step=10):
    print("Training with convergence tracking...")
    X_train_combined = np.hstack([X.reshape(X.shape[0], -1) for X in X_train_list])
    X_test_combined = np.hstack([X.reshape(X.shape[0], -1) for X in X_test_list])

    # Initialize lists to track errors
    train_errors, test_errors, n_trees_list = [], [], []

    # Incrementally train the model with increasing tree numbers
    for n_trees in range(step, max_trees + 1, step):
        model.estimator.set_params(n_estimators=n_trees)
        model.fit(X_train_combined, y_train)

        # Predict and calculate errors
        train_predictions = model.predict(X_train_combined)
        test_predictions = model.predict(X_test_combined)

        train_mse = mean_squared_error(y_train, train_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)

        train_errors.append(train_mse)
        test_errors.append(test_mse)
        n_trees_list.append(n_trees)

        print(f"Trees: {n_trees}, Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")

    return model, n_trees_list, train_errors, test_errors

# Plot error distribution
def plot_error_distribution(y_true, y_pred, output_path="error_distribution.png"):
    residuals = y_true - y_pred
    n_outputs = residuals.shape[1]

    for i in range(n_outputs):
        plt.figure(figsize=(8, 5))
        sns.histplot(residuals[:, i], kde=True, bins=30, color='blue', alpha=0.7)
        plt.title(f"Error Distribution for Output {i+1}")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        output_file = output_path.replace(".png", f"_output_{i+1}.png")
        plt.savefig(output_file, dpi=300)
        plt.show()
        print(f"Error distribution plot saved to {output_file}")

# Plot convergence behavior
def plot_convergence(n_trees_list, train_errors, test_errors, output_path="convergence.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(n_trees_list, train_errors, label="Train Error (MSE)", marker='o')
    plt.plot(n_trees_list, test_errors, label="Test Error (MSE)", marker='s', linestyle='--')
    plt.xlabel("Number of Trees")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Model Convergence: Error Reduction with Increasing Trees")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Convergence plot saved to {output_path}")

# Main
if __name__ == "__main__":
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
            'Wine cellar [kW]': ['temperature', 'apparentTemperature', 'dewPoint', 'pressure', 'windSpeed', 'visibility'],
            'Furnace 1 [kW]': ['temperature', 'apparentTemperature', 'dewPoint', 'pressure', 'windSpeed', 'windBearing'],
            'Furnace 2 [kW]': ['temperature', 'apparentTemperature', 'dewPoint', 'windSpeed', 'humidity', 'windBearing']
        },
        "combined": {
            'Wine cellar [kW]': ['temperature_x_pressure', 'dewPoint_x_apparentTemperature', 'humidity_x_visibility'],
            'Furnace 1 [kW]': ['windSpeed_x_dewPoint', 'temperature_x_pressure', 'humidity_x_visibility'],
            'Furnace 2 [kW]': ['temperature_x_windSpeed', 'dewPoint_x_pressure', 'humidity_x_visibility']
        },
        "mixed": {
            'Wine cellar [kW]': ['temperature', 'dewPoint', 'pressure', 'temperature_x_pressure', 'dewPoint_x_apparentTemperature'],
            'Furnace 1 [kW]': ['temperature', 'dewPoint', 'pressure', 'windSpeed_x_dewPoint', 'temperature_x_pressure'],
            'Furnace 2 [kW]': ['temperature', 'dewPoint', 'windSpeed', 'temperature_x_windSpeed', 'dewPoint_x_pressure']
        }
    }

    # Select configuration
    feature_config = custom_features["mixed"]  # Change to "combined" or "mixed" as needed

    # Feature mapping
    feature_index_mapping = {feature: idx for idx, feature in enumerate(selected_features_hourly + interaction_features)}

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

    # Build and train model
    model = build_multi_input_rf_model(n_estimators=100, max_depth=5)
    model, n_trees_list, train_errors, test_errors = train_multi_input_rf_model_with_tracking(
        model, X_train_list, y_train, X_test_list, y_test, max_trees=100, step=10
    )

    # Save model
    model_save_path = "models/rf_model_single.pkl"
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot convergence
    plot_convergence(n_trees_list, train_errors, test_errors, output_path="pictures/convergence.png")

    # Plot error distribution
    plot_error_distribution(
        y_train,
        model.predict(np.hstack([X.reshape(X.shape[0], -1) for X in X_train_list])),
        output_path="pictures/error/train_error_distribution.png"
    )
    plot_error_distribution(
        y_test,
        model.predict(np.hstack([X.reshape(X.shape[0], -1) for X in X_test_list])),
        output_path="pictures/error/test_error_distribution.png"
    )