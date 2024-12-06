import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt


def validate_model_input_features(model, X_sample):
    if isinstance(model, MultiOutputRegressor):
        base_estimator = model.estimators_[0]
        expected_features = base_estimator.n_features_in_
    else:
        expected_features = model.n_features_in_

    if X_sample.shape[1] != expected_features:
        raise ValueError(
            f"Input features ({X_sample.shape[1]}) do not match model's expected features ({expected_features})."
        )


def prepare_combined_features(X_list):
    X_combined = np.hstack([X.reshape(X.shape[0], -1) for X in X_list])
    return X_combined


def plot_cv_results(cv_results, output_path="pictures/cv_results.png"):
    n_outputs = len(cv_results[0]['mse'])
    n_folds = len(cv_results)

    for output_idx in range(n_outputs):
        mse_values = [fold_result['mse'][output_idx] for fold_result in cv_results]
        mae_values = [fold_result['mae'][output_idx] for fold_result in cv_results]
        rmse_values = [fold_result['rmse'][output_idx] for fold_result in cv_results]

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, n_folds + 1), mse_values, label="MSE", marker='o')
        plt.plot(range(1, n_folds + 1), mae_values, label="MAE", marker='s')
        plt.plot(range(1, n_folds + 1), rmse_values, label="RMSE", marker='^')
        plt.xlabel("Fold Number")
        plt.ylabel("Error Value")
        plt.title(f"Cross-Validation Results for Output {output_idx + 1}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        output_file = output_path.replace(".png", f"_output_{output_idx + 1}.png")
        plt.savefig(output_file, dpi=300)
        plt.show()
        print(f"Saved CV result plot for Output {output_idx + 1} to {output_file}")


def perform_time_series_cv_with_saved_model(X_list, y, model_path, n_splits=5, custom_feature_config=None):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = joblib.load(model_path)

    # Apply custom feature selection if provided
    if custom_feature_config:
        feature_index_mapping = {feature: idx for idx, feature in enumerate(custom_feature_config['all_features'])}
        X_list = [
            X[..., [feature_index_mapping[feature] for feature in custom_feature_config['selected_features'][device]]]
            for device, X in zip(custom_feature_config['selected_features'], X_list)
        ]

    # Combine features for the model
    X_combined = prepare_combined_features(X_list)

    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_combined)):
        print(f"\nProcessing Fold {fold + 1}/{n_splits}...")

        X_train, X_val = X_combined[train_idx], X_combined[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        validate_model_input_features(model, X_val)

        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred, multioutput='raw_values')
        mae = mean_absolute_error(y_val, y_pred, multioutput='raw_values')
        rmse = np.sqrt(mse)

        fold_result = {'mse': mse, 'mae': mae, 'rmse': rmse}
        cv_results.append(fold_result)

        print(f"Fold {fold + 1} Results:")
        for i, (mse_val, mae_val, rmse_val) in enumerate(zip(mse, mae, rmse)):
            print(f"  Output {i + 1}: MSE={mse_val:.6f}, MAE={mae_val:.6f}, RMSE={rmse_val:.6f}")

    # Visualize CV results
    plot_cv_results(cv_results)

    return cv_results


if __name__ == "__main__":
    # Load data and model
    training_data_path = "data/hourly_time_series_data.pkl"
    model_file_path = "models/rf_model_single.pkl"

    X_train_list, _, y_train, _ = joblib.load(training_data_path)

    # Define custom feature configurations for three modes
    custom_feature_configs = {
        'single': {
            'all_features': [
                'temperature', 'apparentTemperature', 'dewPoint', 'windSpeed', 'windBearing', 'pressure',
                'humidity', 'visibility', 'Wine cellar [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]'
            ],
            'selected_features': {
                'Wine cellar [kW]': ['temperature', 'dewPoint', 'pressure', 'windSpeed', 'visibility'],
                'Furnace 1 [kW]': ['temperature', 'dewPoint', 'pressure', 'windSpeed', 'windBearing'],
                'Furnace 2 [kW]': ['temperature', 'dewPoint', 'windSpeed', 'humidity', 'windBearing']
            }
        },
        'combined': {
            'all_features': [
                'temperature_x_pressure', 'dewPoint_x_apparentTemperature', 'windSpeed_x_dewPoint',
                'temperature_x_windSpeed', 'dewPoint_x_pressure', 'humidity_x_visibility'
            ],
            'selected_features': {
                'Wine cellar [kW]': ['humidity_x_visibility', 'temperature_x_pressure', 'dewPoint_x_apparentTemperature'],
                'Furnace 1 [kW]': ['windSpeed_x_dewPoint', 'temperature_x_pressure', 'humidity_x_visibility'],
                'Furnace 2 [kW]': ['temperature_x_windSpeed', 'dewPoint_x_pressure', 'humidity_x_visibility']
            }
        },
        'mixed': {
            'all_features': [
                'temperature', 'apparentTemperature', 'dewPoint', 'windSpeed', 'windBearing', 'pressure',
                'temperature_x_pressure', 'dewPoint_x_apparentTemperature', 'windSpeed_x_dewPoint',
                'temperature_x_windSpeed', 'dewPoint_x_pressure', 'humidity_x_visibility'
            ],
            'selected_features': {
                'Wine cellar [kW]': [
                    'temperature', 'dewPoint', 'pressure', 'temperature_x_pressure', 'dewPoint_x_apparentTemperature'
                ],
                'Furnace 1 [kW]': [
                    'temperature', 'dewPoint', 'pressure', 'windSpeed_x_dewPoint', 'temperature_x_pressure'
                ],
                'Furnace 2 [kW]': [
                    'temperature', 'dewPoint', 'windSpeed', 'temperature_x_windSpeed', 'dewPoint_x_pressure'
                ]
            }
        }
    }

    for mode, config in custom_feature_configs.items():
        print(f"\nEvaluating mode: {mode}")
        perform_time_series_cv_with_saved_model(
            X_train_list, y_train, model_file_path, n_splits=5, custom_feature_config=config
        )