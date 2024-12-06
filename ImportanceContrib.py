import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def load_model(model_path):
    return joblib.load(model_path)


def load_data(file_path):
    return joblib.load(file_path)


def prepare_combined_features(X_list, custom_feature_config):
    """
    根据特征配置处理输入特征。
    """
    feature_index_mapping = {feature: idx for idx, feature in enumerate(custom_feature_config['all_features'])}
    X_list_selected = [
        X[..., [feature_index_mapping[feature] for feature in custom_feature_config['selected_features'][device]]]
        for device, X in zip(custom_feature_config['selected_features'], X_list)
    ]
    return np.hstack([X.reshape(X.shape[0], -1) for X in X_list_selected])


def calculate_feature_importance(model, feature_names, device_name, output_path="pictures/feature_importance.png"):
    """
    计算并绘制特征重要性。
    """
    if hasattr(model.estimator, "feature_importances_"):
        importances = model.estimator.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sorted_importances)), sorted_importances, align="center", alpha=0.7)
        plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha="right", fontsize=10)
        plt.title(f"Feature Importance for {device_name}")
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.tight_layout()
        file_name = output_path.replace(".png", f"_{device_name.replace(' ', '_')}.png")
        plt.savefig(file_name, dpi=300)
        plt.show()
        print(f"Feature importance plot saved to {file_name}")


def perform_feature_importance_analysis(model_path, X_list, feature_config, device_names):
    """
    针对不同输入模式和设备计算特征重要性。
    """
    model = load_model(model_path)
    X_combined = prepare_combined_features(X_list, feature_config)

    for i, device_name in enumerate(device_names):
        device_model = model.estimators_[i] if isinstance(model, joblib.MultiOutputRegressor) else model
        device_features = feature_config['selected_features'][device_name]
        calculate_feature_importance(device_model, device_features, device_name)


if __name__ == "__main__":
    # 数据路径和模型路径
    data_file_path = "data/hourly_time_series_data.pkl"
    model_file_path = "models/rf_model_single.pkl"

    # 加载数据
    X_train_list, _, y_train, _ = load_data(data_file_path)

    # 定义三种输入模式的特征配置
    feature_configs = {
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

    # 设备名称
    device_names = ['Wine cellar [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]']

    # 对每种输入模式进行特征重要性分析
    for mode, config in feature_configs.items():
        print(f"\nPerforming feature importance analysis for mode: {mode}")
        perform_feature_importance_analysis(model_file_path, X_train_list, config, device_names)