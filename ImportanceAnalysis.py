import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


# 加载数据
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# 计算信息增益
def calculate_information_gain(data, environment_features, device_power_features):
    info_gain_results = {}
    for target in device_power_features:
        info_gain = mutual_info_regression(data[environment_features], data[target])
        info_gain_results[target] = pd.DataFrame({
            'Feature': environment_features,
            'Information Gain': info_gain
        }).sort_values(by='Information Gain', ascending=False)
    return info_gain_results

# 计算皮尔逊相关系数
def calculate_pearson_correlation(data, environment_features, device_power_features):
    correlation_results = {}
    for target in device_power_features:
        correlation = data[environment_features].corrwith(data[target])
        correlation_results[target] = pd.DataFrame({
            'Feature': environment_features,
            'Pearson Correlation': correlation.values
        }).sort_values(by='Pearson Correlation', ascending=False, key=abs)
    return correlation_results


def calculate_rfe_importance(data, features, target_column, save_path=None):
    """
    使用递归特征消除 (RFE) 计算特征重要性。
    """

    rfe_results = {}
    for target in device_power_features:
        X = data[environment_features]
        y = data[target]

        # 使用随机森林作为基模型
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        rfe = RFE(estimator=model, n_features_to_select=1, step=1)
        rfe.fit(X, y)

        # 创建 RFE 排名 DataFrame
        rfe_results[target] = pd.DataFrame({
            'Feature': environment_features,
            'RFE Ranking': rfe.ranking_
        }).sort_values(by='RFE Ranking')
    return rfe_results

# 可视化特征重要性
def plot_feature_importance(importance_df, target_name, metric_name, save_path=None):
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'], importance_df[metric_name], color='skyblue')
    plt.xlabel(metric_name)
    plt.ylabel('Features')
    plt.title(f'{metric_name} for {target_name}')
    plt.gca().invert_yaxis()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# 保存结果到文本文件
def save_results_to_txt(results_dict, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)
    for target, df in results_dict.items():
        file_path = f"{output_dir}/{target}_importance.txt"
        df.to_csv(file_path, index=False)
        print(f"Saved importance_results results for {target} to {file_path}")

if __name__ == "__main__":
    # 文件路径
    data_file_path = "D:\Essay\HomeC.csv\HomeC.csv"
    output_dir = "importance_results"

    # 加载数据
    data = load_data(data_file_path)

    # 定义环境特征和目标设备功率特征
    environment_features = [
        'temperature', 'humidity', 'visibility', 'apparentTemperature',
        'pressure', 'windSpeed', 'windBearing', 'precipIntensity',
        'dewPoint', 'precipProbability'
    ]
    device_power_features = ['Wine cellar [kW]','Furnace 1 [kW]','Furnace 2 [kW]']


    data = data.dropna(subset=environment_features + device_power_features)


    info_gain_results = calculate_information_gain(data, environment_features, device_power_features)


    correlation_results = calculate_pearson_correlation(data, environment_features, device_power_features)


    rfe_results = calculate_rfe_importance(data, environment_features, device_power_features)


    save_results_to_txt(info_gain_results, f"{output_dir}/information_gain")
    save_results_to_txt(correlation_results, f"{output_dir}/correlation")
    save_results_to_txt(rfe_results, f"{output_dir}/rfe")


    for target, importance_df in info_gain_results.items():
        plot_feature_importance(importance_df, target_name=target, metric_name='Information Gain',
                                save_path=f"{output_dir}/information_gain_{target}.png")

    for target, importance_df in correlation_results.items():
        plot_feature_importance(importance_df, target_name=target, metric_name='Pearson Correlation',
                                save_path=f"{output_dir}/correlation_{target}.png")

    for target, importance_df in rfe_results.items():
        plot_feature_importance(importance_df, target_name=target, metric_name='RFE Ranking',
                                save_path=f"{output_dir}/rfe_{target}.png")