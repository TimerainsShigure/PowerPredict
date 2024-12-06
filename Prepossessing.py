import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


#data = pd.read_csv('D:\Essay\HomeC.csv\HomeC.csv')
def load_data(file_path):

    data = pd.read_csv(file_path, low_memory=False)
    return data


def add_time_index(data, start_date='2016-01-01'):
    # New time index
    new_time_index = pd.date_range(start=start_date, periods=len(data), freq='min')
    data['new_time'] = new_time_index
    data.set_index('new_time', inplace=True)
    return data


def generate_interaction_features(data, feature_pairs):
    interaction_data = pd.DataFrame(index=data.index)
    for feature1, feature2 in feature_pairs:
        interaction_name = f"{feature1}_x_{feature2}"
        interaction_data[interaction_name] = data[feature1] * data[feature2]
    return interaction_data

def feature_selection(data, selected_features):

    filtered_data = data[selected_features]
    return filtered_data

def preprocess_data(data):

    data.ffill(inplace=True)

    numeric_data = data.select_dtypes(include=['float64', 'int64'])


    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)  # 保持索引一致


    joblib.dump(scaler, 'data\scaler.pkl')

    return scaled_data

if __name__ == "__main__":
    # Load the original
    file_path = "D:\Essay\HomeC.csv\HomeC.csv"
    data = load_data(file_path)

    # time index
    data = add_time_index(data)

    selected_features_hourly = [
        'temperature', 'apparentTemperature', 'dewPoint', 'windSpeed', 'windBearing', 'pressure',   #Main Environmental Feature
        'humidity', 'visibility', #Environmental Feature not so important
        'Wine cellar [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]'  # Device Feature with high correlation
    ]

    selected_features_daily = [
        'temperature', 'apparentTemperature', 'dewPoint',  # Environmental Feature
        'gen [kW]','Furnace 1 [kW]','Furnace 2 [kW]','Fridge [kW]','Wine cellar [kW]','Kitchen 38 [kW]','Solar [kW]' # Device Feature with high correlation
    ]

    interaction_feature_pairs = [
        ('temperature', 'pressure'),
        ('dewPoint', 'apparentTemperature'),
        ('windSpeed', 'dewPoint'),
        ('temperature', 'windSpeed'),
        ('dewPoint', 'pressure'),
        ('humidity', 'visibility')
    ]

    data_hourly = feature_selection(data, selected_features_hourly)
    data_daily = feature_selection(data, selected_features_daily)

    # interaction Features
    interaction_features = generate_interaction_features(
        data_hourly[['temperature', 'humidity', 'dewPoint', 'pressure', 'windSpeed', 'apparentTemperature', 'visibility']],
        interaction_feature_pairs
    )
    combined_data = pd.concat([data_hourly, interaction_features], axis=1)

    # preprossed data
    preprocessed_data_hourly = preprocess_data(data_hourly)
    preprocessed_data_daily = preprocess_data(data_daily)
    preprocessed_data_combine = preprocess_data(combined_data)

    preprocessed_data_hourly.to_csv("data\preprocessed_data_hourly.csv")
    print("preprocessed_data_hourly.csv done!")
    preprocessed_data_daily.to_csv("data\preprocessed_data_daily.csv")
    print("preprocessed_data_daily.csv done!")
    preprocessed_data_combine.to_csv("data\preprocessed_data_combine.csv")
    print("preprocessed_data_combine.csv done!")

    # Seek
    print("Columns in preprocessed_data_combine:")
    print(preprocessed_data_combine.columns.tolist())

    # Save
    with open("data\preprocessed_data_combine_columns.txt", "w") as f:
        for col in preprocessed_data_combine.columns:
            f.write(f"{col}\n")
    print("Columns saved to 'data\preprocessed_data_combine_columns.txt'")