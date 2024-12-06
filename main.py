import os
from Prepossessing import load_data, add_time_index, feature_selection, preprocess_data
from SampleDistribution import load_preprocessed_data, resample_data, create_time_series, split_data
from LSTM import build_multi_input_lstm_model, train_multi_input_lstm_model
from RandomForest import build_multi_input_rf_model, train_multi_input_rf_model, evaluate_multi_input_rf_model
from Estimate import evaluate_model
import joblib

# 全局设置
data_file_path = ""          #path/to/your/data.csv
preprocessed_data_path = "preprocessed_data.csv"
time_series_data_path = "time_series_data.pkl"
lstm_model_path = "models/multi_input_lstm_model.h5"
rf_model_path = "models/rf_model.pkl"

def main():

    print("Step 1: Prepossesing")
    raw_data = load_data(data_file_path)
    data_with_time = add_time_index(raw_data)
    selected_features = [

    ]
    filtered_data = feature_selection(data_with_time, selected_features)
    preprocessed_data = preprocess_data(filtered_data)
    preprocessed_data.to_csv(preprocessed_data_path)
    print("Save as ", preprocessed_data_path)

    print("Step 2: Time series data and feature")
    data = load_preprocessed_data(preprocessed_data_path)
    resampled_data_hourly = resample_data(data, frequency=)
    X, y = create_time_series(resampled_data_hourly, time_steps=)
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8)
    joblib.dump((X_train, X_test, y_train, y_test), time_series_data_path)
    print("Save as :", time_series_data_path)

    # Step 3: LSTM
    print("Step 3: LSTM model")
    input_shapes = [(X_train.shape[1], X_train.shape[2] // 2), (X_train.shape[1], X_train.shape[2] // 2)]
    output_dim = y_train.shape[1]  # 输出设备数量
    lstm_model = build_multi_input_lstm_model(input_shapes=input_shapes, output_dim=output_dim)
    X_train_list = [X_train[..., :X_train.shape[2] // 2], X_train[..., X_train.shape[2] // 2:]]
    X_test_list = [X_test[..., :X_test.shape[2] // 2], X_test[..., X_test.shape[2] // 2:]]
    lstm_model, lstm_history = train_multi_input_lstm_model(lstm_model, X_train_list, y_train, X_test_list, y_test)
    lstm_model.save(lstm_model_path)
    print("LSTM saved as:", lstm_model_path)

    # Step 4: Random Forest
    print("Step 4: Random Forest")
    rf_model = build_multi_input_rf_model(n_estimators=200, max_depth=15)
    rf_model = train_multi_input_rf_model(rf_model, X_train_list, y_train)
    joblib.dump(rf_model, rf_model_path)
    print("Rf model saved as:", rf_model_path)

    # Step 5: Evaluation
    # Not done
    print("Step 5: Evaluation")
    evaluate_model(rf_model, X_test_list, y_test)  #

if __name__ == "__main__":
    main()