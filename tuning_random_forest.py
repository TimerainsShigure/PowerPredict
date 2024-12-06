import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# 加载数据
def load_time_series_data(file_path):
    X_train, X_test, y_train, y_test = joblib.load(file_path)
    return X_train, X_test, y_train, y_test


# 定义参数网格
param_grid = {
    'estimator__n_estimators': [50, 100, 150],
    'estimator__max_depth': [5, 10, 15],
    'estimator__max_leaf_nodes': [10, 20, 30],
    'estimator__min_samples_leaf': [1, 2, 5]
}


# 构建并执行网格搜索
def random_search_rf(X_train, y_train):
    model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    random_search = RandomizedSearchCV(
        model, param_grid, n_iter=10, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, random_state=42
    )
    random_search.fit(X_train, y_train)

    print("Best parameters found:", random_search.best_params_)
    return random_search.best_params_


# 主函数
if __name__ == "__main__":

    data_file_path = "data/daily_time_series_data.pkl"#daily/hourly
    X_train, X_test, y_train, y_test = load_time_series_data(data_file_path)

    #For hourly data
    X_train_list = [
        X_train[..., [0, 1, 2, 3]],
        X_train[..., [0, 1, 2, 4]],
        X_train[..., [0, 1, 2, 5]]
    ]
    X_train_combined = np.hstack([X.reshape(X.shape[0], -1) for X in X_train_list])


    y_train = y_train[:, :3]



    # Random search
    best_params = random_search_rf(X_train_combined, y_train)

    # best parameters
    print("Best parameters to use in RandomForest.py:", best_params)