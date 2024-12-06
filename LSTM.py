from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import numpy as np



def load_time_series_data(file_path):
    X_train, X_test, y_train, y_test = joblib.load(file_path)
    return X_train, X_test, y_train, y_test



def build_multi_input_lstm_model(input_shapes, output_dim):

    inputs = []
    lstm_layers = []
    for input_shape in input_shapes:
        input_layer = Input(shape=input_shape)
        lstm_out = LSTM(64, return_sequences=True)(input_layer)
        lstm_out = Dropout(0.2)(lstm_out)
        lstm_out = LSTM(32, return_sequences=False)(lstm_out)
        lstm_out = Dropout(0.2)(lstm_out)
        inputs.append(input_layer)
        lstm_layers.append(lstm_out)


    merged = concatenate(lstm_layers)

    output_layers = [Dense(1, name=f'output_{i}')(merged) for i in range(output_dim)]

    model = Model(inputs=inputs, outputs=output_layers)
    model.compile(optimizer='adam', loss='mse')
    return model



def train_multi_input_lstm_model(model, X_train, y_train, X_test, y_test, batch_size=64, epochs=30):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, [y_train[:, i] for i in range(y_train.shape[1])],
        validation_data=(X_test, [y_test[:, i] for i in range(y_test.shape[1])]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )
    return model, history


if __name__ == "__main__":

    data_file_path = "hourly_time_series_data.pkl"
    X_train, X_test, y_train, y_test = load_time_series_data(data_file_path)


    input_shapes = [(X_train.shape[1], X_train.shape[2] // 3), (X_train.shape[1], X_train.shape[2] // 3)]
    output_dim = y_train.shape[1]


    model = build_multi_input_lstm_model(input_shapes=input_shapes, output_dim=output_dim)
    model, history = train_multi_input_lstm_model(model, [X_train[..., :X_train.shape[2] // 3],
                                                          X_train[...,
                                                          X_train.shape[2] // 3:X_train.shape[2] // 3 * 2]],
                                                  y_train,
                                                  [X_test[..., :X_test.shape[2] // 3],
                                                   X_test[..., X_test.shape[2] // 3:X_test.shape[2] // 3 * 2]],
                                                  y_test)
    model.save("models/multi_input_lstm_model.h5")
    print("Save as multi_input_lstm_model.h5")