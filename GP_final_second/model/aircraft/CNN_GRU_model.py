from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Input, Flatten
from tensorflow.keras.optimizers import Adam


def create_cnn_gru_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),

        # CNN Layers
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),

        # GRU Layers
        GRU(64, return_sequences=True),
        GRU(64),

        # Output Layer
        Dense(3)
    ])

    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model
