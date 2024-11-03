from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam

def create_gru_model7(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(128, return_sequences=True),
        GRU(128),
        Dense(3)
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model
