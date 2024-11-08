from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.optimizers import Adam

def create_gru_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True),
        GRU(64),
        Dense(6)
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model
