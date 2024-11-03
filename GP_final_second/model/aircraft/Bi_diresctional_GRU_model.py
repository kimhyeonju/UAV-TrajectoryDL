from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Input
from tensorflow.keras.optimizers import Adam

def create_Bi_gru_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(GRU(64, return_sequences=True)),
        GRU(64),
        Dense(3)
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model
