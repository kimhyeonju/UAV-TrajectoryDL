from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Input
from tensorflow.keras.optimizers import Adam

def create_Bi_gru_model3(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(GRU(64)),
        Dense(6)
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model
