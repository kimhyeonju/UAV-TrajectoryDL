from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam

def create_gru_model5(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True),
        GRU(64),
        Dense(3)
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

# 모델 생성
input_shape = (10, 3)  # 예시 입력 형태 (타임스텝=10, 특성=4)
model = create_gru_model5(input_shape)

# 모델 구조 요약 출력
model.summary()

