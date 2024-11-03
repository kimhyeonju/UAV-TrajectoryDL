from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Input, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam

def create_weighted_input_model8(input_shape):
    # 입력 레이어
    inputs = Input(shape=input_shape)

    # 가중치 적용을 위한 각 구간별 가중치 설정
    altitude_weight = Lambda(lambda x: x * 1.5)  # 고도 가중치
    speed_weight = Lambda(lambda x: x * 1.5)  # 속도 가중치

    # 입력 변수 중 고도와 속도에 가중치를 적용
    altitude_scaled = altitude_weight(Lambda(lambda x: x[:, :, 2:3])(inputs))
    speed_scaled = speed_weight(Lambda(lambda x: x[:, :, 3:4])(inputs))
    other_inputs = Lambda(lambda x: x[:, :, :2])(inputs)  # 위도와 경도

    # 가중치 적용 후 병합
    combined_inputs = Concatenate(axis=2)([altitude_scaled, speed_scaled, other_inputs])

    # GRU 레이어
    gru_out = GRU(128, return_sequences=True)(combined_inputs)
    gru_out = GRU(128)(gru_out)

    # 출력 레이어 (위도, 경도, 고도 예측)
    outputs = Dense(3)(gru_out)

    # 모델 생성 및 컴파일
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model
