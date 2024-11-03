import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Input, Multiply, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam

def create_weighted_input_model5(input_shape):
    # 입력 레이어
    inputs = Input(shape=input_shape)

    # 가중치를 적용할 입력 변수 설정 (고도와 속도)
    altitude_weight = 1.5  # 고도에 더 높은 가중치
    speed_weight = 1.5     # 속도에 더 높은 가중치

    # 입력의 고도와 속도에 가중치 적용
    altitude_scaled = Lambda(lambda x: x[:, :, 2:3] * altitude_weight)(inputs)  # 고도 (세 번째 입력)
    speed_scaled = Lambda(lambda x: x[:, :, 3:4] * speed_weight)(inputs)  # 속도 (네 번째 입력)

    # 나머지 입력 변수들을 선택 (위도, 경도) - 풍속과 풍향 제외
    other_inputs = Lambda(lambda x: tf.gather(x, [0, 1], axis=2))(inputs)

    # 가중치가 적용된 고도, 속도와 나머지 입력을 합침
    combined_inputs = Concatenate(axis=2)([altitude_scaled, speed_scaled, other_inputs])

    # GRU 레이어
    gru_out = GRU(64, return_sequences=True)(combined_inputs)
    gru_out = GRU(64)(gru_out)

    # 출력 레이어 (위도, 경도, 고도 예측)
    outputs = Dense(3)(gru_out)

    # 모델 생성
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model