import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Input, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


def create_weighted_input_model(input_shape):
    # 입력 레이어
    inputs = Input(shape=input_shape)

    # 고도 예측을 위한 가중치 설정 (경도와 속도에만 가중치 부여)
    longitude_weight = 1.5  # 경도에 가중치 부여
    speed_weight = 1.5  # 속도에 가중치 부여

    # 고도 예측 시 사용되는 입력: 경도와 속도에 가중치 부여
    longitude_scaled = Lambda(lambda x: x[:, :, 1:2] * longitude_weight)(inputs)  # 경도 (두 번째 입력)
    speed_scaled = Lambda(lambda x: x[:, :, 3:4] * speed_weight)(inputs)  # 속도 (네 번째 입력)

    # 가중치가 적용되지 않은 다른 입력들 (위도, 고도)
    other_inputs = Lambda(lambda x: tf.gather(x, [0, 2], axis=2))(inputs)

    # 고도 예측을 위한 입력: 가중치가 적용된 경도와 속도, 나머지 변수 (위도, 고도)
    altitude_inputs = Concatenate(axis=2)([longitude_scaled, speed_scaled, other_inputs])

    # 나머지 변수 (위도, 경도) 예측 시 사용될 입력: 가중치가 적용되지 않은 전체 입력
    general_inputs = inputs

    # GRU 레이어 (고도 예측용)
    altitude_gru_out = GRU(64, return_sequences=True)(altitude_inputs)
    altitude_gru_out = GRU(64)(altitude_gru_out)

    # GRU 레이어 (위도, 경도 예측용)
    general_gru_out = GRU(64, return_sequences=True)(general_inputs)
    general_gru_out = GRU(64)(general_gru_out)

    # 출력 레이어 (고도 예측)
    altitude_output = Dense(1, name='altitude_output')(altitude_gru_out)

    # 출력 레이어 (위도, 경도 예측)
    lat_long_output = Dense(2, name='lat_long_output')(general_gru_out)

    # 모델 생성 (위도, 경도, 고도 출력)
    model = Model(inputs=inputs, outputs=[lat_long_output, altitude_output])
    model.compile(optimizer=Adam(), loss='mean_squared_error')

    return model

input_shape = (10, 4)  # 예시 입력 크기 설정 (look_back = 10, 입력 파라미터는 4개)
model = create_weighted_input_model(input_shape)

# # 모델 구조 출력
model.summary()

plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)
