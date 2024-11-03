import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Input, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
# from torchviz import make_dot

# 모델 구조를 이미지 파일로 저장


def create_weighted_input_model(input_shape):
    # 입력 레이어
    inputs = Input(shape=input_shape)

    # 고도 예측을 위한 가중치 설정 (고도에만 가중치 부여)
    altitude_weight = 1.5  # 고도에 가중치 부여

    # 고도 예측 시 사용되는 입력: 고도에 가중치 부여
    altitude_scaled = Lambda(lambda x: x[:, :, 2:3] * altitude_weight)(inputs)  # 고도 (세 번째 입력)

    # 가중치가 적용되지 않은 다른 입력들 (위도, 경도, 속도)
    other_inputs = Lambda(lambda x: tf.gather(x, [0, 1, 3], axis=2))(inputs)

    # 고도 예측을 위한 입력: 가중치가 적용된 고도, 나머지 변수 (위도, 경도, 속도)
    altitude_inputs = Concatenate(axis=2)([altitude_scaled, other_inputs])

    # GRU 레이어 (고도 예측용)
    altitude_gru_out = GRU(128, return_sequences=True)(altitude_inputs)
    altitude_gru_out = GRU(128)(altitude_gru_out)

    # GRU 레이어 (위도, 경도 예측용)
    general_gru_out = GRU(128, return_sequences=True)(inputs)
    general_gru_out = GRU(128)(general_gru_out)

    # 출력 레이어 (고도 예측)
    altitude_output = Dense(1, name='altitude_output')(altitude_gru_out)

    # 출력 레이어 (위도, 경도 예측)
    lat_long_output = Dense(2, name='lat_long_output')(general_gru_out)

    # 모델 생성 (위도, 경도, 고도 출력)
    model = Model(inputs=inputs, outputs=[lat_long_output, altitude_output])
    model.compile(optimizer=Adam(), loss='mean_squared_error')

    return model

# 모델 생성
input_shape = (10, 4)  # 예시 입력 형태 (타임스텝=10, 특성=4)
model = create_weighted_input_model(input_shape)

# 모델 구조 요약 출력
model.summary()

plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)



