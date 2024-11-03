import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam


def create_flight_phase_classifier(input_shape):
    # 입력 레이어
    inputs = Input(shape=input_shape)

    # GRU 레이어로 구간 분류
    gru_out = GRU(64, return_sequences=True)(inputs)
    gru_out = GRU(64)(gru_out)

    # 구간 분류 출력 (이륙, 순항, 착륙에 대한 softmax 출력)
    classification_output = Dense(3, activation='softmax', name="classification_output")(gru_out)

    # 모델 생성
    classifier_model = Model(inputs=inputs, outputs=classification_output)
    classifier_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return classifier_model