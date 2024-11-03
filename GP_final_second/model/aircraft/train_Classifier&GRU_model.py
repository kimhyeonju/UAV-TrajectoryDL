import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Input, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# 1. 구간 예측 모델 생성 함수
def create_classifier_model(input_shape):
    inputs = Input(shape=input_shape)
    x = GRU(64, return_sequences=True)(inputs)
    x = GRU(64)(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 2. 경로 예측 모델 생성 함수
def create_weighted_input_model(input_shape, phase_weights):
    inputs = Input(shape=input_shape)

    # 구간에 따른 가중치 적용
    altitude_weight = phase_weights['altitude']
    speed_weight = phase_weights['speed']

    altitude_scaled = Lambda(lambda x: x[:, :, 2:3] * altitude_weight)(inputs)
    speed_scaled = Lambda(lambda x: x[:, :, 3:4] * speed_weight)(inputs)

    other_inputs = Lambda(lambda x: x[:, :, :2])(inputs)
    combined_inputs = Concatenate(axis=2)([other_inputs, altitude_scaled, speed_scaled])

    x = GRU(128, return_sequences=True)(combined_inputs)
    x = GRU(128)(x)
    outputs = Dense(3)(x)  # 위도, 경도, 고도 예측
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model


# 3. 데이터 준비 함수 (Look_back + Phase 레이블)
def create_dataset_with_phase_labels(data, look_back, forward_length):
    X, y, labels = [], [], []
    for i in range(len(data) - look_back - forward_length):
        X.append(data[i:(i + look_back), :4])  # 입력 파라미터: 위도, 경도, 고도, 속도
        y.append(data[i + look_back + forward_length, :3])  # 예측할 파라미터: 위도, 경도, 고도
        labels.append(data[i, 4])  # Phase 레이블 (이륙, 순항, 착륙)
    return np.array(X), np.array(y), np.array(labels)


# 4. 학습 및 테스트 과정
def train_and_evaluate(data_folder, look_back, forward_length):
    file_list = [file for file in os.listdir(data_folder) if file.endswith('.csv')]

    classifier_model = create_classifier_model(input_shape=(look_back, 4))  # 구간 예측 모델
    phase_weights = {'takeoff': {'altitude': 1.5, 'speed': 1.0},
                     'cruise': {'altitude': 1.0, 'speed': 1.5},
                     'landing': {'altitude': 2.0, 'speed': 1.0}}  # 구간별 가중치

    scaler = MinMaxScaler(feature_range=(0, 1))
    for file_name in file_list:
        file_path = os.path.join(data_folder, file_name)
        df = pd.read_csv(file_path)

        # 데이터 전처리
        df_filtered = df[['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)', 'Phase']]
        df_filtered.loc[:, ['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)']] = scaler.fit_transform(
            df_filtered[['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)']])

        # 데이터셋 준비
        X, y, labels = create_dataset_with_phase_labels(df_filtered.values, look_back, forward_length)
        y_labels = to_categorical(labels, num_classes=3)  # 이륙, 순항, 착륙 라벨 인코딩

        # 데이터 분할
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        labels_train, labels_test = y_labels[:split_index], y_labels[split_index:]

        # 구간 예측 학습
        classifier_model.fit(X_train, labels_train, epochs=50, batch_size=32, verbose=1)

        # 구간 예측 테스트 및 결과
        phase_predictions = classifier_model.predict(X_test)
        phase_accuracy = np.mean(np.argmax(phase_predictions, axis=1) == np.argmax(labels_test, axis=1))
        print(f"{file_name} - 구간 예측 정확도: {phase_accuracy:.4f}")

        # 경로 예측 모델 가중치 결정
        param_names = ['Latitude', 'Longitude', 'Altitude (m)']
        weights = {'altitude': 1.0, 'speed': 1.0}  # 기본 가중치

        # 경로 예측 모델 학습 (한 번 학습 후 전체 테스트 데이터에 대해 예측)
        for i in range(X_test.shape[0]):
            predicted_phase = np.argmax(phase_predictions[i])
            if predicted_phase == 0:  # 이륙
                weights = phase_weights['takeoff']
            elif predicted_phase == 1:  # 순항
                weights = phase_weights['cruise']
            else:  # 착륙
                weights = phase_weights['landing']

        path_model = create_weighted_input_model(input_shape=(look_back, 4), phase_weights=weights)
        path_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

        # 전체 테스트 데이터로 예측
        test_predictions = path_model.predict(X_test)

        # 출력 파라미터 별 MSE 계산
        for idx, param in enumerate(param_names):
            mse = mean_squared_error(y_test[:, idx], test_predictions[:, idx])
            print(f"{file_name} - {param} MSE: {mse:.4f}")


# 학습 및 평가 실행
data_folder = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Labeled_Routes'
look_back = 10
forward_length = 0
train_and_evaluate(data_folder, look_back, forward_length)