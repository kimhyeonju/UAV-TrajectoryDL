import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Input, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# 1. Focal-like weighted categorical crossentropy for phase output
def weighted_categorical_crossentropy(weights):
    """
    손실 함수에 가중치를 적용하는 함수.
    weights: 각 클래스별 가중치 리스트 (예: [1.5, 1.0, 2.0])
    """
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # 예측값 안정성 확보
        weight_per_class = K.constant(weights)
        weighted_losses = y_true * K.log(y_pred) * weight_per_class
        loss = -K.sum(weighted_losses, axis=-1)
        return loss
    return loss

# 2. 통합된 모델 생성 함수 (구간 예측 + 경로 예측)
def create_combined_model(input_shape, phase_class_weights):
    # 입력 레이어
    inputs = Input(shape=input_shape)

    # 구간 예측용 GRU
    gru_phase = GRU(128, return_sequences=True)(inputs)
    gru_phase = GRU(128)(gru_phase)
    phase_output = Dense(3, activation='softmax', name='phase_output')(gru_phase)

    # 경로 예측용 가중치 적용
    altitude_weight = Lambda(lambda x: x[:, :, 2:3] * phase_class_weights['altitude'])(inputs)
    speed_weight = Lambda(lambda x: x[:, :, 3:4] * phase_class_weights['speed'])(inputs)
    other_inputs = Lambda(lambda x: x[:, :, :2])(inputs)

    combined_inputs = Concatenate(axis=2)([other_inputs, altitude_weight, speed_weight])

    # 경로 예측용 GRU
    gru_path = GRU(128, return_sequences=True)(combined_inputs)
    gru_path = GRU(128)(gru_path)
    path_output = Dense(3, name='path_output')(gru_path)  # 위도, 경도, 고도 예측

    # 통합 모델 생성
    model = Model(inputs=inputs, outputs=[path_output, phase_output])

    # 손실 함수 가중치 적용
    model.compile(optimizer=Adam(),
                  loss={'path_output': 'mean_squared_error',
                        'phase_output': weighted_categorical_crossentropy([1.5, 1.0, 2.0])},  # 이 부분 수정
                  metrics={'path_output': 'mse', 'phase_output': 'accuracy'})

    return model


# 3. 데이터셋 생성 함수 (Look_back + Phase 레이블)
def create_dataset_with_phase_labels(data, look_back, forward_length):
    X, y, labels = [], [], []
    for i in range(len(data) - look_back - forward_length):
        X.append(data[i:(i + look_back), :4])  # 위도, 경도, 고도, 속도
        y.append(data[i + look_back + forward_length, :3])  # 예측할 위도, 경도, 고도
        labels.append(data[i + look_back, 4])  # 구간 레이블 (이륙, 순항, 착륙)
    return np.array(X), np.array(y), np.array(labels)


# 4. 시각화 함수 (테스트 결과 플롯)
def plot_test_results(df, predicted_df, file_name):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 원본 경로
    ax.plot(df['Longitude'], df['Latitude'], df['Altitude (m)'],
            label='Actual Path', color='blue')

    # 예측 경로
    ax.plot(predicted_df['Longitude'], predicted_df['Latitude'], predicted_df['Altitude (m)'],
            label='Predicted Path', color='red')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude')
    plt.title(f"Test Results for {file_name}")
    ax.legend()
    plt.savefig(f"{file_name}_test_plot.png")
    plt.close(fig)


# 5. 학습 및 평가 함수
def train_and_evaluate(data_folder, look_back, forward_length, phase_weights):
    file_list = [file for file in os.listdir(data_folder) if file.endswith('.csv')]

    scaler = MinMaxScaler(feature_range=(0, 1))

    for file_name in file_list:
        file_path = os.path.join(data_folder, file_name)
        df = pd.read_csv(file_path)

        # 데이터 전처리
        df_filtered = df[['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)', 'Phase']]
        df_filtered.loc[:, ['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)']] = scaler.fit_transform(
            df_filtered[['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)']])

        # 데이터셋 생성
        X, y, labels = create_dataset_with_phase_labels(df_filtered.values, look_back, forward_length)
        y_labels = to_categorical(labels, num_classes=3)

        # 데이터 분할
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        labels_train, labels_test = y_labels[:split_index], y_labels[split_index:]

        # 통합 모델 생성
        model = create_combined_model(input_shape=(look_back, 4), phase_class_weights=phase_weights)
        # 학습
        model.fit(X_train, {'path_output': y_train, 'phase_output': labels_train},
                  epochs=50, batch_size=32, verbose=1)

        # 테스트 예측
        test_results = model.evaluate(X_test, {'path_output': y_test, 'phase_output': labels_test})
        print(f"{file_name} - 평가 결과: {test_results}")

        # 경로 예측 수행 및 MSE 계산
        test_predictions = model.predict(X_test)[0]
        mse = mean_squared_error(y_test, test_predictions)
        print(f"{file_name} - 경로 예측 MSE: {mse:.7f}")

        # 예측 결과 DataFrame 생성
        predicted_coordinates = scaler.inverse_transform(
            np.concatenate([test_predictions, np.zeros((test_predictions.shape[0], 1))], axis=1))
        predicted_df = pd.DataFrame(predicted_coordinates[:, :3], columns=['Latitude', 'Longitude', 'Altitude (m)'])

        # 시각화
        plot_test_results(df_filtered, predicted_df, file_name)


# 6. 가중치 설정
phase_weights = {'altitude': 1.5, 'speed': 1.0}

# 학습 및 평가 실행
data_folder = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Labeled_Routes'
look_back = 10
forward_length = 0
train_and_evaluate(data_folder, look_back, forward_length, phase_weights)