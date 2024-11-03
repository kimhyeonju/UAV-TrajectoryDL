import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Input, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt



# 통합된 모델 생성 함수
def create_integrated_model(input_shape):
    inputs = Input(shape=input_shape)

    # 공유 GRU 레이어로 입력 데이터 처리
    shared_gru = GRU(64, return_sequences=True)(inputs)
    shared_gru = GRU(64)(shared_gru)

    # 구간 예측(Phase Classification) 분기
    phase_output = Dense(3, activation='softmax', name="phase_output")(shared_gru)

    # 경로 예측(Trajectory Prediction) 분기
    altitude_weight = 1.5
    speed_weight = 1.0

    altitude_scaled = Lambda(lambda x: x[:, :, 2:3] * altitude_weight)(inputs)
    speed_scaled = Lambda(lambda x: x[:, :, 3:4] * speed_weight)(inputs)
    other_inputs = Lambda(lambda x: x[:, :, :2])(inputs)
    combined_inputs = Concatenate(axis=2)([other_inputs, altitude_scaled, speed_scaled])

    path_gru = GRU(128, return_sequences=True)(combined_inputs)
    path_gru = GRU(128)(path_gru)
    path_output = Dense(3, name="path_output")(path_gru)

    # 통합 모델 생성
    model = Model(inputs=inputs, outputs=[phase_output, path_output])
    model.compile(optimizer=Adam(),
                  loss={'phase_output': 'categorical_crossentropy', 'path_output': 'mean_squared_error'},
                  metrics={'phase_output': 'accuracy', 'path_output': 'mse'})
    return model


# 데이터 준비 함수 (Look_back + Phase 레이블)
def create_dataset_with_phase_labels(data, look_back, forward_length):
    X, y, labels = [], [], []
    for i in range(len(data) - look_back - forward_length):
        X.append(data[i:(i + look_back), :4])  # 위도, 경도, 고도, 속도
        y.append(data[i + look_back + forward_length, :3])  # 예측할 파라미터: 위도, 경도, 고도
        labels.append(data[i, 4])  # Phase 레이블 (이륙, 순항, 착륙)
    return np.array(X), np.array(y), np.array(labels)


# 학습 및 평가 함수
# 학습 및 테스트 과정 + 시각화
def train_and_evaluate(data_folder, look_back, forward_length):
    file_list = [file for file in os.listdir(data_folder) if file.endswith('.csv')]

    model = create_integrated_model(input_shape=(look_back, 4))  # 통합 모델
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
        y_labels = to_categorical(labels, num_classes=3)  # 구간 라벨 원핫 인코딩

        # 데이터 분할
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        labels_train, labels_test = y_labels[:split_index], y_labels[split_index:]

        # 모델 학습
        model.fit(X_train, {'phase_output': labels_train, 'path_output': y_train},
                  epochs=50, batch_size=32, verbose=1)

        # 모델 평가
        evaluation = model.evaluate(X_test, {'phase_output': labels_test, 'path_output': y_test})
        print(f"{file_name} - 평가 결과: {evaluation}")

        # 구간 예측과 경로 예측 결과
        phase_predictions, path_predictions = model.predict(X_test)

        # 구간 예측 정확도
        phase_accuracy = np.mean(np.argmax(phase_predictions, axis=1) == np.argmax(labels_test, axis=1))
        print(f"{file_name} - 구간 예측 정확도: {phase_accuracy:.7f}")

        # 경로 예측 MSE 계산
        param_names = ['Latitude', 'Longitude', 'Altitude (m)']
        for idx, param in enumerate(param_names):
            mse = mean_squared_error(y_test[:, idx], path_predictions[:, idx])
            print(f"{file_name} - {param} MSE: {mse:.4f}")

        # 시각화: 테스트 실제 경로와 예측 경로 비교 (3D)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # 테스트 실제 경로 (검정색)
        ax.plot(df_filtered['Longitude'][split_index:], df_filtered['Latitude'][split_index:],
                df_filtered['Altitude (m)'][split_index:], label='Test Actual Path', color='black', alpha=0.6)

        # 예측 경로 (녹색)
        ax.plot(path_predictions[:, 1], path_predictions[:, 0], path_predictions[:, 2],
                label='Predicted Path', color='green')

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Altitude')
        plt.title(f"{file_name} - 경로 비교")
        ax.legend()

        # 플롯 저장
        # plt.savefig(f'/path_to_save/{file_name}_test_plot.png')
        plt.show()



# 학습 및 평가 실행
data_folder = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Labeled_Routes'
look_back = 10
forward_length = 0
train_and_evaluate(data_folder, look_back, forward_length)