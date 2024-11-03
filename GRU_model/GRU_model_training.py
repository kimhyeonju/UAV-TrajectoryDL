import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import os
import time
from GRU_model_v2 import create_gru_model

# 데이터 스케일링을 위한 MinMaxScaler 인스턴스 생성
scaler = MinMaxScaler(feature_range=(0, 1))

# 학습 데이터와 라벨 생성 함수
def create_dataset(data, look_back, forward_length):
    X, y = [], []
    for i in range(len(data) - look_back - forward_length + 1):
        X.append(data[i:(i + look_back), :])
        y.append(data[i + look_back + forward_length - 1, :])
    return np.array(X), np.array(y)

# 파일 폴더 경로 설정
folder_path = '/Users/admin/PycharmProjects/GP_test/data/split_10s'
file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    print('--------------------------------------------------------')
    print('')
    print(f"Processing file: {file_name}")

    df = pd.read_csv(file_path)

    df_filtered = df[['Latitude', 'Longitude', 'Altitude (m)']]

    # 데이터 스케일링
    df_filtered.loc[:, ['Latitude', 'Longitude', 'Altitude (m)']] = scaler.fit_transform(df_filtered[['Latitude', 'Longitude', 'Altitude (m)']])

    # 학습 데이터와 라벨 생성
    look_back = 10
    forward_length = 5
    X, y = create_dataset(df_filtered.values, look_back, forward_length)

    if len(X) < 5:  # 데이터가 적을 경우
        print(f"Not enough data to perform cross-validation for file: {file_name}")
        continue

    # 데이터를 학습용과 테스트용으로 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 생성
    model = create_gru_model(input_shape=(look_back, 3))

    # 모델 학습
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # 예측 시간 측정
    start_time = time.time()
    predictions = model.predict(X_test)
    end_time = time.time()

    # MSE 계산 및 예측 시간 기록
    mse = mean_squared_error(y_test, predictions)
    prediction_time_per_sample = (end_time - start_time) / len(X_test)

    # 결과 출력
    print(f"File: {file_name}, MSE: {mse}, Prediction Time per Sample: {prediction_time_per_sample:.5f} seconds")

# 모델 구조를 이미지로 저장
plot_model(model, to_file='GRU_model_v3_aircraft_structure.png', show_shapes=True, show_layer_names=True)

# 모델 저장
model.save('GRU_model_v3.h5')
