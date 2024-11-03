import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
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
folder_path = './data/split_10s/'
file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# K-fold 교차 검증 설정
folds = 5
kf = KFold(n_splits=folds, shuffle=True, random_state=42)

for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    print('--------------------------------------------------------')
    print('')
    print(f"Processing file: {file_name}")

    df = pd.read_csv(file_path)

    # # DataFrame에서 'frame_id' 열을 활용하여 첫 'frame_id' 값에서 시작하여 300 간격으로 데이터 선택
    # start_frame_id = df['Frame_id'].iloc[0]  # 첫 'frame_id' 값
    # max_frame_id = df['Frame_id'].iloc[-1]  # 마지막 'frame_id' 값
    #
    # # 모든 가능한 300 간격의 'frame_id' 계산
    # frame_ids = range(start_frame_id, max_frame_id + 1, 300)
    #
    # # 이러한 'frame_id'를 기준으로 필터링
    # df_filtered = df[df['Frame_id'].isin(frame_ids)]

    df_filtered = df[['Latitude', 'Longitude', 'Altitude (m)']]

    # 데이터 스케일링
    df_filtered.loc[:, ['Latitude', 'Longitude', 'Altitude (m)']] = scaler.fit_transform(df_filtered[['Latitude', 'Longitude', 'Altitude (m)']])

    # 학습 데이터와 라벨 생성
    look_back = 10
    forward_length = 5
    X, y = create_dataset(df_filtered.values, look_back, forward_length)

    if len(X) < 5:  # 데이터가 교차 검증 분할 수보다 적을 경우
        print(f"Not enough data to perform 5 folds cross-validation for file: {file_name}")
        continue

    # 교차 검증 및 학습 실행
    scores = []
    fold_prediction_times = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

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
        scores.append(mse)
        fold_prediction_times.append((end_time - start_time) / len(X_test))

    # 결과 출력
    average_mse = np.mean(scores)
    average_prediction_time = np.mean(fold_prediction_times)
    print(f"Average MSE: {average_mse}, Average Prediction Time: {average_prediction_time:.5f} seconds")

model.summary()

# 모델 구조를 이미지로 저장
plot_model(model, to_file='GRU_model_v2_aircraft_structure.png', show_shapes=True, show_layer_names=True)

# 모델 저장
model.save('GRU_model_v2.h5')
