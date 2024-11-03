from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
import glob

# 데이터 폴더 경로
data_folder = 'data/Drone_data/split_300_frame'
all_files = glob.glob(os.path.join(data_folder, "*.csv"))

# 모델 아키텍처 정의 함수
def create_model():
    model = Sequential([
        Input(shape=(1, 3)),
        GRU(50),
        Dense(50, activation='relu'),
        Dense(3)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# 모델 초기화
model = create_model()

# MSE 저장을 위한 리스트 초기화
train_mse_list = []
val_mse_list = []
test_mse_list = []

# 각 파일에 대해 모델 학습
for file in all_files:
    # 데이터 로드 및 전처리
    df = pd.read_csv(file)
    df = df[['Latitude', 'Longitude', 'Altitude (m)']]

    # 정규화
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # X, y 생성
    X = df_normalized[['Latitude', 'Longitude', 'Altitude (m)']].values[:-1]
    y = df_normalized[['Latitude', 'Longitude', 'Altitude (m)']].shift(-1).values[:-1]

    # NaN 값 제거 (예측할 다음 위치가 없는 마지막 행 때문에 필요)
    X = X[:-1]
    y = y[:-1]

    # 데이터를 훈련, 검증, 테스트 세트로 분할
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.18, random_state=42)  # 0.18 * 0.85 ≈ 0.15

    # 데이터를 3차원으로 변환
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # 모델 학습
    model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), batch_size=32, verbose=2)

    # 훈련, 검증, 테스트 세트에 대한 MSE 계산
    train_mse = model.evaluate(X_train, y_train, verbose=0)
    val_mse = model.evaluate(X_val, y_val, verbose=0)
    test_mse = model.evaluate(X_test, y_test, verbose=0)

    # MSE 저장
    train_mse_list.append(train_mse)
    val_mse_list.append(val_mse)
    test_mse_list.append(test_mse)

    print(f"File: {file}")
    print(f"Train MSE: {train_mse}")
    print(f"Validation MSE: {val_mse}")
    print(f"Test MSE: {test_mse}")

# 최종 평균 MSE 계산
average_train_mse = np.mean(train_mse_list)
average_val_mse = np.mean(val_mse_list)
average_test_mse = np.mean(test_mse_list)

print("--------------------------------------------------------")
print(f"Average Train MSE: {average_train_mse}")
print(f"Average Validation MSE: {average_val_mse}")
print(f"Average Test MSE: {average_test_mse}")

# 모델 저장
model.save('GRU_model_drone.h5')
