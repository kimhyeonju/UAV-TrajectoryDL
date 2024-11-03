import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator


# 데이터 스케일러 및 모델 로드/생성
scaler = MinMaxScaler(feature_range=(0, 1))
model_path = 'GRU_model_v3.h5'

model = load_model(model_path)
model.compile(optimizer=Adam(), loss='mean_squared_error')  # 모델 로드 후 재컴파일

# 새로운 데이터 로드
new_data_path = '/Users/admin/PycharmProjects/GP_test/data/split_10s/inter_GreaterSapiens Flight Leg 7 GRU BOG_split_10s.csv'
new_df = pd.read_csv(new_data_path)

# 필터링할 열
filtered_df = new_df[['Latitude', 'Longitude', 'Altitude (m)']]

# 데이터 스케일링
scaled_data = scaler.fit_transform(filtered_df.values)

# look_back 및 forward_length 설정
look_back = 50
forward_length = 0

# 특정 지점에서 look_back 길이의 입력 데이터 생성 함수
def prepare_input_data(data, start_index, look_back):
    if start_index - look_back + 1 < 0:
        raise ValueError("Not enough data to create look_back sequence")
    return data[start_index - look_back + 1: start_index + 1, :]

# 지점 지정 (예: 50번째 지점)
start_index = 51

# 학습 데이터 생성 (지정된 지점에서 look_back 이전부터 지정된 지점까지)
X_train = prepare_input_data(scaled_data, start_index, look_back)
X_train = np.expand_dims(X_train, axis=0)  # 배치 차원 추가

# 예측할 y_train 설정 (forward_length 이후의 지점)
y_index = start_index + forward_length
if y_index >= len(scaled_data):
    raise ValueError("Not enough data to create y_train sequence")

y_train = np.expand_dims(scaled_data[y_index], axis=0)  # forward_length 이후의 데이터

# 모델 학습
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=0)

# 예측 수행을 위한 초기 입력 데이터 준비
input_data = prepare_input_data(scaled_data, start_index, look_back)
input_data = np.expand_dims(input_data, axis=0)  # 배치 차원 추가

# 예측 수행
predicted_point = model.predict(input_data)
predicted_coordinates = scaler.inverse_transform(predicted_point)

# 결과 출력 (예측된 지점과 원본 데이터 지점 함께 출력)
print("Predicted and Original Coordinates:")

original_coords = filtered_df.iloc[y_index].values
print(f"Predicted:")
print(f"Latitude: {predicted_coordinates[0][0]}, Longitude: {predicted_coordinates[0][1]}, Altitude: {predicted_coordinates[0][2]}")
print(f"Original:")
print(f"Latitude: {original_coords[0]}, Longitude: {original_coords[1]}, Altitude: {original_coords[2]}")

# 3D 플롯
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 예측된 좌표
ax.scatter(predicted_coordinates[0][0], predicted_coordinates[0][1], predicted_coordinates[0][2], label='Predicted Point', color='r', marker='o')

# 실제 좌표
ax.scatter(original_coords[0], original_coords[1], original_coords[2], label='Actual Point', color='b', marker='^')
