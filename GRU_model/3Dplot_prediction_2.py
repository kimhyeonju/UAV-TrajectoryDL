import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tensorflow.keras.optimizers import Adam


# 데이터 스케일러 및 모델 로드
scaler = MinMaxScaler(feature_range=(0, 1))
model_path = 'GRU_model_v4_drone.h5'
model = load_model(model_path)
model.compile(optimizer=Adam(), loss='mean_squared_error')  # 모델 로드 후 재컴파일


# 새로운 데이터 로드
new_data_path = '/Users/admin/PycharmProjects/GP_test/data/Drone_data/inter_대구_수성못_202008301743_60m_45도_1_split_30.csv'
new_df = pd.read_csv(new_data_path)

# 필터링할 열
filtered_df = new_df[['Latitude', 'Longitude', 'Altitude (m)']]

# 데이터 스케일링
scaled_data = scaler.fit_transform(filtered_df.values)

# look_back 및 forward_length 설정
look_back = 10
forward_length = 5

# 학습 데이터와 라벨 생성 함수
def create_dataset(data, start_index, look_back, forward_length):
    X = data[start_index:start_index + look_back]
    y = data[start_index + look_back + forward_length]
    return np.array(X), np.array(y)

# 초기 데이터 설정 (look_back 길이만큼)
start_index = 0
predictions = [None] * (look_back + forward_length)  # look_back + forward_length 길이만큼의 None으로 초기화

# 반복 예측
while start_index + look_back + forward_length < len(scaled_data):
    # 학습 데이터 준비
    if start_index >= look_back:
        X_train, y_train = create_dataset(scaled_data, start_index - look_back, look_back, forward_length)
        X_train = np.expand_dims(X_train, axis=0)  # 배치 차원 추가
        y_train = np.expand_dims(y_train, axis=0)

        # 모델 학습
        model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=0)

    # 예측 수행
    input_sequence = scaled_data[start_index:start_index + look_back]
    input_sequence = np.expand_dims(input_sequence, axis=0)  # 배치 차원 추가
    predicted_point = model.predict(input_sequence)
    predicted_coordinates = scaler.inverse_transform(predicted_point)[0]
    predictions.append(predicted_coordinates)

    # 인덱스 업데이트
    start_index += 1

# None 값을 제외하고 예측된 지점을 DataFrame으로 변환
predictions = [pred for pred in predictions if pred is not None]
predicted_df = pd.DataFrame(predictions, columns=['Latitude', 'Longitude', 'Altitude (m)'])

predicted_df.to_csv('inter_대구_수성못_202008301743_60m_45도_1_split_30_predicted_results_드론모델_이륙강화.csv', index=False)

# 원본 데이터와 예측 데이터를 비교하는 플롯
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 실제 경로
ax.plot(filtered_df['Longitude'], filtered_df['Latitude'], filtered_df['Altitude (m)'], label='Actual Path', color='b')

# 예측 경로
ax.plot(predicted_df['Longitude'], predicted_df['Latitude'], predicted_df['Altitude (m)'], label='Predicted Path', color='r')

# 축 설정
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Altitude')

# 축 범위 설정 및 간격 조정
latitude_range = [filtered_df['Latitude'].min(), filtered_df['Latitude'].max()]
longitude_range = [filtered_df['Longitude'].min(), filtered_df['Longitude'].max()]
altitude_range = [filtered_df['Altitude (m)'].min(), filtered_df['Altitude (m)'].max()]

ax.set_xlim(longitude_range)
ax.set_ylim(latitude_range)
ax.set_zlim(altitude_range)

# 각 축의 눈금 간격 설정
ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
ax.zaxis.set_major_locator(MaxNLocator(nbins=10))

ax.legend()
plt.show()
