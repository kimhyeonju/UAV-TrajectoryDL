import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam

# 데이터 스케일러 및 모델 로드
scaler = MinMaxScaler(feature_range=(0, 1))
model_path = '/Users/admin/PycharmProjects/GP_test/GP_final/aircraft/model/look_back=30&forward=5/GRU_model.h5'
model = load_model(model_path)
model.compile(optimizer=Adam(), loss='mean_squared_error')  # 모델 로드 후 재컴파일

# 새로운 데이터 로드
new_data_path = '/Users/admin/PycharmProjects/GP_test/data/interpolation_s/test/Inter_GreaterSapiens Flight Leg 3 Perth Antarctica.csv'
new_df = pd.read_csv(new_data_path)

# 필터링할 열
filtered_df = new_df[['Latitude', 'Longitude', 'Altitude (m)']]

# 데이터 스케일링
scaled_data = scaler.fit_transform(filtered_df.values)

# look_back 및 forward_length 설정
look_back = 30
forward_length = 5

# 학습 데이터와 라벨 생성 함수
def create_dataset(data, start_index, look_back, forward_length):
    X = data[start_index:start_index + look_back]
    y = data[start_index + look_back + forward_length]
    return np.array(X), np.array(y)

# 초기 데이터 설정 (look_back 길이만큼)
start_index = 0
predictions = []
actuals = []
mse_values = []


# MSE 로그 파일 설정
mse_log_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/aircraft/look_back=30&forward=5/All/mse_log_gerneral.txt'

# 반복 예측 및 MSE 계산
with open(mse_log_path, 'w') as mse_log_file:
    while start_index + look_back + forward_length < len(scaled_data):
        # 학습 데이터 준비
        if start_index >= look_back:
            X_train, y_train = create_dataset(scaled_data, start_index - look_back, look_back, forward_length)
            X_train = np.expand_dims(X_train, axis=0)  # 배치 차원 추가
            y_train = np.expand_dims(y_train, axis=0)

            # 모델 학습
            model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

            # 예측 수행
            input_sequence = scaled_data[start_index:start_index + look_back]
            input_sequence = np.expand_dims(input_sequence, axis=0)  # 배치 차원 추가
            predicted_point = model.predict(input_sequence)
            predicted_coordinates = scaler.inverse_transform(predicted_point)[0]
            predictions.append(predicted_coordinates)

            # 실제 값 저장
            actual_coordinates = scaler.inverse_transform([scaled_data[start_index + look_back + forward_length]])[0]
            actuals.append(actual_coordinates)

            # MSE 계산 및 저장
            mse = mean_squared_error([actual_coordinates], [predicted_coordinates])
            mse_values.append(mse)
            mse_log_file.write(f"Index {start_index}, MSE: {mse}\n")

        # 인덱스 업데이트
        start_index += 1

# 전체 MSE 계산
overall_mse = np.mean(mse_values)
print(f"Overall MSE: {overall_mse}")

# 전체 MSE를 로그 파일에 추가 저장
with open(mse_log_path, 'a') as mse_log_file:
    mse_log_file.write(f"\nOverall MSE: {overall_mse}\n")


# 예측된 지점을 DataFrame으로 변환
predicted_df = pd.DataFrame(predictions, columns=['Latitude', 'Longitude', 'Altitude (m)'])
actual_df = pd.DataFrame(actuals, columns=['Latitude', 'Longitude', 'Altitude (m)'])

# 실제 데이터의 시작점을 예측 데이터와 맞추기 위해 look_back만큼 제거
filtered_df_trimmed = filtered_df.iloc[look_back + forward_length:].reset_index(drop=True)

# 예측 결과를 CSV 파일로 저장
predicted_df.to_csv('/Users/admin/PycharmProjects/GP_test/GP_final/data/aircraft/look_back=30&forward=5/All/csv/Inter_GreaterSapiens Flight Leg 3 Perth Antarctica.csv', index=False)

# 원본 데이터와 예측 데이터를 비교하는 플롯
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 실제 경로
ax.plot(filtered_df_trimmed['Longitude'], filtered_df_trimmed['Latitude'], filtered_df_trimmed['Altitude (m)'], label='Actual Path', color='b')

# 예측 경로
ax.plot(predicted_df['Longitude'], predicted_df['Latitude'], predicted_df['Altitude (m)'], label='Predicted Path', color='r')

# 축 설정
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Altitude')

# 축 범위 설정 및 간격 조정
longitude_range = [min(filtered_df_trimmed['Longitude'].min(), predicted_df['Longitude'].min()) - 1,
                   max(filtered_df_trimmed['Longitude'].max(), predicted_df['Longitude'].max()) + 1]
latitude_range = [min(filtered_df_trimmed['Latitude'].min(), predicted_df['Latitude'].min()) - 1,
                  max(filtered_df_trimmed['Latitude'].max(), predicted_df['Latitude'].max()) + 1]
altitude_range = [min(filtered_df_trimmed['Altitude (m)'].min(), predicted_df['Altitude (m)'].min()) - 1000,
                  max(filtered_df_trimmed['Altitude (m)'].max(), predicted_df['Altitude (m)'].max()) + 1000]

ax.set_xlim(longitude_range)
ax.set_ylim(latitude_range)
ax.set_zlim(altitude_range)

# 각 축의 눈금 간격 설정
ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
ax.zaxis.set_major_locator(MaxNLocator(nbins=10))

ax.legend()

# 플롯 저장
plt.savefig('/Users/admin/PycharmProjects/GP_test/GP_final/data/aircraft/look_back=30&forward=0/AllPredicted/test_plot/Inter_GreaterSapiens Flight Leg 3 Perth Antarctica_test_plot.png')
plt.close(fig)

plt.show()
