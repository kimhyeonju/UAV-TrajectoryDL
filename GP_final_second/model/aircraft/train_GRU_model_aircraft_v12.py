import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from GRU_model_v12 import create_weighted_input_model
from tensorflow.keras.optimizers import Adam
from matplotlib.ticker import MaxNLocator

# 데이터 스케일러 인스턴스 생성
scaler = MinMaxScaler(feature_range=(0, 1))

# 학습 데이터와 라벨 생성 함수 (위도, 경도, 고도 분리)
def create_dataset(data, look_back, forward_length):
    X, y_lat_long, y_altitude = [], [], []
    for i in range(len(data) - look_back - forward_length):
        X.append(data[i:(i + look_back), :4])  # 입력 파라미터: 위도, 경도, 고도, 속도
        y_lat_long.append(data[i + look_back + forward_length, :2])  # 예측할 파라미터: 위도, 경도 (2개)
        y_altitude.append(data[i + look_back + forward_length, 2])  # 예측할 파라미터: 고도 (1개)
    return np.array(X), np.array(y_lat_long), np.array(y_altitude)

# 파일 폴더 경로 설정
folder_path = r'C:\Users\kimhyeonju\PycharmProjects\GP_test\GP_final_second\data\DataProcess\Aircraft_Results'
file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# 결과 저장 경로 설정
predictions_path = r'C:\Users\kimhyeonju\PycharmProjects\GP_test\GP_final_second\data\aircraft\GRU_model_v13\look_back=40&forward=25\test_predictions_general/'
plots_path = r'C:\Users\kimhyeonju\PycharmProjects\GP_test\GP_final_second\data\aircraft\GRU_model_v13\look_back=40&forward=25\test_plot_general/'

# 디렉토리 생성
os.makedirs(predictions_path, exist_ok=True)
os.makedirs(plots_path, exist_ok=True)

# 표준 출력을 파일로 리디렉션
log_file_path = r'C:\Users\kimhyeonju\PycharmProjects\GP_test\GP_final_second\data\aircraft\GRU_model_v13\look_back=40&forward=25/training_log_general.txt'
sys.stdout = open(log_file_path, 'w')

# look_back 및 forward_length 설정
look_back = 40
forward_length = 25

# 전체 MSE 저장 변수 초기화
total_train_mse = 0
total_test_mse = 0
total_files = len(file_list)

for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    print('--------------------------------------------------------')
    print(f"Processing file: {file_name}")

    df = pd.read_csv(file_path)

    # Speed (m/s), wind_speed, wind_direction 추가
    df_filtered = df[['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)']]

    # 데이터 스케일링
    df_filtered.loc[:, ['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)']] = scaler.fit_transform(
        df_filtered[['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)']])

    # 학습 데이터와 테스트 데이터 분할
    X, y_lat_long, y_altitude = create_dataset(df_filtered.values, look_back, forward_length)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train_lat_long, y_test_lat_long = y_lat_long[:split_index], y_lat_long[split_index:]
    y_train_altitude, y_test_altitude = y_altitude[:split_index], y_altitude[split_index:]

    # 모델 생성 및 학습 (입력 파라미터는 4개, 예측은 2개(위도, 경도) + 1개(고도))
    model = create_weighted_input_model(input_shape=(look_back, 4))
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    model.fit(X_train, [y_train_lat_long, y_train_altitude], epochs=100, batch_size=32, verbose=1)

    # 학습 데이터의 MSE 계산
    train_predictions_lat_long, train_predictions_altitude = model.predict(X_train)
    train_mse_lat_long = mean_squared_error(y_train_lat_long, train_predictions_lat_long)
    train_mse_altitude = mean_squared_error(y_train_altitude, train_predictions_altitude)

    # 결합된 전체 학습 MSE 계산
    train_predictions_combined = np.concatenate([train_predictions_lat_long, train_predictions_altitude.reshape(-1, 1)],
                                                axis=1)
    y_train_combined = np.concatenate([y_train_lat_long, y_train_altitude.reshape(-1, 1)], axis=1)
    total_train_mse += mean_squared_error(y_train_combined, train_predictions_combined)

    print(f"Train MSE for {file_name}: {total_train_mse}")

    # 테스트 데이터의 MSE 계산 및 예측
    start_time = time.time()
    test_predictions_lat_long, test_predictions_altitude = model.predict(X_test)
    end_time = time.time()

    test_mse_lat_long = mean_squared_error(y_test_lat_long, test_predictions_lat_long)
    test_mse_altitude = mean_squared_error(y_test_altitude, test_predictions_altitude)

    # 결합된 전체 테스트 MSE 계산
    test_predictions_combined = np.concatenate([test_predictions_lat_long, test_predictions_altitude.reshape(-1, 1)],
                                               axis=1)
    y_test_combined = np.concatenate([y_test_lat_long, y_test_altitude.reshape(-1, 1)], axis=1)
    test_mse = mean_squared_error(y_test_combined, test_predictions_combined)
    total_test_mse += test_mse

    # 추가: RMSE, MAE 계산 및 출력
    test_rmse_lat_long = np.sqrt(test_mse_lat_long)
    test_rmse_altitude = np.sqrt(test_mse_altitude)

    test_mae_lat_long = mean_absolute_error(y_test_lat_long, test_predictions_lat_long)
    test_mae_altitude = mean_absolute_error(y_test_altitude, test_predictions_altitude)

    print(f"Test MSE (Latitude, Longitude) for {file_name}: {test_mse_lat_long}")
    print(f"Test RMSE (Latitude, Longitude) for {file_name}: {test_rmse_lat_long}")
    print(f"Test MAE (Latitude, Longitude) for {file_name}: {test_mae_lat_long}")

    print(f"Test MSE (Altitude) for {file_name}: {test_mse_altitude}")
    print(f"Test RMSE (Altitude) for {file_name}: {test_rmse_altitude}")
    print(f"Test MAE (Altitude) for {file_name}: {test_mae_altitude}")

    prediction_time_per_sample = (end_time - start_time) / len(X_test)
    print(f"Test MSE for {file_name}: {test_mse}, Prediction Time per Sample: {prediction_time_per_sample:.5f} seconds")

    # 테스트 데이터로 예측된 지점을 원래 스케일로 변환 (위도, 경도, 고도만)
    predicted_coordinates_with_zeros = np.concatenate(
        [test_predictions_lat_long, test_predictions_altitude.reshape(-1, 1),
         np.zeros((test_predictions_lat_long.shape[0], 1))], axis=1)
    predicted_coordinates = scaler.inverse_transform(predicted_coordinates_with_zeros)[:, :3]  # 위도, 경도, 고도만 선택

    # 테스트 데이터의 실제 지점을 원래 스케일로 변환 (위도, 경도, 고도만)
    actual_coordinates_with_zeros = np.concatenate(
        [y_test_lat_long, y_test_altitude.reshape(-1, 1), np.zeros((y_test_lat_long.shape[0], 1))], axis=1)
    actual_coordinates = scaler.inverse_transform(actual_coordinates_with_zeros)[:, :3]  # 위도, 경도, 고도만 선택

    # 예측된 지점을 DataFrame으로 변환 (위도, 경도, 고도만)
    predicted_df = pd.DataFrame(predicted_coordinates[:, :3], columns=['Latitude', 'Longitude', 'Altitude (m)'])


    # 실제 테스트 지점을 DataFrame으로 변환 (위도, 경도, 고도만)
    actual_df_test = pd.DataFrame(actual_coordinates[:, :3], columns=['Latitude', 'Longitude', 'Altitude (m)'])

    df_filtered_trimmed = df_filtered.iloc[look_back + forward_length:].reset_index(drop=True)

    actual_df_trimmed = pd.DataFrame(scaler.inverse_transform(df_filtered_trimmed),
                                     columns=['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)'])

    # 예측 결과를 CSV 파일로 저장
    predicted_df.to_csv(f'{predictions_path}{file_name}_test_predictions.csv', index=False)

    # 플롯 생성
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 플롯 제목 설정
    plt.title(f"{file_name}, look_back = {look_back}, forward_length = {forward_length}")

    # 학습 경로 (파란색)
    ax.plot(actual_df_trimmed['Longitude'].iloc[:split_index], actual_df_trimmed['Latitude'].iloc[:split_index],
            actual_df_trimmed['Altitude (m)'].iloc[:split_index], label='Train Path', color='b')

    # 테스트 경로 (빨간색)
    ax.plot(actual_df_trimmed['Longitude'].iloc[split_index:], actual_df_trimmed['Latitude'].iloc[split_index:],
            actual_df_trimmed['Altitude (m)'].iloc[split_index:], label='Test Path', color='r')

    # 예측 경로 (녹색)
    ax.plot(predicted_df['Longitude'], predicted_df['Latitude'], predicted_df['Altitude (m)'], label='Predicted Path',
            color='g')

    # 원본 데이터 (검정색)
    ax.plot(df_filtered['Longitude'], df_filtered['Latitude'], df_filtered['Altitude (m)'], label='Original Path',
            color='k', alpha=0.3)

    # 축 설정
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude')

    longitude_margin = (max(actual_df_trimmed['Longitude'].max(), predicted_df['Longitude'].max()) -min(actual_df_trimmed['Longitude'].min(), predicted_df['Longitude'].min()))* 0.2  # 10% 범위
    latitude_margin =  (max(actual_df_trimmed['Latitude'].max(), predicted_df['Latitude'].max()) -min(actual_df_trimmed['Latitude'].min(), predicted_df['Latitude'].min()))* 0.2  # 10% 범위
    altitude_margin =  (max(actual_df_trimmed['Altitude (m)'].max(), predicted_df['Altitude (m)'].max()) -min(actual_df_trimmed['Altitude (m)'].min(), predicted_df['Altitude (m)'].min()))* 0.2  # 10% 범위

    # 축 범위 설정 및 간격 조정
    # Set manual axis limits based on expected ranges
    # longitude_range = [min(df_filtered['Longitude'].min(), predicted_df['Longitude'].min()) - 50,
    #                   max(df_filtered['Longitude'].max(), predicted_df['Longitude'].max()) + 50]
    # latitude_range = [min(df_filtered['Latitude'].min(), predicted_df['Latitude'].min()) - 50,
    #                  max(df_filtered['Latitude'].max(), predicted_df['Latitude'].max()) + 50]
    # altitude_range = [min(df_filtered['Altitude (m)'].min(), predicted_df['Altitude (m)'].min()) - 100,
    #                  max(df_filtered['Altitude (m)'].max(), predicted_df['Altitude (m)'].max()) + 100]

    longitude_range = [min(actual_df_trimmed['Longitude'].min(), predicted_df['Longitude'].min())- longitude_margin,
                       max(actual_df_trimmed['Longitude'].max(), predicted_df['Longitude'].max()) + longitude_margin]
    latitude_range = [min(actual_df_trimmed['Latitude'].min(), predicted_df['Latitude'].min()) - latitude_margin,max(actual_df_trimmed['Latitude'].max(), predicted_df['Latitude'].max())+ latitude_margin]
    altitude_range =[min(actual_df_trimmed['Altitude (m)'].min(), predicted_df['Altitude (m)'].min())- altitude_margin,
                      max(actual_df_trimmed['Altitude (m)'].max(), predicted_df['Altitude (m)'].max())+ altitude_margin]


    ax.set_xlim(longitude_range)
    ax.set_ylim(latitude_range)
    ax.set_zlim(altitude_range)

    # 각 축의 눈금 간격 설정
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=10))

    ax.legend()

    # 플롯 저장
    plt.savefig(f'{plots_path}{file_name}_test_plot.png')
    plt.close(fig)

# 전체 학습 및 테스트 MSE 계산
average_train_mse = total_train_mse / total_files
average_test_mse = total_test_mse / total_files

print('--------------------------------------------------------')
print(f"Overall Train MSE: {average_train_mse}")
print(f"Overall Test MSE: {average_test_mse}")

# 최종 학습된 모델 저장
model.save(
    r'C:\Users\kimhyeonju\PycharmProjects\GP_test\GP_final_second\SaveModel\aircraft_SaveModel/GRU_model_v13/look_back=40&forward=25/gru_model_13.h5')

# 표준 출력을 원래대로 복구
sys.stdout.close()
sys.stdout = sys.__stdout__