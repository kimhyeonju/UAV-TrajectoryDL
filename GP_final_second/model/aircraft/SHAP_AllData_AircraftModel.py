# import shap
# from tensorflow.keras.models import load_model
# import numpy as np
# import pandas as pd
# import os
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
#
# # 저장된 모델 불러오기
# model = load_model('/Users/admin/PycharmProjects/GP_test/GP_final_second/SaveModel/drone_SaveModel/GRU_model_v3/look_back=10&forward=0/gru_model_3.h5')
#
# # 데이터 스케일러 설정 (모델 학습에 사용된 스케일러와 동일해야 함)
# scaler = MinMaxScaler(feature_range=(0, 1))
#
# # 폴더 경로 설정 (CSV 파일이 있는 경로)
# folder_path = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Results/RE_SplineInterpolation_1s'
# file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]  # CSV 파일만 리스트에 포함
#
# # SHAP 값을 누적할 리스트
# shap_values_latitude_total = []
# shap_values_longitude_total = []
# shap_values_altitude_total = []
#
# # 입력 데이터 누적을 위한 리스트
# reshaped_X_test_total = []
#
# # look_back 설정
# look_back = 10
#
# # 전체 파일에 대해 데이터를 처리하고 SHAP 값 누적
# for file_name in file_list:
#     print(f"Processing file: {file_name}")
#
#     # CSV 파일 읽기
#     file_path = os.path.join(folder_path, file_name)
#     df = pd.read_csv(file_path)
#
#     # 필터링된 데이터 준비 (Latitude, Longitude, Altitude, Speed, wind_speed, wind_direction)
#     df_filtered = df[['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)', 'wind_speed', 'wind_direction']]
#
#     # 데이터를 스케일링 (MinMaxScaler로 스케일)
#     df_filtered_scaled = scaler.fit_transform(df_filtered.values)
#
#     # SHAP을 위한 데이터 준비
#     X_test_shap = np.array([df_filtered_scaled[i:i + look_back, :] for i in range(len(df_filtered_scaled) - look_back)])
#
#     # GradientExplainer를 사용하여 SHAP 값 계산
#     explainer = shap.GradientExplainer(model, X_test_shap[:100])
#
#     # SHAP 값 계산 (각 파일에 대해 슬라이딩 윈도우 적용)
#     shap_values_total = []
#     for i in range(90):  # 1부터 100까지 10개씩 슬라이딩 윈도우로 계산
#         shap_values = explainer.shap_values(X_test_shap[i:i + 10])  # 10개의 타임스텝에 대해 SHAP 값 계산
#         shap_values_total.append(shap_values)
#
#     # 각 파일의 SHAP 값을 합산
#     shap_values_latitude = np.mean([shap_value[0][:, :, 0] for shap_value in shap_values_total], axis=0)
#     shap_values_longitude = np.mean([shap_value[0][:, :, 1] for shap_value in shap_values_total], axis=0)
#     shap_values_altitude = np.mean([shap_value[0][:, :, 2] for shap_value in shap_values_total], axis=0)
#
#     # 누적 리스트에 추가
#     shap_values_latitude_total.append(shap_values_latitude)
#     shap_values_longitude_total.append(shap_values_longitude)
#     shap_values_altitude_total.append(shap_values_altitude)
#
#     # 입력 데이터도 동일한 차원으로 맞춤
#     reshaped_X_test = np.mean([X_test_shap[i:i+10, 0, :] for i in range(90)], axis=0)
#     reshaped_X_test_total.append(reshaped_X_test)
#
# # SHAP 값을 평균내기 위한 준비 (전체 파일에 대한 SHAP 값 평균)
# shap_values_latitude_mean = np.mean(shap_values_latitude_total, axis=0)
# shap_values_longitude_mean = np.mean(shap_values_longitude_total, axis=0)
# shap_values_altitude_mean = np.mean(shap_values_altitude_total, axis=0)
#
# # 입력 데이터도 평균값 계산
# reshaped_X_test_mean = np.mean(reshaped_X_test_total, axis=0)
#
# # SHAP summary plot 출력 (예: Altitude에 대한 설명)
# print(f"SHAP Values Altitude shape: {shap_values_altitude_mean.shape}")
# print(f"reshaped_X_test shape: {reshaped_X_test_mean.shape}")
#
# # 시각적으로 더 나은 SHAP summary plot을 생성
# plt.figure(figsize=(10, 7))  # 플롯 크기를 더 크게 설정하여 시각적으로 더 나은 가독성 제공
# plt.title("SHAP Summary Plot for Altitude (Averaged Across All Files)")
#
# shap.summary_plot(
#     shap_values_altitude_mean,
#     reshaped_X_test_mean,
#     feature_names=['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)', 'wind_speed', 'wind_direction'],
#     plot_type='violin',
#     show=True
# )
#
# plt.show()  # 플롯을 화면에 출력
import shap
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 저장된 모델 불러오기
model = load_model('/Users/admin/PycharmProjects/GP_test/GP_final_second/SaveModel/aircraft_SaveModel/GRU_model_v7/look_back=10&forward=0/gru_model_7.h5')

# 데이터 스케일러 설정 (모델 학습에 사용된 스케일러와 동일해야 함)
scaler = MinMaxScaler(feature_range=(0, 1))

# 폴더 경로 설정 (CSV 파일이 있는 경로)
folder_path = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Aircraft_Results/'
file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]  # CSV 파일만 리스트에 포함

# SHAP 값을 누적할 리스트
shap_values_latitude_total = []
shap_values_longitude_total = []
shap_values_altitude_total = []

# 입력 데이터 누적을 위한 리스트
reshaped_X_test_total = []

# look_back 설정
look_back = 10

# 전체 파일에 대해 데이터를 처리하고 SHAP 값 누적
for file_name in file_list:
    print(f"Processing file: {file_name}")

    # CSV 파일 읽기
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)

    # 필터링된 데이터 준비 (Latitude, Longitude, Altitude, Speed, wind_speed, wind_direction)
    df_filtered = df[['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)']]

    # 데이터를 스케일링 (MinMaxScaler로 스케일)
    df_filtered_scaled = scaler.fit_transform(df_filtered.values)

    # SHAP을 위한 데이터 준비
    X_test_shap = np.array([df_filtered_scaled[i:i + look_back, :] for i in range(len(df_filtered_scaled) - look_back)])

    # GradientExplainer를 사용하여 SHAP 값 계산
    explainer = shap.GradientExplainer(model, X_test_shap[:100])

    # SHAP 값 계산 (각 파일에 대해 슬라이딩 윈도우 적용)
    shap_values_total = []
    for i in range(90):  # 1부터 100까지 10개씩 슬라이딩 윈도우로 계산
        shap_values = explainer.shap_values(X_test_shap[i:i + 10])  # 10개의 타임스텝에 대해 SHAP 값 계산
        shap_values_total.append(shap_values)

    # 각 파일의 SHAP 값을 합산
    shap_values_latitude = np.mean([shap_value[0][:, :, 0] for shap_value in shap_values_total], axis=0)
    shap_values_longitude = np.mean([shap_value[0][:, :, 1] for shap_value in shap_values_total], axis=0)
    shap_values_altitude = np.mean([shap_value[0][:, :, 2] for shap_value in shap_values_total], axis=0)

    # 누적 리스트에 추가
    shap_values_latitude_total.append(shap_values_latitude)
    shap_values_longitude_total.append(shap_values_longitude)
    shap_values_altitude_total.append(shap_values_altitude)

    # 입력 데이터도 동일한 차원으로 맞춤
    reshaped_X_test = np.mean([X_test_shap[i:i+10, 0, :] for i in range(90)], axis=0)
    reshaped_X_test_total.append(reshaped_X_test)

# SHAP 값을 평균내기 위한 준비 (전체 파일에 대한 SHAP 값 평균)
shap_values_latitude_mean = np.mean(shap_values_latitude_total, axis=0)
shap_values_longitude_mean = np.mean(shap_values_longitude_total, axis=0)
shap_values_altitude_mean = np.mean(shap_values_altitude_total, axis=0)

# 입력 데이터도 평균값 계산
reshaped_X_test_mean = np.mean(reshaped_X_test_total, axis=0)

# 각각의 SHAP summary plot을 별도로 생성

# SHAP summary plot for Latitude
plt.figure(figsize=(10, 7))
plt.title("SHAP Summary Plot for Latitude")
shap.summary_plot(
    shap_values_latitude_mean,
    reshaped_X_test_mean,
    feature_names=['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)'],
    plot_type='violin',
    show=True
)

# SHAP summary plot for Longitude
plt.figure(figsize=(10, 7))
plt.title("SHAP Summary Plot for Longitude")
shap.summary_plot(
    shap_values_longitude_mean,
    reshaped_X_test_mean,
    feature_names=['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)'],
    plot_type='violin',
    show=True
)

# SHAP summary plot for Altitude
plt.figure(figsize=(10, 7))
plt.title("SHAP Summary Plot for Altitude")
shap.summary_plot(
    shap_values_altitude_mean,
    reshaped_X_test_mean,
    feature_names=['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)'],
    plot_type='violin',
    show=True
)