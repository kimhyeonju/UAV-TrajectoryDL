# import shap
# from tensorflow.keras.models import load_model
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
#
# # 저장된 모델 불러오기
# model = load_model('/Users/admin/PycharmProjects/GP_test/GP_final_second/SaveModel/drone_SaveModel/GRU_model_v3/look_back=10&forward=0/gru_model_3.h5')
#
# # 데이터 스케일러 설정 (모델 학습에 사용된 스케일러와 동일해야 함)
# scaler = MinMaxScaler(feature_range=(0, 1))
#
# # 예시 데이터 준비 (테스트 데이터)
# file_path = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Results/RE_SplineInterpolation_1s/Inter_IncludeTakeoff_2020-6-24.csv'
# df = pd.read_csv(file_path)
#
# # 필터링된 데이터 준비 (Latitude, Longitude, Altitude, Speed, wind_speed, wind_direction)
# df_filtered = df[['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)', 'wind_speed', 'wind_direction']]
#
# # 데이터를 스케일링 (MinMaxScaler로 스케일)
# df_filtered_scaled = scaler.fit_transform(df_filtered.values)
#
# # SHAP을 위한 데이터 준비 (X_test 데이터로 사용)
# look_back = 10  # look_back 설정에 맞춰 시퀀스 데이터 생성
# X_test_shap = np.array([df_filtered_scaled[i:i+look_back, :] for i in range(len(df_filtered_scaled)-look_back)])
#
# # GradientExplainer를 사용하여 SHAP 값 계산
# explainer = shap.GradientExplainer(model, X_test_shap[:100])  # 모델과 일부 데이터를 참조로 사용
#
# # SHAP 값 계산 (일부 테스트 데이터에 대해)
# shap_values = explainer.shap_values(X_test_shap[100:110])  # 예시로 10개의 테스트 데이터 사용
#
# # 각 출력에 대한 SHAP 값을 선택 (Latitude, Longitude, Altitude에 대해 각각 선택)
# shap_values_latitude = shap_values[0][:, :, 0]  # 첫 번째 출력인 Latitude에 대한 SHAP 값 선택
# shap_values_longitude = shap_values[0][:, :, 1]  # 두 번째 출력인 Longitude에 대한 SHAP 값 선택
# shap_values_altitude = shap_values[0][:, :, 2]  # 세 번째 출력인 Altitude에 대한 SHAP 값 선택
#
# # 입력 데이터도 동일한 차원으로 맞춤 (예시로 첫 번째 타임스텝 선택)
# reshaped_X_test = X_test_shap[100:110, 0, :]  # 타임스텝 중 첫 번째를 선택
#
# # SHAP summary plot 출력 (Latitude에 대한 설명)
# print(f"SHAP Values Latitude shape: {shap_values_latitude.shape}")
# print(f"X_test_shap shape: {reshaped_X_test.shape}")
#
# # 필요한 차원만 사용하여 시각화
# shap.summary_plot(shap_values_latitude, reshaped_X_test, feature_names=['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)', 'wind_speed', 'wind_direction'])
#
import shap
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# 저장된 모델 불러오기
model = load_model('/Users/admin/PycharmProjects/GP_test/GP_final_second/SaveModel/drone_SaveModel/GRU_model_v3/look_back=10&forward=0/gru_model_3.h5')

# 데이터 스케일러 설정 (모델 학습에 사용된 스케일러와 동일해야 함)
scaler = MinMaxScaler(feature_range=(0, 1))

# 예시 데이터 준비 (테스트 데이터)
file_path = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Results/RE_SplineInterpolation_1s/Inter_IncludeTakeoff_2020-6-24.csv'
df = pd.read_csv(file_path)

# 필터링된 데이터 준비 (Latitude, Longitude, Altitude, Speed, wind_speed, wind_direction)
df_filtered = df[['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)', 'wind_speed', 'wind_direction']]

# 데이터를 스케일링 (MinMaxScaler로 스케일)
df_filtered_scaled = scaler.fit_transform(df_filtered.values)

# SHAP을 위한 데이터 준비 (X_test 데이터로 사용)
look_back = 10  # look_back 설정에 맞춰 시퀀스 데이터 생성
X_test_shap = np.array([df_filtered_scaled[i:i + look_back, :] for i in range(len(df_filtered_scaled) - look_back)])

# GradientExplainer를 사용하여 SHAP 값 계산
explainer = shap.GradientExplainer(model, X_test_shap[:100])  # 모델과 일부 데이터를 참조로 사용

# SHAP 값 계산 (100개의 데이터를 구간별로 계산)
shap_values_total = []

for i in range(90):  # 1부터 100까지 10개씩 슬라이딩 윈도우로 계산
    shap_values = explainer.shap_values(X_test_shap[i:i + 10])  # 10개의 타임스텝에 대해 SHAP 값 계산
    shap_values_total.append(shap_values)

# SHAP 값을 평균내기 위한 준비
# 각각의 SHAP 값을 합산 후 평균
shap_values_latitude_total = np.mean([shap_value[0][:, :, 0] for shap_value in shap_values_total], axis=0)
shap_values_longitude_total = np.mean([shap_value[0][:, :, 1] for shap_value in shap_values_total], axis=0)
shap_values_altitude_total = np.mean([shap_value[0][:, :, 2] for shap_value in shap_values_total], axis=0)


# 입력 데이터도 동일한 차원으로 맞춤 (예시로 첫 번째 타임스텝 선택)
reshaped_X_test = np.mean([X_test_shap[i:i+10, 0, :] for i in range(90)], axis=0)

# SHAP summary plot 출력 (Latitude에 대한 설명)
print(f"SHAP Values Altitude shape: {shap_values_latitude_total.shape}")
print(f"X_test_shap shape: {reshaped_X_test.shape}")

# 시각적으로 더 나은 SHAP summary plot을 생성
plt.figure(figsize=(10, 7))  # 플롯 크기를 더 크게 설정하여 시각적으로 더 나은 가독성 제공
plt.title("SHAP Summary Plot for Altitude")

shap.summary_plot(
    shap_values_altitude_total,
    reshaped_X_test,
    feature_names=['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)', 'wind_speed', 'wind_direction'],
    plot_type='violin',  # violin plot을 사용하여 SHAP 값의 분포를 더 명확하게 보여줌
    show=True
)

# ax = plt.gca()
# ax.xaxis.set_major_locator(MaxNLocator(nbins=6))  # 눈금 간격 자동 조정
# plt.xticks(fontsize=8)  # x축 레이블 크기를 줄임


plt.show()  # 플롯을 화면에 출력