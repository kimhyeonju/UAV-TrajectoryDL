import pandas as pd
import matplotlib.pyplot as plt

# 원본 파일 경로 설정
original_data_path = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Aircraft_Results/Interpolated_LLH_FFlight DTA747 Luanda to Sao Paulo.csv'  # 원본 파일 경로
predicted_data_path = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/aircraft/GRU_model_v10/look_back=10&forward=0/test_predictions_general/Interpolated_LLH_FFlight DTA747 Luanda to Sao Paulo.csv_test_predictions.csv'  # 예측 데이터 경로
# CSV 파일 불러오기
df_original = pd.read_csv(original_data_path)
df_predicted = pd.read_csv(predicted_data_path)

# 데이터를 0.8:0.2로 나누기
split_index = int(len(df_original) * 0.8)
train_data = df_original[:split_index]  # 학습 데이터
test_data = df_original[split_index:]   # 테스트 데이터

# 예측 데이터의 인덱스와 시간 데이터 맞추기
df_predicted['Time'] = test_data['Time'].values[:len(df_predicted)]  # 원본 테스트 데이터의 시간을 예측 데이터에 추가

# 고도 데이터 시각화
plt.figure(figsize=(10, 6))

# 원본 테스트 데이터 그래프 (0.2로 나눈 부분)
plt.plot(test_data['Time'], test_data['Altitude (m)'], label='Actual Altitude', color='blue', linestyle='-', alpha=0.7)

# 예측 데이터 그래프 (예측 경로에 시간 추가됨)
plt.plot(df_predicted['Time'], df_predicted['Altitude (m)'], label='Predicted Altitude', color='red', linestyle='--', alpha=0.7)

# 그래프 설정
plt.title('Actual vs Predicted Altitude (Test Data)')
plt.xlabel('Time')
plt.ylabel('Altitude (m)')
plt.legend()

# 그래프 출력
plt.show()
