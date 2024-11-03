# import os
# import sys
# import time
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import load_model, Sequential
# from tensorflow.keras.layers import GRU, Dense
# from GRU_model import create_gru_model
# from tensorflow.keras.optimizers import Adam
# from matplotlib.ticker import MaxNLocator
#
# # 데이터 스케일러 인스턴스 생성
# scaler = MinMaxScaler(feature_range=(0, 1))
#
# # 학습 데이터와 라벨 생성 함수
# def create_dataset(data, look_back, forward_length):
#     X, y = [], []
#     for i in range(len(data) - look_back - forward_length):
#         X.append(data[i:(i + look_back), :])
#         y.append(data[i + look_back + forward_length, :])
#     return np.array(X), np.array(y)
#
# # 파일 폴더 경로 설정
# folder_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/dataset/interpolation_1s'
# file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
#
# # 결과 저장 경로 설정
# predictions_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/test_predictions_new/'
# plots_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/test_plot_new/'
#
# # 디렉토리 생성
# os.makedirs(predictions_path, exist_ok=True)
# os.makedirs(plots_path, exist_ok=True)
#
# # 표준 출력을 파일로 리디렉션
# log_file_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/training_log_new.txt'
# sys.stdout = open(log_file_path, 'w')
#
# # look_back 및 forward_length 설정
# look_back = 10
# forward_length = 5
#
# # 이미 학습된 모델 로드
# # model_path = '/Users/admin/PycharmProjects/GP_test/GP_final/drone/model/GRU_model_new_drone.h5'
# # model = load_model(model_path)
#
# for file_name in file_list:
#     file_path = os.path.join(folder_path, file_name)
#     print('--------------------------------------------------------')
#     print(f"Processing file: {file_name}")
#
#     df = pd.read_csv(file_path)
#     df_filtered = df[['Latitude', 'Longitude', 'Altitude (m)']]
#
#     # 데이터 스케일링
#     df_filtered.loc[:, ['Latitude', 'Longitude', 'Altitude (m)']] = scaler.fit_transform(df_filtered[['Latitude', 'Longitude', 'Altitude (m)']])
#
#     # 학습 데이터와 테스트 데이터 분할
#     X, y = create_dataset(df_filtered.values, look_back, forward_length)
#     split_index = int(len(X) * 0.8)
#     X_train, X_test = X[:split_index], X[split_index:]
#     y_train, y_test = y[:split_index], y[split_index:]
#     # print("X:", X)
#     # print("y:", y)
#
#     # 모델 생성 및 학습
#     model = create_gru_model(input_shape=(look_back, 3))
#     model.compile(optimizer=Adam(), loss='mean_squared_error')
#     model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
#
#     # 학습 데이터의 MSE 계산
#     train_predictions = model.predict(X_train)
#     train_mse = mean_squared_error(y_train, train_predictions)
#     print(f"Train MSE for {file_name}: {train_mse}")
#
#     # 테스트 데이터의 MSE 계산 및 예측
#     start_time = time.time()
#     test_predictions = model.predict(X_test)
#     end_time = time.time()
#     test_mse = mean_squared_error(y_test, test_predictions)
#     prediction_time_per_sample = (end_time - start_time) / len(X_test)
#     print(f"Test MSE for {file_name}: {test_mse}, Prediction Time per Sample: {prediction_time_per_sample:.5f} seconds")
#
#     # 테스트 데이터로 예측된 지점을 원래 스케일로 변환
#     predicted_coordinates = scaler.inverse_transform(test_predictions)
#
#     # 테스트 데이터의 실제 지점을 원래 스케일로 변환
#     actual_coordinates = scaler.inverse_transform(y_test)
#
#     # 예측된 지점을 DataFrame으로 변환
#     predicted_df = pd.DataFrame(predicted_coordinates, columns=['Latitude', 'Longitude', 'Altitude (m)'])
#     actual_df = pd.DataFrame(actual_coordinates, columns=['Latitude', 'Longitude', 'Altitude (m)'])
#
#     # 디버깅을 위한 출력
#     print(f"Actual Data for {file_name}:\n", actual_df.head())
#     print(f"Predicted Data for {file_name}:\n", predicted_df.head())
#
#     # 원본 데이터에서 look_back만큼 제거하여 실제 데이터와 예측 데이터의 시작점을 맞춤
#     # df_filtered_trimmed = df_filtered.iloc[look_back + forward_length:].reset_index(drop=True)
#     # actual_df_trimmed = pd.DataFrame(scaler.inverse_transform(df_filtered_trimmed), columns=['Latitude', 'Longitude', 'Altitude (m)'])
#     test_indices = range(split_index + look_back + forward_length, len(df_filtered))
#     actual_df_trimmed = pd.DataFrame(scaler.inverse_transform(df_filtered.iloc[test_indices]), columns=['Latitude', 'Longitude', 'Altitude (m)'])
#
#     # 예측 결과를 CSV 파일로 저장
#     predicted_df.to_csv(f'{predictions_path}{file_name}_test_predictions_new.csv', index=False)
#
#     # 원본 데이터와 예측 데이터를 비교하는 플롯
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # 플롯 제목 설정
#     plt.title(f"{file_name}, look_back = 10, forward_length = 5")
#
#     # 실제 경로
#     ax.plot(actual_df_trimmed['Longitude'], actual_df_trimmed['Latitude'], actual_df_trimmed['Altitude (m)'], label='Actual Path', color='b')
#
#     # 예측 경로
#     ax.plot(predicted_df['Longitude'], predicted_df['Latitude'], predicted_df['Altitude (m)'], label='Predicted Path', color='r')
#
#     # 축 설정
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
#     ax.set_zlabel('Altitude')
#
#     # 축 범위 설정 및 간격 조정
#     longitude_range = [min(actual_df_trimmed['Longitude'].min(), predicted_df['Longitude'].min()) - 0.001,
#                        max(actual_df_trimmed['Longitude'].max(), predicted_df['Longitude'].max()) + 0.001]
#     latitude_range = [min(actual_df_trimmed['Latitude'].min(), predicted_df['Latitude'].min()) - 0.001,
#                       max(actual_df_trimmed['Latitude'].max(), predicted_df['Latitude'].max()) + 0.001]
#     altitude_range = [min(actual_df_trimmed['Altitude (m)'].min(), predicted_df['Altitude (m)'].min()) - 1,
#                       max(actual_df_trimmed['Altitude (m)'].max(), predicted_df['Altitude (m)'].max()) + 1]
#
#     ax.set_xlim(longitude_range)
#     ax.set_ylim(latitude_range)
#     ax.set_zlim(altitude_range)
#
#     # 각 축의 눈금 간격 설정
#     ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
#     ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
#     ax.zaxis.set_major_locator(MaxNLocator(nbins=10))
#
#     ax.legend()
#
#     # 플롯 저장
#     plt.savefig(f'{plots_path}{file_name}_test_plot.png')
#     plt.close(fig)
#
# # 최종 학습된 모델 저장
# model.save('/Users/admin/PycharmProjects/GP_test/GP_final/drone/model/GRU_model_new_drone.h5')
#
# # 표준 출력을 원래대로 복구
# sys.stdout.close()
# sys.stdout = sys.__stdout__

# import os
# import sys
# import time
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import load_model, Sequential
# from tensorflow.keras.layers import GRU, Dense
# from GRU_model import create_gru_model
# from tensorflow.keras.optimizers import Adam
# from matplotlib.ticker import MaxNLocator
#
# # 데이터 스케일러 인스턴스 생성
# scaler = MinMaxScaler(feature_range=(0, 1))
#
# # 학습 데이터와 라벨 생성 함수
# def create_dataset(data, look_back, forward_length):
#     X, y = [], []
#     for i in range(len(data) - look_back - forward_length):
#         X.append(data[i:(i + look_back), :])
#         y.append(data[i + look_back + forward_length, :])
#     return np.array(X), np.array(y)
#
# # 파일 폴더 경로 설정
# folder_path = '/Users/admin/PycharmProjects/GP_test/data/split_10s'
# file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
#
# # 결과 저장 경로 설정
# predictions_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/aircraft/look_back=50&forward=0/test_predictions_general/'
# plots_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/aircraft/look_back=50&forward=0/test_plot_general/'
#
# # 디렉토리 생성
# os.makedirs(predictions_path, exist_ok=True)
# os.makedirs(plots_path, exist_ok=True)
#
# # 표준 출력을 파일로 리디렉션
# log_file_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/aircraft/look_back=50&forward=0/training_log_general.txt'
# sys.stdout = open(log_file_path, 'w')
#
# # look_back 및 forward_length 설정
# look_back = 50
# forward_length = 0
#
# for file_name in file_list:
#     file_path = os.path.join(folder_path, file_name)
#     print('--------------------------------------------------------')
#     print(f"Processing file: {file_name}")
#
#     df = pd.read_csv(file_path)
#     df_filtered = df[['Latitude', 'Longitude', 'Altitude (m)']]
#
#     # 데이터 스케일링
#     df_filtered.loc[:, ['Latitude', 'Longitude', 'Altitude (m)']] = scaler.fit_transform(df_filtered[['Latitude', 'Longitude', 'Altitude (m)']])
#
#     # 학습 데이터와 테스트 데이터 분할
#     X, y = create_dataset(df_filtered.values, look_back, forward_length)
#     split_index = int(len(X) * 0.8)
#     X_train, X_test = X[:split_index], X[split_index:]
#     y_train, y_test = y[:split_index], y[split_index:]
#     print("X:", X)
#     print("y:", y)
#
#     # 모델 생성 및 학습
#     model = create_gru_model(input_shape=(look_back, 3))
#     model.compile(optimizer=Adam(), loss='mean_squared_error')
#     model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
#
#     # 학습 데이터의 MSE 계산
#     train_predictions = model.predict(X_train)
#     train_mse = mean_squared_error(y_train, train_predictions)
#     print(f"Train MSE for {file_name}: {train_mse}")
#
#     # 테스트 데이터의 MSE 계산 및 예측
#     start_time = time.time()
#     test_predictions = model.predict(X_test)
#     end_time = time.time()
#     test_mse = mean_squared_error(y_test, test_predictions)
#     prediction_time_per_sample = (end_time - start_time) / len(X_test)
#     print(f"Test MSE for {file_name}: {test_mse}, Prediction Time per Sample: {prediction_time_per_sample:.5f} seconds")
#
#     # 테스트 데이터로 예측된 지점을 원래 스케일로 변환
#     predicted_coordinates = scaler.inverse_transform(test_predictions)
#
#     # 테스트 데이터의 실제 지점을 원래 스케일로 변환
#     actual_coordinates = scaler.inverse_transform(y_test)
#
#     # 예측된 지점을 DataFrame으로 변환
#     predicted_df = pd.DataFrame(predicted_coordinates, columns=['Latitude', 'Longitude', 'Altitude (m)'])
#     actual_df = pd.DataFrame(actual_coordinates, columns=['Latitude', 'Longitude', 'Altitude (m)'])
#
#     # 디버깅을 위한 출력
#     print(f"Actual Data for {file_name}:\n", actual_df.head())
#     print(f"Predicted Data for {file_name}:\n", predicted_df.head())
#
#     # 원본 데이터에서 look_back만큼 제거하여 실제 데이터와 예측 데이터의 시작점을 맞춤
#     df_filtered_trimmed = df_filtered.iloc[look_back + forward_length:].reset_index(drop=True)
#     actual_df_trimmed = pd.DataFrame(scaler.inverse_transform(df_filtered_trimmed), columns=['Latitude', 'Longitude', 'Altitude (m)'])
#
#     # 예측 결과를 CSV 파일로 저장
#     predicted_df.to_csv(f'{predictions_path}{file_name}_test_predictions.csv', index=False)
#
#     # 원본 데이터와 예측 데이터를 비교하는 플롯
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # 플롯 제목 설정
#     plt.title(f"{file_name}, look_back = {look_back}, forward_length = {forward_length}")
#
#     # 실제 경로
#     ax.plot(actual_df_trimmed['Longitude'], actual_df_trimmed['Latitude'], actual_df_trimmed['Altitude (m)'], label='Actual Path', color='b')
#
#     # 예측 경로
#     ax.plot(predicted_df['Longitude'], predicted_df['Latitude'], predicted_df['Altitude (m)'], label='Predicted Path', color='r')
#
#     # 축 설정
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
#     ax.set_zlabel('Altitude')
#
#     # 축 범위 설정 및 간격 조정
#     longitude_range = [min(actual_df_trimmed['Longitude'].min(), predicted_df['Longitude'].min()) - 1,
#                        max(actual_df_trimmed['Longitude'].max(), predicted_df['Longitude'].max()) + 1]
#     latitude_range = [min(actual_df_trimmed['Latitude'].min(), predicted_df['Latitude'].min()) - 1,
#                       max(actual_df_trimmed['Latitude'].max(), predicted_df['Latitude'].max()) + 1]
#     altitude_range = [min(actual_df_trimmed['Altitude (m)'].min(), predicted_df['Altitude (m)'].min()) - 1000,
#                       max(actual_df_trimmed['Altitude (m)'].max(), predicted_df['Altitude (m)'].max()) + 1000]
#
#     ax.set_xlim(longitude_range)
#     ax.set_ylim(latitude_range)
#     ax.set_zlim(altitude_range)
#
#     # 각 축의 눈금 간격 설정
#     ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
#     ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
#     ax.zaxis.set_major_locator(MaxNLocator(nbins=10))
#
#     ax.legend()
#
#     # 플롯 저장
#     plt.savefig(f'{plots_path}{file_name}_test_plot.png')
#     plt.close(fig)
#
# # 최종 학습된 모델 저장
# model.save('/Users/admin/PycharmProjects/GP_test/GP_final/aircraft/model/look_back=50&forward=0/GRU_model_general.h5')
#
# # 표준 출력을 원래대로 복구
# sys.stdout.close()
# sys.stdout = sys.__stdout__
# import os
# import sys
# import time
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import load_model, Sequential
# from tensorflow.keras.layers import GRU, Dense
# from GRU_model import create_gru_model
# from tensorflow.keras.optimizers import Adam
# from matplotlib.ticker import MaxNLocator
# from tensorflow.keras.utils import plot_model
#
#
# # 데이터 스케일러 인스턴스 생성
# scaler = MinMaxScaler(feature_range=(0, 1))
#
# # 학습 데이터와 라벨 생성 함수
# def create_dataset(data, look_back, forward_length):
#     X, y = [], []
#     for i in range(len(data) - look_back - forward_length):
#         X.append(data[i:(i + look_back), :])
#         y.append(data[i + look_back + forward_length, :])
#     return np.array(X), np.array(y)
#
# # 파일 폴더 경로 설정
# folder_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/dataset/interpolation_1s'
# file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
#
# # 결과 저장 경로 설정
# predictions_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/look_back=10&forward=0/test_predictions_general/'
# plots_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/look_back=10&forward=0/test_plot_general/'
#
# # 디렉토리 생성
# os.makedirs(predictions_path, exist_ok=True)
# os.makedirs(plots_path, exist_ok=True)
#
# # 표준 출력을 파일로 리디렉션
# log_file_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/look_back=10&forward=0/training_log_general.txt'
# sys.stdout = open(log_file_path, 'w')
#
# # look_back 및 forward_length 설정
# look_back = 10
# forward_length = 0
#
# # 전체 MSE 저장 변수 초기화
# total_train_mse = 0
# total_test_mse = 0
# total_files = len(file_list)
#
# for file_name in file_list:
#     file_path = os.path.join(folder_path, file_name)
#     print('--------------------------------------------------------')
#     print(f"Processing file: {file_name}")
#
#     df = pd.read_csv(file_path)
#     df_filtered = df[['Latitude', 'Longitude', 'Altitude (m)']]
#
#     # 데이터 스케일링
#     df_filtered.loc[:, ['Latitude', 'Longitude', 'Altitude (m)']] = scaler.fit_transform(df_filtered[['Latitude', 'Longitude', 'Altitude (m)']])
#
#     # 학습 데이터와 테스트 데이터 분할
#     X, y = create_dataset(df_filtered.values, look_back, forward_length)
#     split_index = int(len(X) * 0.8)
#     X_train, X_test = X[:split_index], X[split_index:]
#     y_train, y_test = y[:split_index], y[split_index:]
#     print("X:", X)
#     print("y:", y)
#
#     # 모델 생성 및 학습
#     model = create_gru_model(input_shape=(look_back, 3))
#     model.compile(optimizer=Adam(), loss='mean_squared_error')
#     model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
#
#     # 학습 데이터의 MSE 계산
#     train_predictions = model.predict(X_train)
#     train_mse = mean_squared_error(y_train, train_predictions)
#     print(f"Train MSE for {file_name}: {train_mse}")
#
#     # 테스트 데이터의 MSE 계산 및 예측
#     start_time = time.time()
#     test_predictions = model.predict(X_test)
#     end_time = time.time()
#     test_mse = mean_squared_error(y_test, test_predictions)
#     prediction_time_per_sample = (end_time - start_time) / len(X_test)
#     print(f"Test MSE for {file_name}: {test_mse}, Prediction Time per Sample: {prediction_time_per_sample:.5f} seconds")
#
#     # MSE 누적
#     total_train_mse += train_mse
#     total_test_mse += test_mse
#
#     # 테스트 데이터로 예측된 지점을 원래 스케일로 변환
#     predicted_coordinates = scaler.inverse_transform(test_predictions)
#
#     # 테스트 데이터의 실제 지점을 원래 스케일로 변환
#     actual_coordinates = scaler.inverse_transform(y_test)
#
#     # 예측된 지점을 DataFrame으로 변환
#     predicted_df = pd.DataFrame(predicted_coordinates, columns=['Latitude', 'Longitude', 'Altitude (m)'])
#     actual_df = pd.DataFrame(actual_coordinates, columns=['Latitude', 'Longitude', 'Altitude (m)'])
#
#     # 디버깅을 위한 출력
#     print(f"Actual Data for {file_name}:\n", actual_df.head())
#     print(f"Predicted Data for {file_name}:\n", predicted_df.head())
#
#     # 원본 데이터에서 look_back만큼 제거하여 실제 데이터와 예측 데이터의 시작점을 맞춤
#     df_filtered_trimmed = df_filtered.iloc[look_back + forward_length:].reset_index(drop=True)
#     actual_df_trimmed = pd.DataFrame(scaler.inverse_transform(df_filtered_trimmed), columns=['Latitude', 'Longitude', 'Altitude (m)'])
#
#     # 예측 결과를 CSV 파일로 저장
#     predicted_df.to_csv(f'{predictions_path}{file_name}_test_predictions.csv', index=False)
#
#     # 원본 데이터와 예측 데이터를 비교하는 플롯
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # 플롯 제목 설정
#     plt.title(f"{file_name}, look_back = {look_back}, forward_length = {forward_length}")
#
#     # 원본 데이터 (검정색)
#     ax.plot(df_filtered['Longitude'], df_filtered['Latitude'], df_filtered['Altitude (m)'], label='Original Path', color='k', alpha=0.3)
#
#     # 학습 경로 (파란색)
#     ax.plot(actual_df_trimmed['Longitude'].iloc[:split_index], actual_df_trimmed['Latitude'].iloc[:split_index], actual_df_trimmed['Altitude (m)'].iloc[:split_index], label='Train Path', color='b')
#
#     # 테스트 경로 (빨간색)
#     ax.plot(actual_df_trimmed['Longitude'].iloc[split_index:], actual_df_trimmed['Latitude'].iloc[split_index:], actual_df_trimmed['Altitude (m)'].iloc[split_index:], label='Test Path', color='r')
#
#     # 예측 경로 (녹색)
#     ax.plot(predicted_df['Longitude'], predicted_df['Latitude'], predicted_df['Altitude (m)'], label='Predicted Path', color='g')
#
#     # 축 설정
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
#     ax.set_zlabel('Altitude')
#
#     # 축 범위 설정 및 간격 조정
#     longitude_range = [min(actual_df_trimmed['Longitude'].min(), predicted_df['Longitude'].min()) - 0.001,
#                        max(actual_df_trimmed['Longitude'].max(), predicted_df['Longitude'].max()) + 0.001]
#     latitude_range = [min(actual_df_trimmed['Latitude'].min(), predicted_df['Latitude'].min()) - 0.001,
#                       max(actual_df_trimmed['Latitude'].max(), predicted_df['Latitude'].max()) + 0.001]
#     altitude_range = [min(actual_df_trimmed['Altitude (m)'].min(), predicted_df['Altitude (m)'].min()) - 1,
#                       max(actual_df_trimmed['Altitude (m)'].max(), predicted_df['Altitude (m)'].max()) + 1]
#
#     ax.set_xlim(longitude_range)
#     ax.set_ylim(latitude_range)
#     ax.set_zlim(altitude_range)
#
#     # 각 축의 눈금 간격 설정
#     ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
#     ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
#     ax.zaxis.set_major_locator(MaxNLocator(nbins=10))
#
#     ax.legend()
#
#     # 플롯 저장
#     plt.savefig(f'{plots_path}{file_name}_test_plot.png')
#     plt.close(fig)
#
# # 전체 학습 및 테스트 MSE 계산
# average_train_mse = total_train_mse / total_files
# average_test_mse = total_test_mse / total_files
#
# print('--------------------------------------------------------')
# print(f"Overall Train MSE: {average_train_mse}")
# print(f"Overall Test MSE: {average_test_mse}")
#
# # 최종 학습된 모델 저장
# model.save('/Users/admin/PycharmProjects/GP_test/GP_final/drone/model/look_back=10&forward=0/GRU_model_general.h5')
#
# # 표준 출력을 원래대로 복구
# sys.stdout.close()
# sys.stdout = sys.__stdout__


import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import GRU, Dense
from GRU_model import create_gru_model
from tensorflow.keras.optimizers import Adam
from matplotlib.ticker import MaxNLocator

# 데이터 스케일러 인스턴스 생성
scaler = MinMaxScaler(feature_range=(0, 1))

# 학습 데이터와 라벨 생성 함수
def create_dataset(data, look_back, forward_length):
    X, y = [], []
    for i in range(len(data) - look_back - forward_length):
        X.append(data[i:(i + look_back), :])
        y.append(data[i + look_back + forward_length, :])
    return np.array(X), np.array(y)

# 파일 폴더 경로 설정
folder_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/dataset/interpolation_1s'
file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# 결과 저장 경로 설정
predictions_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/look_back=30&forward=15/test_predictions_general/'
plots_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/look_back=30&forward=15/test_plot_general/'

# 디렉토리 생성
os.makedirs(predictions_path, exist_ok=True)
os.makedirs(plots_path, exist_ok=True)

# 표준 출력을 파일로 리디렉션
log_file_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/look_back=30&forward=15/training_log_general.txt'
sys.stdout = open(log_file_path, 'w')

# look_back 및 forward_length 설정
look_back = 30
forward_length = 15

# 전체 MSE 저장 변수 초기화
total_train_mse = 0
total_test_mse = 0
total_files = len(file_list)

for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    print('--------------------------------------------------------')
    print(f"Processing file: {file_name}")

    df = pd.read_csv(file_path)
    df_filtered = df[['Latitude', 'Longitude', 'Altitude (m)']]

    # 데이터 스케일링
    df_filtered.loc[:, ['Latitude', 'Longitude', 'Altitude (m)']] = scaler.fit_transform(df_filtered[['Latitude', 'Longitude', 'Altitude (m)']])

    # 학습 데이터와 테스트 데이터 분할
    X, y = create_dataset(df_filtered.values, look_back, forward_length)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    print("X:", X)
    print("y:", y)

    # 모델 생성 및 학습
    model = create_gru_model(input_shape=(look_back, 3))
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # 학습 데이터의 MSE 계산
    train_predictions = model.predict(X_train)
    train_mse = mean_squared_error(y_train, train_predictions)
    print(f"Train MSE for {file_name}: {train_mse}")

    # 테스트 데이터의 MSE 계산 및 예측
    start_time = time.time()
    test_predictions = model.predict(X_test)
    end_time = time.time()
    test_mse = mean_squared_error(y_test, test_predictions)
    prediction_time_per_sample = (end_time - start_time) / len(X_test)
    print(f"Test MSE for {file_name}: {test_mse}, Prediction Time per Sample: {prediction_time_per_sample:.5f} seconds")

    # MSE 누적
    total_train_mse += train_mse
    total_test_mse += test_mse

    # 테스트 데이터로 예측된 지점을 원래 스케일로 변환
    predicted_coordinates = scaler.inverse_transform(test_predictions)

    # 테스트 데이터의 실제 지점을 원래 스케일로 변환
    actual_coordinates = scaler.inverse_transform(y_test)

    # 예측된 지점을 DataFrame으로 변환
    predicted_df = pd.DataFrame(predicted_coordinates, columns=['Latitude', 'Longitude', 'Altitude (m)'])
    actual_df = pd.DataFrame(actual_coordinates, columns=['Latitude', 'Longitude', 'Altitude (m)'])

    # 디버깅을 위한 출력
    print(f"Actual Data for {file_name}:\n", actual_df.head())
    print(f"Predicted Data for {file_name}:\n", predicted_df.head())

    # 원본 데이터에서 look_back만큼 제거하여 실제 데이터와 예측 데이터의 시작점을 맞춤
    df_filtered_trimmed = df_filtered.iloc[look_back + forward_length:].reset_index(drop=True)
    actual_df_trimmed = pd.DataFrame(scaler.inverse_transform(df_filtered_trimmed), columns=['Latitude', 'Longitude', 'Altitude (m)'])

    # 예측 결과를 CSV 파일로 저장
    predicted_df.to_csv(f'{predictions_path}{file_name}_test_predictions.csv', index=False)

    # 원본 데이터와 예측 데이터를 비교하는 플롯
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 플롯 제목 설정
    plt.title(f"{file_name}, look_back = {look_back}, forward_length = {forward_length}")

    # 원본 데이터 (검정색)
    ax.plot(df_filtered['Longitude'], df_filtered['Latitude'], df_filtered['Altitude (m)'], label='Original Path', color='k', alpha=0.3)

    # 학습 경로 (파란색)
    ax.plot(actual_df_trimmed['Longitude'].iloc[:split_index], actual_df_trimmed['Latitude'].iloc[:split_index], actual_df_trimmed['Altitude (m)'].iloc[:split_index], label='Train Path', color='b')

    # 테스트 경로 (빨간색)
    ax.plot(actual_df_trimmed['Longitude'].iloc[split_index:], actual_df_trimmed['Latitude'].iloc[split_index:], actual_df_trimmed['Altitude (m)'].iloc[split_index:], label='Test Path', color='r')

    # 예측 경로 (녹색)
    ax.plot(predicted_df['Longitude'], predicted_df['Latitude'], predicted_df['Altitude (m)'], label='Predicted Path', color='g')

    # 축 설정
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude')

    # 축 범위 설정 및 간격 조정
    longitude_range = [min(actual_df_trimmed['Longitude'].min(), predicted_df['Longitude'].min()) - 0.001,
                        max(actual_df_trimmed['Longitude'].max(), predicted_df['Longitude'].max()) + 0.001]
    latitude_range = [min(actual_df_trimmed['Latitude'].min(), predicted_df['Latitude'].min()) - 0.001,
                        max(actual_df_trimmed['Latitude'].max(), predicted_df['Latitude'].max()) + 0.001]
    altitude_range = [min(actual_df_trimmed['Altitude (m)'].min(), predicted_df['Altitude (m)'].min()) - 1,
                        max(actual_df_trimmed['Altitude (m)'].max(), predicted_df['Altitude (m)'].max()) + 1]

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
model.save('/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/model/look_back=30&forward=15/GRU_model.h5')

# 표준 출력을 원래대로 복구
sys.stdout.close()
sys.stdout = sys.__stdout__
