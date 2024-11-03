# import os
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import GRU, Dense, Input
# from tensorflow.keras.optimizers import Adam
# import matplotlib.pyplot as plt
# import time
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
# # GRU 모델 생성 함수
# def create_gru_model(input_shape, units_list):
#     model = Sequential()
#     model.add(Input(shape=input_shape))
#     for units in units_list[:-1]:
#         model.add(GRU(units, return_sequences=True))
#     model.add(GRU(units_list[-1]))
#     model.add(Dense(3))
#     model.compile(optimizer=Adam(), loss='mean_squared_error')
#     return model
#
# # 파일 폴더 경로 설정
# folder_path = '/Users/admin/PycharmProjects/GP_test/data/test'
# file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
#
# # 하이퍼파라미터 탐색
# hidden_layers_options = [1, 2, 3]  # 레이어 수
# hidden_units_options = [16, 32, 64, 128]  # 각 레이어의 노드 수
#
# # look_back 및 forward_length 설정
# look_back = 30
# forward_length = 0
#
# # 학습 기록 저장 변수 초기화
# history_layers_dict = {}
# history_units_dict = {}
# prediction_times_layers = {}
# prediction_times_units = {}
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
#     X_train, X_val = X[:split_index], X[split_index:]
#     y_train, y_val = y[:split_index], y[split_index:]
#
#     # Hidden Layer 수만 변경
#     for num_layers in hidden_layers_options:
#         units_list = [64] * num_layers  # units 고정
#         model = create_gru_model(input_shape=(look_back, 3), units_list=units_list)
#         history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_val, y_val), verbose=0)
#         key = f'layers_{num_layers}_units_64'
#         if key not in history_layers_dict:
#             history_units_dict[key] = np.array(history.history['val_loss'])
#         else:
#             history_layers_dict[key] = np.add(history_layers_dict[key], history.history['val_loss'])
#
#         # 예측 시간 측정
#         start_time = time.time()
#         model.predict(X_val)
#         end_time = time.time()
#         prediction_time = (end_time - start_time) / len(X_val)
#         if key not in prediction_times_layers:
#             prediction_times_layers[key] = prediction_time
#         else:
#             prediction_times_layers[key] = (prediction_times_layers[key] + prediction_time) / 2
#
#     # Units 수만 변경
#     for units in hidden_units_options:
#         units_list = [units] * 2  # layers 고정
#         model = create_gru_model(input_shape=(look_back, 3), units_list=units_list)
#         history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_val, y_val), verbose=0)
#         key = f'layers_2_units_{units}'
#         if key not in history_units_dict:
#             history_layers_dict[key] = np.array(history.history['val_loss'])  # numpy 배열로 변환
#         else:
#             history_units_dict[key] = np.add(history_units_dict[key], history.history['val_loss'])
#
#         # 예측 시간 측정
#         start_time = time.time()
#         model.predict(X_val)
#         end_time = time.time()
#         prediction_time = (end_time - start_time) / len(X_val)
#         if key not in prediction_times_layers:
#             prediction_times_layers[key] = prediction_time
#         else:
#             prediction_times_layers[key] = (prediction_times_layers[key] + prediction_time) / 2
#
# # 에포크별 평균 validation loss 계산
# num_files = len(file_list)
# for key in history_layers_dict:
#     history_layers_dict[key] = history_layers_dict[key] / num_files
#
# for key in history_units_dict:
#     history_units_dict[key] = history_units_dict[key] / num_files
#
# # Hidden Layer 수 변경에 따른 validation loss 시각화
# plt.figure(figsize=(14, 10))
# for num_layers in hidden_layers_options:
#     key = f'Validation Loss layers {num_layers}'
#     if key in history_layers_dict:
#         # plt.plot(range(1, len(history_layers_dict[key]) + 1), history_layers_dict[key], label=key)
#         plt.plot(range(1, len(history_layers_dict[key]) + 1), history_layers_dict[key],
#                  label=f'Layers {num_layers} (Units fixed at 64)')
#
# plt.xlabel('Epochs')
# plt.ylabel('Validation Loss')
# plt.title('Validation Loss for Number of Layers (Units fixed at 64)')
# plt.legend()
# plt.grid(True)
# plt.savefig('validation_loss_layers.png')
# plt.show()
#
# # Units 수 변경에 따른 validation loss 시각화
# plt.figure(figsize=(14, 10))
# for units in hidden_units_options:
#     key = f'Validation Loss units {units}'
#     if key in history_units_dict:
#         # plt.plot(range(1, len(history_units_dict[key]) + 1), history_units_dict[key], label=key)
#         plt.plot(range(1, len(history_units_dict[key]) + 1), history_units_dict[key],
#                  label=f'Layers fixed at 2 (Units {units})')
#
# plt.xlabel('Epochs')
# plt.ylabel('Validation Loss')
# plt.title('Validation Loss for Number of Units (Layers fixed at 2)')
# plt.legend()
# plt.grid(True)
# plt.savefig('validation_loss_units.png')
# plt.show()
#
# # 예측 시간 출력
# print("Prediction Times for Different Number of Layers (Units fixed at 64):")
# for key, time in prediction_times_layers.items():
#     print(f"{key}: {time:.5f} seconds per sample")
#
# print("\nPrediction Times for Different Number of Units (Layers fixed at 2):")
# for key, time in prediction_times_units.items():
#     print(f"{key}: {time:.5f} seconds per sample")
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import time

# 데이터 스케일러 인스턴스 생성
scaler = MinMaxScaler(feature_range=(0, 1))

# 학습 데이터와 라벨 생성 함수
def create_dataset(data, look_back, forward_length):
    X, y = [], []
    for i in range(len(data) - look_back - forward_length):
        X.append(data[i:(i + look_back), :])
        y.append(data[i + look_back + forward_length, :])
    return np.array(X), np.array(y)

# GRU 모델 생성 함수
def create_gru_model(input_shape, units_list):
    model = Sequential()
    model.add(Input(shape=input_shape))
    for units in units_list[:-1]:
        model.add(GRU(units, return_sequences=True))
    model.add(GRU(units_list[-1]))
    model.add(Dense(3))
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

# 파일 폴더 경로 설정
folder_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/dataset/interpolation_1s/test'
file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# 하이퍼파라미터 탐색
hidden_layers_options = [1, 2, 3]  # 레이어 수
hidden_units_options = [16, 32, 64, 128]  # 각 레이어의 노드 수

# look_back 및 forward_length 설정
look_back = 30
forward_length = 10

# 학습 기록 저장 변수 초기화
history_layers_dict = {}
history_units_dict = {}
prediction_times_layers = {}
prediction_times_units = {}

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
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    # Hidden Layer 수만 변경
    for num_layers in hidden_layers_options:
        units_list = [64] * num_layers  # units 고정
        model = create_gru_model(input_shape=(look_back, 3), units_list=units_list)
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=0)
        key = f'layers_{num_layers}_units_64'
        if key not in history_layers_dict:
            history_layers_dict[key] = np.array(history.history['val_loss'])
        else:
            history_layers_dict[key] = np.add(history_layers_dict[key], history.history['val_loss'])

        # 예측 시간 측정
        start_time = time.time()
        model.predict(X_val)
        end_time = time.time()
        prediction_time = (end_time - start_time) / len(X_val)
        if key not in prediction_times_layers:
            prediction_times_layers[key] = prediction_time
        else:
            prediction_times_layers[key] = (prediction_times_layers[key] + prediction_time) / 2

    # Units 수만 변경
    for units in hidden_units_options:
        units_list = [units] * 2  # layers 고정
        model = create_gru_model(input_shape=(look_back, 3), units_list=units_list)
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=0)
        key = f'layers_2_units_{units}'
        if key not in history_units_dict:
            history_units_dict[key] = np.array(history.history['val_loss'])  # numpy 배열로 변환
        else:
            history_units_dict[key] = np.add(history_units_dict[key], history.history['val_loss'])

        # 예측 시간 측정
        start_time = time.time()
        model.predict(X_val)
        end_time = time.time()
        prediction_time = (end_time - start_time) / len(X_val)
        if key not in prediction_times_units:
            prediction_times_units[key] = prediction_time
        else:
            prediction_times_units[key] = (prediction_times_units[key] + prediction_time) / 2

# 에포크별 평균 validation loss 계산
num_files = len(file_list)
for key in history_layers_dict:
    history_layers_dict[key] = history_layers_dict[key] / num_files
    print(history_layers_dict)

for key in history_units_dict:
    history_units_dict[key] = history_units_dict[key] / num_files
    print(history_units_dict)

# Hidden Layer 수 변경에 따른 validation loss 시각화
plt.figure(figsize=(14, 10))
for num_layers in hidden_layers_options:
    key = f'layers_{num_layers}_units_64'
    if key in history_layers_dict:
        plt.plot(range(1, len(history_layers_dict[key]) + 1), history_layers_dict[key], label=f'Layers {num_layers}')

plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
# plt.ylim(0, 0.001)  # y축 범위 설정
plt.title('Validation Loss for Number of Layers (Units fixed at 64)')
plt.legend()
plt.grid(True)
plt.savefig('validation_loss_layers.png')
plt.show()

# Units 수 변경에 따른 validation loss 시각화
plt.figure(figsize=(14, 10))
for units in hidden_units_options:
    key = f'layers_2_units_{units}'
    if key in history_units_dict:
        plt.plot(range(1, len(history_units_dict[key]) + 1), history_units_dict[key], label=f'Units {units}')

plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
# plt.ylim(0, 0.01)  # y축 범위 설정
plt.title('Validation Loss for Number of Units (Layers fixed at 2)')
plt.legend()
plt.grid(True)
plt.savefig('validation_loss_units.png')
plt.show()

# 예측 시간 출력
print("Prediction Times for Different Number of Layers (Units fixed at 64):")
for key, time in prediction_times_layers.items():
    print(f"{key}: {time:.5f} seconds per sample")

print("\nPrediction Times for Different Number of Units (Layers fixed at 2):")
for key, time in prediction_times_units.items():
    print(f"{key}: {time:.5f} seconds per sample")
