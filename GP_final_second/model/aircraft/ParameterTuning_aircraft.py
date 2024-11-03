import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam

# 데이터 스케일러 인스턴스 생성
scaler = MinMaxScaler(feature_range=(0, 1))


# 학습 데이터와 라벨 생성 함수
def create_dataset(data, look_back, forward_length):
    X, y = [], []
    for i in range(len(data) - look_back - forward_length):
        X.append(data[i:(i + look_back), :])
        y.append(data[i + look_back + forward_length,
                 :3])  # 3개의 출력 (Latitude, Longitude, Altitude)
    return np.array(X), np.array(y)


# 모델 생성 함수
def create_model(model_type, input_shape, gru_units=64, bi_gru_units=64, last_layer_units=64, num_layers=2,
                 last_layer_type='GRU'):
    model = Sequential()

    # GRU 모델 구성
    if model_type == 'GRU':
        for i in range(num_layers):
            model.add(GRU(gru_units, return_sequences=(i < num_layers - 1), input_shape=input_shape))

    # BiGRU 모델 구성
    elif model_type == 'BiGRU':
        for i in range(num_layers):
            model.add(Bidirectional(GRU(bi_gru_units, return_sequences=(i < num_layers - 1), input_shape=input_shape)))

    # GRU + BiGRU 혼합 모델 구성
    elif model_type == 'GRU+BiGRU':
        # 첫 번째 레이어는 GRU
        model.add(GRU(gru_units, return_sequences=True, input_shape=input_shape))
        # 두 번째 레이어는 BiGRU
        if num_layers == 2:
            model.add(Bidirectional(GRU(bi_gru_units, return_sequences=False)))
        elif num_layers == 3:
            model.add(Bidirectional(GRU(bi_gru_units, return_sequences=True)))
            # 세 번째 레이어는 GRU 또는 BiGRU 선택 가능
            if last_layer_type == 'GRU':
                model.add(GRU(last_layer_units, return_sequences=False))
            else:
                model.add(Bidirectional(GRU(last_layer_units, return_sequences=False)))

    # 출력 레이어
    model.add(Dense(3))

    return model


# 실험 설정
experiment_settings = [
    # GRU 단독 실험
    {'model_type': 'GRU', 'gru_units': 32, 'bi_gru_units': 0, 'last_layer_units': 0, 'num_layers': 1,
     'last_layer_type': None},
    {'model_type': 'GRU', 'gru_units': 32, 'bi_gru_units': 0, 'last_layer_units': 0, 'num_layers': 2,
     'last_layer_type': None},
    {'model_type': 'GRU', 'gru_units': 32, 'bi_gru_units': 0, 'last_layer_units': 0, 'num_layers': 3,
     'last_layer_type': None},
    {'model_type': 'GRU', 'gru_units': 64, 'bi_gru_units': 0, 'last_layer_units': 0, 'num_layers': 1,
     'last_layer_type': None},
    {'model_type': 'GRU', 'gru_units': 64, 'bi_gru_units': 0, 'last_layer_units': 0, 'num_layers': 2,
     'last_layer_type': None},
    {'model_type': 'GRU', 'gru_units': 64, 'bi_gru_units': 0, 'last_layer_units': 0, 'num_layers': 3,
     'last_layer_type': None},
    {'model_type': 'GRU', 'gru_units': 128, 'bi_gru_units': 0, 'last_layer_units': 0, 'num_layers': 1,
     'last_layer_type': None},
    {'model_type': 'GRU', 'gru_units': 128, 'bi_gru_units': 0, 'last_layer_units': 0, 'num_layers': 2,
     'last_layer_type': None},
    {'model_type': 'GRU', 'gru_units': 128, 'bi_gru_units': 0, 'last_layer_units': 0, 'num_layers': 3,
     'last_layer_type': None},

    # BiGRU 단독 실험
    {'model_type': 'BiGRU', 'gru_units': 0, 'bi_gru_units': 32, 'last_layer_units': 0, 'num_layers': 1,
     'last_layer_type': None},
    {'model_type': 'BiGRU', 'gru_units': 0, 'bi_gru_units': 32, 'last_layer_units': 0, 'num_layers': 2,
     'last_layer_type': None},
    {'model_type': 'BiGRU', 'gru_units': 0, 'bi_gru_units': 32, 'last_layer_units': 0, 'num_layers': 3,
     'last_layer_type': None},
    {'model_type': 'BiGRU', 'gru_units': 0, 'bi_gru_units': 64, 'last_layer_units': 0, 'num_layers': 1,
     'last_layer_type': None},
    {'model_type': 'BiGRU', 'gru_units': 0, 'bi_gru_units': 64, 'last_layer_units': 0, 'num_layers': 2,
     'last_layer_type': None},
    {'model_type': 'BiGRU', 'gru_units': 0, 'bi_gru_units': 64, 'last_layer_units': 0, 'num_layers': 3,
     'last_layer_type': None},
    {'model_type': 'BiGRU', 'gru_units': 0, 'bi_gru_units': 128, 'last_layer_units': 0, 'num_layers': 1,
     'last_layer_type': None},
    {'model_type': 'BiGRU', 'gru_units': 0, 'bi_gru_units': 128, 'last_layer_units': 0, 'num_layers': 2,
     'last_layer_type': None},
    {'model_type': 'BiGRU', 'gru_units': 0, 'bi_gru_units': 128, 'last_layer_units': 0, 'num_layers': 3,
     'last_layer_type': None},

    # GRU + BiGRU 혼합 실험 (2개의 레이어)
    {'model_type': 'GRU+BiGRU', 'gru_units': 32, 'bi_gru_units': 32, 'last_layer_units': 0, 'num_layers': 2,
     'last_layer_type': None},
    {'model_type': 'GRU+BiGRU', 'gru_units': 32, 'bi_gru_units': 64, 'last_layer_units': 0, 'num_layers': 2,
     'last_layer_type': None},
    {'model_type': 'GRU+BiGRU', 'gru_units': 32, 'bi_gru_units': 128, 'last_layer_units': 0, 'num_layers': 2,
     'last_layer_type': None},
    {'model_type': 'GRU+BiGRU', 'gru_units': 64, 'bi_gru_units': 64, 'last_layer_units': 0, 'num_layers': 2,
     'last_layer_type': None},
    {'model_type': 'GRU+BiGRU', 'gru_units': 64, 'bi_gru_units': 128, 'last_layer_units': 0, 'num_layers': 2,
     'last_layer_type': None},
    {'model_type': 'GRU+BiGRU', 'gru_units': 128, 'bi_gru_units': 128, 'last_layer_units': 0, 'num_layers': 2,
     'last_layer_type': None},

    # GRU + BiGRU 혼합 실험 (3개의 레이어)
    {'model_type': 'GRU+BiGRU', 'gru_units': 32, 'bi_gru_units': 32, 'last_layer_units': 32, 'num_layers': 3,
     'last_layer_type': 'GRU'},
    {'model_type': 'GRU+BiGRU', 'gru_units': 32, 'bi_gru_units': 64, 'last_layer_units': 128, 'num_layers': 3,
     'last_layer_type': 'GRU'},
    {'model_type': 'GRU+BiGRU', 'gru_units': 64, 'bi_gru_units': 64, 'last_layer_units': 64, 'num_layers': 3,
     'last_layer_type': 'GRU'},
    {'model_type': 'GRU+BiGRU', 'gru_units': 64, 'bi_gru_units': 64, 'last_layer_units': 128, 'num_layers': 3,
     'last_layer_type': 'GRU'},
    {'model_type': 'GRU+BiGRU', 'gru_units': 32, 'bi_gru_units': 32, 'last_layer_units': 32, 'num_layers': 3,
     'last_layer_type': 'BiGRU'},
    {'model_type': 'GRU+BiGRU', 'gru_units': 32, 'bi_gru_units': 64, 'last_layer_units': 128, 'num_layers': 3,
     'last_layer_type': 'BiGRU'},
    {'model_type': 'GRU+BiGRU', 'gru_units': 64, 'bi_gru_units': 64, 'last_layer_units': 64, 'num_layers': 3,
     'last_layer_type': 'BiGRU'},
    {'model_type': 'GRU+BiGRU', 'gru_units': 64, 'bi_gru_units': 64, 'last_layer_units': 128, 'num_layers': 3,
     'last_layer_type': 'BiGRU'}

]

# 파일 경로 설정
folder_path = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Aircraft_Results'
file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

look_back = 10
forward_length = 0

# 실험 결과 저장
results = []

for experiment in experiment_settings:
    print(
        f"Running experiment with model_type: {experiment['model_type']}, GRU units: {experiment['gru_units']}, BiGRU units: {experiment['bi_gru_units']}, Last layer: {experiment['last_layer_type']}, Num layers: {experiment['num_layers']}")

    total_train_mse = 0
    total_test_mse = 0

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)

        # 데이터 필터링 및 스케일링
        df_filtered = df[['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)']]
        df_filtered.loc[:, ['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)']] = scaler.fit_transform(df_filtered[['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)']])

        # 데이터셋 생성
        X, y = create_dataset(df_filtered.values, look_back, forward_length)
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # 모델 생성 및 학습
        model = create_model(model_type=experiment['model_type'], input_shape=(look_back, 4),
                             gru_units=experiment['gru_units'], bi_gru_units=experiment['bi_gru_units'],
                             last_layer_units=experiment['last_layer_units'], num_layers=experiment['num_layers'],
                             last_layer_type=experiment['last_layer_type'])
        model.compile(optimizer=Adam(), loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

        # 학습 및 테스트 MSE 계산
        train_predictions = model.predict(X_train)
        train_mse = mean_squared_error(y_train, train_predictions)
        total_train_mse += train_mse

        test_predictions = model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_predictions)
        total_test_mse += test_mse

    # 각 실험의 평균 MSE 계산
    avg_train_mse = total_train_mse / len(file_list)
    avg_test_mse = total_test_mse / len(file_list)

    print(f"Experiment -> Avg Train MSE: {avg_train_mse}, Avg Test MSE: {avg_test_mse}")

    # 결과 저장
    results.append({
        'model_type': experiment['model_type'],
        'gru_units': experiment['gru_units'],
        'bi_gru_units': experiment['bi_gru_units'],
        'num_layers': experiment['num_layers'],
        'last_layer_type': experiment['last_layer_type'],
        'avg_train_mse': avg_train_mse,
        'avg_test_mse': avg_test_mse
    })

# 결과 출력
for result in results:
    print(
        f"Model Type: {result['model_type']}, GRU Units: {result['gru_units']}, BiGRU Units: {result['bi_gru_units']}, Last Layer Type: {result['last_layer_type']}, Num Layers: {result['num_layers']}, Avg Train MSE: {result['avg_train_mse']}, Avg Test MSE: {result['avg_test_mse']}")

