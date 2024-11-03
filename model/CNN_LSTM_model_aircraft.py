from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import pandas as pd
import os
import glob

# 데이터 폴더 경로
data_folder = 'data/split_10s/'
all_files = glob.glob(os.path.join(data_folder, "*.csv"))


# 모델 아키텍처 정의 함수
def create_model():
    model = Sequential([
        Input(shape=(1, 3)),
        Conv1D(filters=64, kernel_size=1, activation='relu', padding='same'),
        LSTM(50),
        Dense(50, activation='relu'),
        Dense(3)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


# 모델 초기화
model = create_model()

# 각 파일에 대해 모델 학습
for file in all_files:
    # 데이터 로드 및 전처리
    df = pd.read_csv(file)
    df = df[['Latitude', 'Longitude', 'Altitude (m)']]

    # 정규화
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # X, y 생성
    X = df_normalized[['Latitude', 'Longitude', 'Altitude (m)']].values[:-1]
    y = df_normalized[['Latitude', 'Longitude', 'Altitude (m)']].shift(-1).values[:-1]

    # NaN 값 제거 (예측할 다음 위치가 없는 마지막 행 때문에 필요)
    X = X[:-1]
    y = y[:-1]

    # 데이터를 훈련, 검증, 테스트 세트로 분할
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.18,
                                                      random_state=42)  # 0.18 * 0.85 ≈ 0.15

    # 데이터를 3차원으로 변환
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # 모델 학습
    model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), batch_size=32, verbose=2)

    # 검증 및 테스트 세트에 대한 MSE 계산
    train_mse = model.evaluate(X_train, y_train, verbose=0)
    val_mse = model.evaluate(X_val, y_val, verbose=0)
    test_mse = model.evaluate(X_test, y_test, verbose=0)

    print(f"File: {file}")
    print(f"Train MSE: {train_mse}")
    print(f"Validation MSE: {val_mse}")
    print(f"Test MSE: {test_mse}")

# 모델 저장
# model.save('CNN_LSTM_model_aircraft.h5')

# 모델 요약 정보 출력
model.summary()

# 모델 구조를 이미지로 저장
plot_model(model, to_file='CNN_LSTM_model_aircraft_structure.png', show_shapes=True, show_layer_names=True)
