from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import os
from tensorflow.keras.layers import Input


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# 데이터 로드
data_path = 'data/split_10s/Flight Wolfie Sydney Dallas Original 26381 Points_interpolation_split_10s.csv'  # CSV 파일 경로
df = pd.read_csv(data_path)

df = df[['Latitude', 'Longitude', 'Altitude (m)']]


# 정규화
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 표준화
# scaler = StandardScaler()
# df_standardized = pd.DataFrame(scaler.fit_transform(df_normalized), columns=df_normalized.columns)

# 위도, 경도, 고도 선택
X = df_normalized[['Latitude', 'Longitude', 'Altitude (m)']].values[:-1]
y = df_normalized[['Latitude', 'Longitude', 'Altitude (m)']].shift(-1).values[:-1]


# NaN 값 제거 (예측할 다음 위치가 없는 마지막 행 때문에 필요)
X = X[:-1]
y = y[:-1]
# print(df.head())
# print(len(X))
# print(len(y))


# 데이터를 훈련, 검증, 테스트 세트로 분할
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.18, random_state=42) # 0.18 * 0.85 ≈ 0.15


# 데이터를 3차원으로 변환
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# print(X_train.shape)
# 모델 아키텍처 정의
model = Sequential([
    Input(shape=(1, 3)),
    Conv1D(filters=64, kernel_size=1, activation='relu', padding='same'),
    # MaxPooling1D(pool_size=1),
    LSTM(50),
    Dense(50, activation='relu'),
    Dense(3)
])
model.summary()
# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 모델 학습
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), batch_size=32, verbose=2)

# 검증 및 테스트 세트에 대한 MSE 계산
train_mse = model.evaluate(X_train, y_train, verbose=0)
val_mse = model.evaluate(X_val, y_val, verbose=0)
test_mse = model.evaluate(X_test, y_test, verbose=0)

print(f"Train MSE: {train_mse}")
print(f"Validation MSE: {val_mse}")
print(f"Test MSE: {test_mse}")
