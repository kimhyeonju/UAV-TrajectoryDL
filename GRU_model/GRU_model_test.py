import pandas as pd
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging


def create_dataset(data, look_back, forward_length):
    X, y = [], []
    for i in range(len(data) - look_back - forward_length + 1):
        X.append(data[i:(i + look_back), :])
        y.append(data[i + look_back + forward_length - 1, :])
    return np.array(X), np.array(y)

# 모델 로드
model = load_model('GRU_model_v2.h5')

new_data_path = '/Users/admin/PycharmProjects/GP_test/inter_GreaterSapiens Flight Leg 5 SYD JNB_split_10s.csv'
new_df = pd.read_csv(new_data_path)

# start_frame_id = new_df['Frame_id'].iloc[0]  # 첫 'frame_id' 값
# max_frame_id = new_df['Frame_id'].iloc[-1]  # 마지막 'frame_id' 값
#
# # 모든 가능한 300 간격의 'frame_id' 계산
# frame_ids = range(start_frame_id, max_frame_id + 1, 300)
#
# df_filtered = new_df[new_df['Frame_id'].isin(frame_ids)]

df_filtered = new_df[['Latitude', 'Longitude', 'Altitude (m)']]

# 필터링된 데이터 출력
# print(df_filtered)

# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
df_filtered.loc[:, ['Latitude', 'Longitude', 'Altitude (m)']] = scaler.fit_transform(df_filtered[['Latitude', 'Longitude', 'Altitude (m)']])


# 테스트 데이터셋 생성
look_back = 10  # 과거 데이터 포인트 수
forward_length = 5  # 예측 길이
X_test, y_test = create_dataset(df_filtered.values, look_back, forward_length)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error on new dataset: {mse}")

logging.basicConfig(filename='test_results.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info(f"Tested on new data: {new_data_path}")
logging.info(f"Mean Squared Error on new dataset: {mse}")
