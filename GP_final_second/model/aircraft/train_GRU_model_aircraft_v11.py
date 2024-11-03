# 이착륙 구간 따로 학습 방법
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.optimizers import Adam
from GRU_model_v8 import create_weighted_input_model8


# 학습 데이터와 라벨 생성 함수
def create_dataset(data, look_back, forward_length):
    X, y = [], []
    for i in range(len(data) - look_back - forward_length):
        X.append(data[i:(i + look_back), :4])  # 입력 파라미터: 4개 (위도, 경도, 고도, 속도)
        y.append(data[i + look_back + forward_length, :3])  # 예측할 파라미터: 위도, 경도, 고도 (3개)
    return np.array(X), np.array(y)


# 데이터 경로 및 폴더 설정
base_folder_path = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/RouteSplit_Files'
takeoff_folder = os.path.join(base_folder_path, 'Takeoff')
cruise_folder = os.path.join(base_folder_path, 'Cruise')
landing_folder = os.path.join(base_folder_path, 'Landing')

# 모델 및 결과 저장 경로 설정
model_save_path = '/Users/admin/PycharmProjects/GP_test/GP_final_second/SaveModel/aircraft_SaveModel/GRU_model_v11/'
predictions_path = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/aircraft/GRU_model_v11/path_to_save_predictions/'
plots_path = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/aircraft/GRU_model_v11/path_to_save_plots/'
log_dir = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/aircraft/GRU_model_v11/'
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, 'output_log.txt')
os.makedirs(predictions_path, exist_ok=True)
os.makedirs(plots_path, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)




look_back = 10
forward_length = 0


# 폴더 탐색 및 처리
folders = {
    'Takeoff': takeoff_folder,
    'Cruise': cruise_folder,
    'Landing': landing_folder
}

# 로그 파일 열기
with open(log_file_path, 'w') as log_file:
    for phase, folder in folders.items():
        file_list = [file for file in os.listdir(folder) if file.endswith('.csv')]
        for file_name in file_list:
            file_path = os.path.join(folder, file_name)
            df = pd.read_csv(file_path)
            df_filtered = df[['Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)']]

            # 데이터 스케일링
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(df_filtered)

            # 학습 및 테스트 데이터 준비
            X, y = create_dataset(df_scaled, look_back, forward_length)
            split_index = int(len(X) * 0.8)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            # 모델 생성 및 학습
            model = create_weighted_input_model8(input_shape=(look_back, 4))
            model.compile(optimizer=Adam(), loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

            # 테스트 데이터로 예측
            test_predictions = model.predict(X_test)

            # 원래 스케일로 변환
            predicted_coordinates = scaler.inverse_transform(
                np.concatenate([test_predictions, np.zeros((test_predictions.shape[0], 1))], axis=1))
            actual_coordinates = scaler.inverse_transform(
                np.concatenate([y_test, np.zeros((y_test.shape[0], 1))], axis=1))

            # 예측된 지점 및 실제 지점 DataFrame 변환
            predicted_df = pd.DataFrame(predicted_coordinates[:, :3], columns=['Latitude', 'Longitude', 'Altitude (m)'])
            actual_df = pd.DataFrame(actual_coordinates[:, :3], columns=['Latitude', 'Longitude', 'Altitude (m)'])

            # 전체 경로, 학습 경로, 테스트 실제 경로, 테스트 예측 경로 시각화 (3D 플롯)
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            # 전체 경로
            ax.plot(df_filtered['Longitude'], df_filtered['Latitude'], df_filtered['Altitude (m)'],
                    label='Full Path', color='gray', alpha=0.3)

            # 학습 경로
            ax.plot(df_filtered['Longitude'][:split_index], df_filtered['Latitude'][:split_index],
                    df_filtered['Altitude (m)'][:split_index], label='Train Path', color='blue')

            # 테스트 실제 경로
            ax.plot(df_filtered['Longitude'][split_index:], df_filtered['Latitude'][split_index:],
                    df_filtered['Altitude (m)'][split_index:], label='Test Actual Path', color='red')

            # 테스트 예측 경로
            ax.plot(predicted_df['Longitude'], predicted_df['Latitude'], predicted_df['Altitude (m)'],
                    label='Test Predicted Path', color='green')

            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_zlabel('Altitude')

            plt.title(f"{file_name} - {phase.capitalize()}")
            ax.legend()

            plt.savefig(f'{plots_path}{file_name}_{phase}_test_plot.png')
            plt.close(fig)

            # MSE, RMSE, MAE 계산 및 출력 (출력 변수 별로 계산)
            param_names = ['Latitude', 'Longitude', 'Altitude (m)']
            mse_total, rmse_total, mae_total = 0, 0, 0

            for i, param in enumerate(param_names):
                mse = mean_squared_error(actual_df[param], predicted_df[param])
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(actual_df[param], predicted_df[param])
                mse_total += mse
                rmse_total += rmse
                mae_total += mae
                log_file.write(f"{file_name} - {phase.capitalize()} - {param} - MSE: {mse}, RMSE: {rmse}, MAE: {mae}\n")
                print(f"{file_name} - {phase.capitalize()} - {param} - MSE: {mse}, RMSE: {rmse}, MAE: {mae}\n")

            # 전체 MSE, RMSE, MAE 계산
            mse_total /= 3  # 평균 MSE
            rmse_total /= 3  # 평균 RMSE
            mae_total /= 3   # 평균 MAE

            log_file.write(f"{file_name} - {phase.capitalize()} - Total - MSE: {mse_total}, RMSE: {rmse_total}, MAE: {mae_total}\n")
            print(f"{file_name} - {phase.capitalize()} - Total - MSE: {mse_total}, RMSE: {rmse_total}, MAE: {mae_total}\n")

# 모델 저장
model.save(f'{model_save_path}_gru_model_v11.h5')