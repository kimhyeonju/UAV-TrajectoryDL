import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import os

# 데이터 폴더 경로 설정
input_folder = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/LLH_AircraftRoute/'
output_folder = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Aircraft_Results/'

# 모든 파일에 대해 처리
for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):  # CSV 파일만 처리
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, 'Interpolated_' + file_name)

        # 데이터 불러오기
        df = pd.read_csv(input_file_path)

        # 'datetime' 컬럼 생성 (Date와 Time을 합침)
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S')

        # 'datetime' 컬럼을 인덱스로 설정
        df = df.set_index('datetime')

        # 5초 간격으로 새로운 인덱스 생성
        start_time = df.index.min()
        end_time = df.index.max()
        new_index = pd.date_range(start=start_time, end=end_time, freq='5S')

        # 보간을 위한 빈 데이터프레임 생성
        df_new = pd.DataFrame(index=new_index)

        # 기존 데이터와 새로운 인덱스 병합
        df_combined = df_new.join(df)

        # 각 열에 대해 보간 수행 (위도, 경도는 스플라인 보간, 고도는 선형 보간)
        for col, new_col in zip(['Latitude', 'Longitude'], ['Latitude', 'Longitude']):
            valid = df_combined[col].dropna().index
            values = df_combined[col].dropna().values

            if len(valid) < 2 or len(values) < 2:
                print(f"Skipping interpolation for column {col} in {file_name} due to insufficient data.")
                df_combined[new_col] = np.nan
                continue

            # 스플라인 보간 함수 생성
            cs = CubicSpline(valid, values)
            df_combined[new_col] = cs(df_combined.index)

        # 고도에 대해서는 선형 보간 적용
        df_combined['Altitude (m)'] = df_combined['Altitude (m)'].interpolate(method='linear')

        # 고도는 음수가 될 수 없으므로 음수 값을 0으로 변환
        df_combined['Altitude (m)'] = df_combined['Altitude (m)'].apply(lambda x: max(x, 0))  # ReLU(0) 적용

        # 속도에 대해서도 스플라인 보간 적용
        valid = df_combined['Speed (m/s)'].dropna().index
        values = df_combined['Speed (m/s)'].dropna().values

        if len(valid) < 2 or len(values) < 2:
            print(f"Skipping interpolation for Speed (m/s) in {file_name} due to insufficient data.")
            df_combined['Speed (m/s)'] = np.nan
        else:
            cs = CubicSpline(valid, values)
            df_combined['Speed (m/s)'] = cs(df_combined.index)

        # datetime을 다시 분리하여 저장
        df_combined['Date'] = df_combined.index.date
        df_combined['Time'] = df_combined.index.time

        # 최종 데이터 프레임에서 필요한 컬럼만 선택
        df_final = df_combined[['Date', 'Time', 'Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)']]

        # 결과를 CSV 파일로 저장
        df_final.to_csv(output_file_path, index=False)

        print(f"Interpolated data saved to {output_file_path}")