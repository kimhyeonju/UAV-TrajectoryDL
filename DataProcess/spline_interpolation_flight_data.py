import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import os

# 원본 데이터 폴더 경로
source_folder = 'data/LLH_data_csv/'

# 결과 데이터를 저장할 폴더 경로
destination_folder = 'data/Interpolation_LLH_data_csv/'

# 원본 데이터 폴더 내의 모든 CSV 파일 목록을 가져옴
csv_files = [file for file in os.listdir(source_folder) if file.endswith('.csv')]

for file_name in csv_files:
    # CSV 파일 읽기
    df = pd.read_csv(source_folder + file_name)

    # 'date'와 'time' 컬럼을 합쳐서 파이썬 datetime 객체로 변환
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    # 시작 시간을 기준으로 시간을 초 단위로 변환
    start_time = df['datetime'].iloc[0]
    df['seconds'] = (df['datetime'] - start_time).dt.total_seconds()
    df = df.drop_duplicates(subset=['seconds'])

    print(file_name)
    # 3차 스플라인 보간 함수 생성
    interp_func_lat = interp1d(df['seconds'], df['Latitude'], kind='cubic', fill_value="extrapolate")
    interp_func_lon = interp1d(df['seconds'], df['Longitude'], kind='cubic', fill_value="extrapolate")
    interp_func_alt_m = interp1d(df['seconds'], df['Altitude (m)'], kind='cubic', fill_value="extrapolate")
    interp_func_alt_ft = interp1d(df['seconds'], df['Altitude (ft)'], kind='cubic', fill_value="extrapolate")

    # 새로운 시간 배열 생성 (1초 간격)
    new_seconds = np.arange(df['seconds'].iloc[0], df['seconds'].iloc[-1] + 1)

    # 보간된 데이터 계산
    new_Latitudes = interp_func_lat(new_seconds)
    new_Longitudes = interp_func_lon(new_seconds)
    new_Altitudes_m = interp_func_alt_m(new_seconds)
    new_Altitudes_ft = interp_func_alt_ft(new_seconds)

    # 새로운 데이터프레임 생성
    new_df = pd.DataFrame({
        'datetime': [start_time + timedelta(seconds=s) for s in new_seconds],
        'Latitude': new_Latitudes,
        'Longitude': new_Longitudes,
        'Altitude (m)': new_Altitudes_m,
        'Altitude (ft)': new_Altitudes_ft
    })

    # 'date'와 'time' 컬럼 분리
    new_df['Date'] = new_df['datetime'].apply(lambda x: x.date())
    new_df['Time'] = new_df['datetime'].apply(lambda x: x.time())


    new_file_path = destination_folder + 'inter_' + file_name
    new_df.to_csv(new_file_path, index=False, columns=['Date', 'Time', 'Latitude', 'Longitude', 'Altitude (m)'])
    print(f'File processed and saved: {new_file_path}')

