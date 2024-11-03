import pandas as pd
import numpy as np
import os

file_name = 'GreaterSapiens Flight Leg 9 SAL SFO.csv'

# 데이터 파일 경로 설정
input_file_path = os.path.join('/Users/admin/PycharmProjects/GP_test/data/LLH_data_csv/', file_name)
output_file_path = os.path.join('/Users/admin/PycharmProjects/GP_test/data/interpolation_s/', 'Inter_' + file_name)

# 데이터 불러오기
df = pd.read_csv(input_file_path)

# 'datetime' 컬럼 생성 (date와 time을 합침)
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# 'datetime' 컬럼에서 변환되지 않은 항목 제거
df = df.dropna(subset=['datetime'])

# 'datetime' 컬럼을 인덱스로 설정
df = df.set_index('datetime')

# 1초 간격으로 새로운 인덱스 생성
start_time = df.index.min()
end_time = df.index.max()
new_index = pd.date_range(start=start_time, end=end_time, freq='5S')

# 보간을 위한 빈 데이터프레임 생성
df_new = pd.DataFrame(index=new_index)

# 기존 데이터와 새로운 인덱스 병합
df_combined = df_new.join(df)

# 각 열에 대해 선형 보간 수행
df_combined[['Latitude', 'Longitude', 'Altitude (m)']] = df_combined[['Latitude', 'Longitude', 'Altitude (m)']].interpolate(method='linear')

# 고도에서 음수 값을 0으로 설정
df_combined['Altitude (m)'] = df_combined['Altitude (m)'].apply(lambda x: max(0, x))

# 컬럼 이름 변경
df_combined.rename(columns={'Latitude': 'Latitude', 'Longitude': 'Longitude', 'Altitude (m)': 'Altitude (m)'}, inplace=True)

# datetime을 다시 분리
df_combined['Date'] = df_combined.index.date
df_combined['Time'] = df_combined.index.time

# 최종 데이터 프레임에서 필요한 컬럼만 선택
df_final = df_combined[['Date', 'Time', 'Latitude', 'Longitude', 'Altitude (m)']]

# CSV 파일로 저장
df_final.to_csv(output_file_path, index=False)

print(f"Interpolated data saved to {output_file_path}")
