import pandas as pd

# 데이터 파일 경로
input_file_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/dataset/IncludeTakeoff.csv'
output_file_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/dataset/IncludeTakeoff_1s.csv'

# 데이터 불러오기
df = pd.read_csv(input_file_path)

# 필요한 컬럼만 선택
df_filtered = df[['date', 'time', 'lat', 'lon', 'alt']]

# 'datetime' 컬럼 생성
df_filtered['datetime'] = pd.to_datetime(df_filtered['date'] + ' ' + df_filtered['time'])

# 1초 간격으로 자르기
df_filtered = df_filtered.set_index('datetime').resample('1S').first().reset_index()

# 'datetime' 컬럼 분리
df_filtered['date'] = df_filtered['datetime'].dt.date
df_filtered['time'] = df_filtered['datetime'].dt.time

# 최종 데이터 프레임에서 'datetime' 컬럼 제거
df_final = df_filtered[['date', 'time', 'lat', 'lon', 'alt']]

# 컬럼 이름 변경
df_final.columns = ['date', 'time', 'Latitude', 'Longitude', 'Altitude (m)']

# CSV 파일로 저장
df_final.to_csv(output_file_path, index=False)

print(f"Filtered data saved to {output_file_path}")
