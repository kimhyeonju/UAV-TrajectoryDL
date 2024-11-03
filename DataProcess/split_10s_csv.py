
import pandas as pd

# 원본 CSV 파일 로딩
df = pd.read_csv('/Users/admin/PycharmProjects/GP_test/inter_GreaterSapiens Flight Leg 5 SYD JNB.csv')

# 'Date'와 'Time' 열을 합쳐서 datetime 타입으로 파싱
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# DateTime 열을 기준으로 초 단위로 변환
df['Seconds'] = df['DateTime'].dt.hour * 3600 + df['DateTime'].dt.minute * 60 + df['DateTime'].dt.second

# 시작 시간 (초 단위) 설정
start_second = df['Seconds'].min()

# 10초 간격의 시작점 데이터만 필터링
# 시작 시간에서 10초 간격으로 설정된 시간에 해당하는 데이터만 남깁니다.
filtered_df = df[df['Seconds'].subtract(start_second).mod(10) == 0]

# 필요한 열 선택
final_df = filtered_df[['Date', 'Time', 'Latitude', 'Longitude', 'Altitude (m)']]

# 결과를 하나의 CSV 파일로 저장
final_df.to_csv('inter_GreaterSapiens Flight Leg 5 SYD JNB_split_10s.csv', index=False)
print('Saved to data_at_10_second_intervals.csv')
