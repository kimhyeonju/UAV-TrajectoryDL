# # import pandas as pd
# # import numpy as np
# # from scipy.interpolate import CubicSpline
# # import os
# #
# # file_name = 'IncludeTakeoff_2020-7-24.csv'
# #
# # # 데이터 파일 경로 설정
# # input_file_path = os.path.join('/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/dataset/dateSplit/', file_name)
# # output_file_path = os.path.join('/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Results/SplineInterpolation_1s/', 'Inter_' + file_name)
# #
# # # 데이터 불러오기
# # df = pd.read_csv(input_file_path)
# #
# # # 'datetime' 컬럼 생성 (date와 time을 합침)
# # df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y/%m/%d %H:%M:%S', errors='coerce')
# #
# # # 'datetime' 컬럼에서 변환되지 않은 항목 제거
# # df = df.dropna(subset=['datetime'])
# #
# # # 'datetime' 컬럼을 인덱스로 설정
# # df = df.set_index('datetime')
# #
# # # 1초 간격으로 새로운 인덱스 생성
# # start_time = df.index.min()
# # end_time = df.index.max()
# # new_index = pd.date_range(start=start_time, end=end_time, freq='1S')
# #
# # # 보간을 위한 빈 데이터프레임 생성
# # df_new = pd.DataFrame(index=new_index)
# #
# # # 기존 데이터와 새로운 인덱스 병합
# # df_combined = df_new.join(df)
# #
# # # 각 열에 대해 스플라인 보간 수행 (위도, 경도, 고도)
# # for col, new_col in zip(['lat', 'lon', 'alt'], ['Latitude', 'Longitude', 'Altitude (m)']):
# #     valid = df_combined[col].dropna().index
# #     values = df_combined[col].dropna().values
# #
# #     if len(valid) < 2 or len(values) < 2:
# #         print(f"Skipping interpolation for column {col} due to insufficient data.")
# #         df_combined[new_col] = np.nan
# #         continue
# #
# #     # 스플라인 보간 함수 생성
# #     cs = CubicSpline(valid, values)
# #     df_combined[new_col] = cs(df_combined.index)
# #
# # # 전체 속도 계산 및 보간 (NED 좌표계)
# # if 'north' in df_combined.columns and 'east' in df_combined.columns and 'down' in df_combined.columns:
# #     df_combined['total_speed'] = np.sqrt(df_combined['north']**2 + df_combined['east']**2 + df_combined['down']**2)
# #     valid = df_combined['total_speed'].dropna().index
# #     values = df_combined['total_speed'].dropna().values
# #
# #     if len(valid) < 2 or len(values) < 2:
# #         print(f"Skipping interpolation for total_speed due to insufficient data.")
# #         df_combined['Total Speed (m/s)'] = np.nan
# #     else:
# #         cs = CubicSpline(valid, values)
# #         df_combined['Speed (m/s)'] = cs(df_combined.index)
# # else:
# #     print("Skipping total_speed calculation due to missing NED coordinates.")
# #
# # # 풍속(wind_speed) 및 풍향(wind_direction) 보간
# # for col in ['wind_speed', 'wind_direction']:
# #     valid = df_combined[col].dropna().index
# #     values = df_combined[col].dropna().values
# #
# #     if len(valid) < 2 or len(values) < 2:
# #         print(f"Skipping interpolation for column {col} due to insufficient data.")
# #         df_combined[col] = np.nan
# #         continue
# #
# #     cs = CubicSpline(valid, values)
# #     df_combined[col] = cs(df_combined.index)
# #
# # # datetime을 다시 분리
# # df_combined['date'] = df_combined.index.date
# # df_combined['time'] = df_combined.index.time
# #
# # # 최종 데이터 프레임에서 필요한 컬럼만 선택
# # df_final = df_combined[['date', 'time', 'Latitude', 'Longitude', 'Altitude (m)',
# #                         'Speed (m/s)', 'wind_speed', 'wind_direction']]
# #
# # # CSV 파일로 저장
# # df_final.to_csv(output_file_path, index=False)
# #
# # print(f"Interpolated data saved to {output_file_path}")
#
# import pandas as pd
# import numpy as np
# from scipy.interpolate import CubicSpline
# import os
#
# file_name = 'IncludeTakeoff_2020-6-19.csv'
#
# # 데이터 파일 경로 설정
# input_file_path = os.path.join('/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/dataset/dateSplit/', file_name)
# output_file_path = os.path.join(
#     '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Results/RE_SplineInterpolation_1s/',
#     'Inter_' + file_name)
#
# # 데이터 불러오기
# df = pd.read_csv(input_file_path)
#
# # 'datetime' 컬럼 생성 (date와 time을 합침)
# df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y/%m/%d %H:%M:%S', errors='coerce')
#
# # 'datetime' 컬럼에서 변환되지 않은 항목 제거
# df = df.dropna(subset=['datetime'])
#
# # 'datetime' 컬럼을 인덱스로 설정
# df = df.set_index('datetime')
#
# # 1초 간격으로 새로운 인덱스 생성
# start_time = df.index.min()
# end_time = df.index.max()
# new_index = pd.date_range(start=start_time, end=end_time, freq='1S')
#
# # 보간을 위한 빈 데이터프레임 생성
# df_new = pd.DataFrame(index=new_index)
#
# # 기존 데이터와 새로운 인덱스 병합
# df_combined = df_new.join(df)
#
# # 각 열에 대해 스플라인 보간 수행 (위도, 경도, 고도)
# for col, new_col in zip(['lat', 'lon', 'alt'], ['Latitude', 'Longitude', 'Altitude (m)']):
#     valid = df_combined[col].dropna().index
#     values = df_combined[col].dropna().values
#
#     if len(valid) < 2 or len(values) < 2:
#         print(f"Skipping interpolation for column {col} due to insufficient data.")
#         df_combined[new_col] = np.nan
#         continue
#
#     # 스플라인 보간 함수 생성
#     cs = CubicSpline(valid, values)
#     df_combined[new_col] = cs(df_combined.index)
#
#     # 고도는 음수가 될 수 없으므로 음수 값을 0으로 변환
#     if new_col == 'Altitude (m)':
#         df_combined[new_col] = df_combined[new_col].apply(lambda x: max(x, 0))  # ReLU(0) 적용
#
# # 전체 속도 계산 및 보간 (NED 좌표계)
# if 'north' in df_combined.columns and 'east' in df_combined.columns and 'down' in df_combined.columns:
#     df_combined['total_speed'] = np.sqrt(
#         df_combined['north'] ** 2 + df_combined['east'] ** 2 + df_combined['down'] ** 2)
#     valid = df_combined['total_speed'].dropna().index
#     values = df_combined['total_speed'].dropna().values
#
#     if len(valid) < 2 or len(values) < 2:
#         print(f"Skipping interpolation for total_speed due to insufficient data.")
#         df_combined['Total Speed (m/s)'] = np.nan
#     else:
#         cs = CubicSpline(valid, values)
#         df_combined['Speed (m/s)'] = cs(df_combined.index)
# else:
#     print("Skipping total_speed calculation due to missing NED coordinates.")
#
# # 풍속(wind_speed) 및 풍향(wind_direction) 보간
# for col in ['wind_speed', 'wind_direction']:
#     valid = df_combined[col].dropna().index
#     values = df_combined[col].dropna().values
#
#     if len(valid) < 2 or len(values) < 2:
#         print(f"Skipping interpolation for column {col} due to insufficient data.")
#         df_combined[col] = np.nan
#         continue
#
#     cs = CubicSpline(valid, values)
#     df_combined[col] = cs(df_combined.index)
#
# # datetime을 다시 분리
# df_combined['date'] = df_combined.index.date
# df_combined['time'] = df_combined.index.time
#
# # 최종 데이터 프레임에서 필요한 컬럼만 선택
# df_final = df_combined[['date', 'time', 'Latitude', 'Longitude', 'Altitude (m)',
#                         'Speed (m/s)', 'wind_speed', 'wind_direction']]
#
# # CSV 파일로 저장
# df_final.to_csv(output_file_path, index=False)
#
# print(f"Interpolated data saved to {output_file_path}")

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import os

file_name = 'IncludeTakeoff_2020-7-24.csv'

# 데이터 파일 경로 설정
input_file_path = os.path.join('/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/dataset/dateSplit/', file_name)
output_file_path = os.path.join(
    '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Results/RE_SplineInterpolation_1s/',
    'Inter_' + file_name)

# 데이터 불러오기
df = pd.read_csv(input_file_path)

# 'datetime' 컬럼 생성 (date와 time을 합침)
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y/%m/%d %H:%M:%S', errors='coerce')

# 'datetime' 컬럼에서 변환되지 않은 항목 제거
df = df.dropna(subset=['datetime'])

# 'datetime' 컬럼을 인덱스로 설정
df = df.set_index('datetime')

# 1초 간격으로 새로운 인덱스 생성
start_time = df.index.min()
end_time = df.index.max()
new_index = pd.date_range(start=start_time, end=end_time, freq='1S')

# 보간을 위한 빈 데이터프레임 생성
df_new = pd.DataFrame(index=new_index)

# 기존 데이터와 새로운 인덱스 병합
df_combined = df_new.join(df)

# 각 열에 대해 보간 수행 (위도, 경도는 스플라인 보간, 고도는 선형 보간)
for col, new_col in zip(['lat', 'lon'], ['Latitude', 'Longitude']):
    valid = df_combined[col].dropna().index
    values = df_combined[col].dropna().values

    if len(valid) < 2 or len(values) < 2:
        print(f"Skipping interpolation for column {col} due to insufficient data.")
        df_combined[new_col] = np.nan
        continue

    # 스플라인 보간 함수 생성
    cs = CubicSpline(valid, values)
    df_combined[new_col] = cs(df_combined.index)

# 고도에 대해서는 선형 보간 적용
df_combined['Altitude (m)'] = df_combined['alt'].interpolate(method='linear')

# 고도는 음수가 될 수 없으므로 음수 값을 0으로 변환
df_combined['Altitude (m)'] = df_combined['Altitude (m)'].apply(lambda x: max(x, 0))  # ReLU(0) 적용

# 전체 속도 계산 및 보간 (NED 좌표계)
if 'north' in df_combined.columns and 'east' in df_combined.columns and 'down' in df_combined.columns:
    df_combined['total_speed'] = np.sqrt(
        df_combined['north'] ** 2 + df_combined['east'] ** 2 + df_combined['down'] ** 2)
    valid = df_combined['total_speed'].dropna().index
    values = df_combined['total_speed'].dropna().values

    if len(valid) < 2 or len(values) < 2:
        print(f"Skipping interpolation for total_speed due to insufficient data.")
        df_combined['Total Speed (m/s)'] = np.nan
    else:
        cs = CubicSpline(valid, values)
        df_combined['Speed (m/s)'] = cs(df_combined.index)
else:
    print("Skipping total_speed calculation due to missing NED coordinates.")

# 풍속(wind_speed) 및 풍향(wind_direction) 보간
for col in ['wind_speed', 'wind_direction']:
    valid = df_combined[col].dropna().index
    values = df_combined[col].dropna().values

    if len(valid) < 2 or len(values) < 2:
        print(f"Skipping interpolation for column {col} due to insufficient data.")
        df_combined[col] = np.nan
        continue

    cs = CubicSpline(valid, values)
    df_combined[col] = cs(df_combined.index)

# datetime을 다시 분리
df_combined['date'] = df_combined.index.date
df_combined['time'] = df_combined.index.time

# 최종 데이터 프레임에서 필요한 컬럼만 선택
df_final = df_combined[['date', 'time', 'Latitude', 'Longitude', 'Altitude (m)',
                        'Speed (m/s)', 'wind_speed', 'wind_direction']]

# CSV 파일로 저장
df_final.to_csv(output_file_path, index=False)

print(f"Interpolated data saved to {output_file_path}")