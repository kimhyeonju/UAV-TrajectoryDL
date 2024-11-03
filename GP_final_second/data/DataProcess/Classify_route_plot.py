# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # 데이터 파일이 있는 폴더 경로 설정
# folder_path = '/GP_final_second/data/DataProcess/Aircraft_Results/'
#
# # 폴더에서 모든 CSV 파일 불러오기
# file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
#
# # 각 파일에 대해 반복
# for file_name in file_list:
#     file_path = os.path.join(folder_path, file_name)
#
#     # CSV 파일 불러오기
#     df = pd.read_csv(file_path)
#
#     # 고도 차이 계산
#     df['Altitude Change'] = df['Altitude (m)'].diff()
#
#     # 이륙, 순항, 착륙 구간 나누기 (이 값들은 비행 경로에 따라 달라질 수 있습니다)
#     takeoff = df[df['Altitude Change'] > 5]  # 고도가 증가하는 이륙 구간
#     cruise = df[(df['Altitude (m)'].diff() < 5) & (df['Altitude (m)'].diff() > -5) & (df['Altitude (m)'] > 5000)]  # 순항 구간 (안정된 고도와 고도 조건 추가)
#     landing = df[df['Altitude (m)'].diff() < -10]  # 착륙 구간 (고도 하강)
#
#     # 3D 플롯 설정
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # 이륙 구간 플롯
#     ax.plot(takeoff['Longitude'], takeoff['Latitude'], takeoff['Altitude (m)'], label='Takeoff', color='green')
#
#     # 순항 구간 플롯
#     ax.plot(cruise['Longitude'], cruise['Latitude'], cruise['Altitude (m)'], label='Cruise', color='blue')
#
#     # 착륙 구간 플롯
#     ax.plot(landing['Longitude'], landing['Latitude'], landing['Altitude (m)'], label='Landing', color='red')
#
#     # 축 레이블 설정
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
#     ax.set_zlabel('Altitude (m)')
#
#     # 플롯 제목과 범례 설정
#     plt.title(f'Flight Path: Takeoff, Cruise, Landing - {file_name}')
#     ax.legend()
#
#     output_folder = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Classify_route_v2/'
#     # 플롯 저장
#     output_file_path = os.path.join(output_folder, f'flight_path_{file_name.split(".")[0]}.png')
#     plt.savefig(output_file_path)
#
#     # 3D 플롯 닫기 (메모리 절약을 위해)
#     plt.close()
#
#     print(f"{file_name} 파일의 플롯 저장 완료: {output_file_path}")

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# CSV 파일 불러오기
file_path = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Aircraft_Results/Interpolated_LLH_Flight QFA14 Buenos Aires to Darwin.csv'
df = pd.read_csv(file_path)

# 고도 차이 계산
df['Altitude Change'] = df['Altitude (m)'].diff()

# 이륙, 순항, 착륙 구간 나누기 (이 값들은 비행 경로에 따라 달라질 수 있습니다)
takeoff = df[df['Altitude Change'] > 10]  # 고도가 증가하는 이륙 구간
cruise = df[(df['Altitude (m)'].diff() < 10) & (df['Altitude (m)'].diff() > -10) & (df['Altitude (m)'] > 5000)]  # 순항 구간 (안정된 고도와 고도 조건 추가)
landing = df[df['Altitude (m)'].diff() < -10]  # 착륙 구간 (고도 하강)

# 3D 플롯 설정
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 이륙 구간 플롯
ax.plot(takeoff['Longitude'], takeoff['Latitude'], takeoff['Altitude (m)'], label='Takeoff', color='green')

# 순항 구간 플롯
ax.plot(cruise['Longitude'], cruise['Latitude'], cruise['Altitude (m)'], label='Cruise', color='blue')

# 착륙 구간 플롯
ax.plot(landing['Longitude'], landing['Latitude'], landing['Altitude (m)'], label='Landing', color='red')

# 축 레이블 설정
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Altitude (m)')

# 플롯 제목과 범례 설정
plt.title('Flight Path: Takeoff, Cruise, Landing')
ax.legend()

# 3D 플롯 보여주기
plt.show()
