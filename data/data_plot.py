import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker

# CSV 파일 로드
file_path = '/Users/admin/PycharmProjects/GP_test/data/interpolation_s/Inter_GreaterSapiens Flight Leg 3 Perth Antarctica.csv'
sampled_data = pd.read_csv(file_path)

# 데이터 샘플링 (예: 5개마다 1개씩 선택)
# sampled_data = data.iloc[::5, :]

# 3D 플롯 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 데이터 포인트 플로팅 (마커 크기 및 스타일 조정)
ax.scatter(sampled_data['Longitude'], sampled_data['Latitude'], sampled_data['Altitude (m)'], c='r', marker='o', s=0.1)

# 라벨 설정
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Altitude (m)')

# 과학적 표기법 사용하지 않도록 설정
ax.xaxis.se_major_formatter(ticker.FuncFormatter(lambda x, pos: x))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: x))
ax.zaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: x))

# 라벨 위치 조정
ax.xaxis.labelpad = 20
ax.yaxis.labelpad = 20
ax.zaxis.labelpad = 10

# 플롯 표시
plt.show()
