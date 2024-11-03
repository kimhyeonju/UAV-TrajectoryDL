import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 경로
original_data_file = '../data/LLH_data_csv/Flight Wolfie Sydney Dallas Original 26381 Points.csv'  # 보간 전 데이터 파일 경로
interpolated_data_file = 'data/Interpolation_LLH_data_csv/Flight Wolfie Sydney Dallas Original 26381 Points_interpolation.csv'  # 보간 후 데이터 파일 경로

# 데이터 로딩
original_df = pd.read_csv(original_data_file)
interpolated_df = pd.read_csv(interpolated_data_file)

# 시간을 초 단위로 변환하여 'time_seconds' 열 추가 (예시로 사용, 실제 데이터에 맞게 조정 필요)
original_df['time_seconds'] = pd.to_datetime(original_df['Time']).dt.hour * 3600 + pd.to_datetime(original_df['Time']).dt.minute * 60 + pd.to_datetime(original_df['Time']).dt.second
interpolated_df['time_seconds'] = pd.to_datetime(interpolated_df['Time']).dt.hour * 3600 + pd.to_datetime(interpolated_df['Time']).dt.minute * 60 + pd.to_datetime(interpolated_df['Time']).dt.second

# 원본 데이터 그래프 설정
plt.plot(original_df['time_seconds'], original_df['Latitude'], label='Original Data', linestyle='-', linewidth=1, marker='o', markersize=4, color='blue', alpha=0.7)

# 보간된 데이터 그래프 설정
plt.plot(interpolated_df['time_seconds'], interpolated_df['Latitude'], label='Interpolated Data', linestyle='-', linewidth=0.5, marker='x', markersize=4, color='red', alpha=0.7)

# 그래프 세부 설정
plt.legend()
plt.xlabel('Time (seconds)')
plt.ylabel('Latitude')
plt.title('Latitude Comparison: Original vs Interpolated')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()  # 그래프 레이아웃 조정

# 그래프 표시
plt.show()