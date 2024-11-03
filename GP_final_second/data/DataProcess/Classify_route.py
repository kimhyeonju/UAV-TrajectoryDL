import os
import pandas as pd

# 데이터 파일이 있는 폴더 경로 설정
folder_path = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Aircraft_Results/'
output_folder = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/RouteSplit_Files/'

# 이륙, 순항, 착륙 폴더 생성
takeoff_folder = os.path.join(output_folder, 'Takeoff')
cruise_folder = os.path.join(output_folder, 'Cruise')
landing_folder = os.path.join(output_folder, 'Landing')
os.makedirs(takeoff_folder, exist_ok=True)
os.makedirs(cruise_folder, exist_ok=True)
os.makedirs(landing_folder, exist_ok=True)

# 폴더에서 모든 CSV 파일 불러오기
# file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]


# 이륙, 순항, 착륙 구간 정의 함수
def split_flight_phases(df):
    df['Altitude Change'] = df['Altitude (m)'].diff()  # 고도 차이 계산

    # 이륙, 순항, 착륙 구간 나누기 (이 값들은 비행 경로에 따라 달라질 수 있음)
    takeoff = df[df['Altitude Change'] > 10]  # 고도가 증가하는 이륙 구간
    cruise = df[
        (df['Altitude (m)'].diff() < 10) & (df['Altitude (m)'].diff() > -10) & (df['Altitude (m)'] > 5000)]  # 순항 구간
    landing = df[df['Altitude (m)'].diff() < -10]  # 고도가 감소하는 착륙 구간

    return takeoff, cruise, landing


# 파일 처리
# for file_name in file_list:
    # file_path = os.path.join(folder_path, file_name)
file_path = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Aircraft_Results/Interpolated_LLH_Flight QFA14 Buenos Aires to Darwin.csv'
file_name = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Aircraft_Results/Interpolated_LLH_Flight QFA14 Buenos Aires to Darwin.csv'

    # CSV 파일 불러오기
df = pd.read_csv(file_path)

    # 이륙, 순항, 착륙 구간 나누기
takeoff, cruise, landing = split_flight_phases(df)

    # 이륙, 순항, 착륙 구간을 각각 CSV로 저장
takeoff_file = os.path.join(takeoff_folder, f'{file_name.split(".")[0]}_takeoff.csv')
cruise_file = os.path.join(cruise_folder, f'{file_name.split(".")[0]}_cruise.csv')
landing_file = os.path.join(landing_folder, f'{file_name.split(".")[0]}_landing.csv')

takeoff.to_csv(takeoff_file, index=False)
cruise.to_csv(cruise_file, index=False)
landing.to_csv(landing_file, index=False)

print(f'{file_name}: 이륙, 순항, 착륙 CSV 파일로 분할 완료.')