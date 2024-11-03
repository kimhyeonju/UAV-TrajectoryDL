import pandas as pd
import os
import glob

# 원본 데이터 폴더 및 출력 폴더 경로 설정
input_folder = '/Users/admin/PycharmProjects/GP_test/data/Drone_data/'
output_folder = '/Users/admin/PycharmProjects/GP_test/data/Drone_data/'
os.makedirs(output_folder, exist_ok=True)

# 원본 데이터 파일 목록 가져오기
all_files = glob.glob(os.path.join(input_folder, "*.csv"))

# 300개 간격으로 데이터 필터링 및 저장
for file in all_files:
    df = pd.read_csv(file)

    # Frame_id를 기준으로 300개 간격으로 데이터 선택
    filtered_df = df.iloc[::30]

    # 필요한 열 선택 (원본 데이터와 동일한 열)
    final_df = filtered_df[['Date', 'Frame_id', 'Latitude', 'Longitude', 'Altitude (m)']]

    # 새로운 파일명 생성
    output_file = os.path.join(output_folder, os.path.basename(file).replace('.csv', '_split_30.csv'))

    # 새로운 파일로 저장
    final_df.to_csv(output_file, index=False)
    print(f'Saved to {output_file}')
