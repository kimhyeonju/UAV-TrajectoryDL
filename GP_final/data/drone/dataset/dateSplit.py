import pandas as pd
import os

# 데이터 파일 경로
input_file_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/dataset/WithoutTakeoff.csv'
output_folder_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/drone/dataset/dateSplit/'

# 출력 폴더가 존재하지 않으면 생성
os.makedirs(output_folder_path, exist_ok=True)

# 데이터 불러오기
df = pd.read_csv(input_file_path)

# 날짜별로 그룹화하여 CSV 파일로 저장
for date, group in df.groupby('date'):
    date_str = date.replace('/', '-')  # 날짜 형식에 따라 '/'을 '-'로 변경
    output_file_path = os.path.join(output_folder_path, f'IncludeTakeoff_{date_str}.csv')

    # 파일을 저장할 디렉토리가 존재하지 않으면 생성
    parent_dir = os.path.dirname(output_file_path)
    os.makedirs(parent_dir, exist_ok=True)

    # CSV 파일로 저장
    group.to_csv(output_file_path, index=False)
    print(f"Data for {date} saved to {output_file_path}")
