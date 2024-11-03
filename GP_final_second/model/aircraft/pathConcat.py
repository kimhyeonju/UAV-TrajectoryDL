import os
import pandas as pd

# 이륙, 순항, 착륙 폴더 경로 설정
takeoff_folder = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/RouteSplit_Files/Takeoff'
cruise_folder = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/RouteSplit_Files/Cruise'
landing_folder = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/RouteSplit_Files/Landing'

# 결과를 저장할 폴더 경로 설정
labeled_folder = '/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/Labeled_Routes/'
os.makedirs(labeled_folder, exist_ok=True)

# 이륙, 순항, 착륙 파일 목록을 불러오기
takeoff_files = sorted([file for file in os.listdir(takeoff_folder) if file.endswith('.csv')])
cruise_files = sorted([file for file in os.listdir(cruise_folder) if file.endswith('.csv')])
landing_files = sorted([file for file in os.listdir(landing_folder) if file.endswith('.csv')])

# 레이블링 및 병합 함수
def label_and_merge_files(takeoff_file, cruise_file, landing_file, output_file):
    # 이륙, 순항, 착륙 파일 불러오기
    takeoff_df = pd.read_csv(takeoff_file)
    cruise_df = pd.read_csv(cruise_file)
    landing_df = pd.read_csv(landing_file)

    # 레이블 추가 (이륙: 0, 순항: 1, 착륙: 2)
    takeoff_df['Phase'] = 0
    cruise_df['Phase'] = 1
    landing_df['Phase'] = 2

    # 이륙, 순항, 착륙 구간을 병합
    merged_df = pd.concat([takeoff_df, cruise_df, landing_df], ignore_index=True)

    # 병합된 데이터 저장
    merged_df.to_csv(output_file, index=False)
    print(f"Saved labeled and merged file: {output_file}")

# 병합 작업 수행
for takeoff_file, cruise_file, landing_file in zip(takeoff_files, cruise_files, landing_files):
    # 파일 이름에서 공통된 경로를 추출하여 저장 파일 이름 생성
    common_name = os.path.basename(takeoff_file).replace('_takeoff.csv', '')
    output_file = os.path.join(labeled_folder, f"{common_name}_labeled_merged.csv")

    # 각 구간 파일 경로 생성
    takeoff_path = os.path.join(takeoff_folder, takeoff_file)
    cruise_path = os.path.join(cruise_folder, cruise_file)
    landing_path = os.path.join(landing_folder, landing_file)

    # 파일 레이블링 및 병합
    label_and_merge_files(takeoff_path, cruise_path, landing_path, output_file)