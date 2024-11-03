import os
import json

# JSON 파일들이 위치한 디렉토리 경로
directory_path = '../data/Drone_data/interpolation_data/inter_대구_금호지구_202007171220_60m_45도_1.csv'

file_names = []

# 디렉토리 내의 모든 파일명을 순회
for file_name in os.listdir(directory_path):
    if file_name.endswith('.json') and file_name.split('.')[0].isdigit():  # JSON 파일이면서 숫자로만 구성되어 있는지 확인
        file_names.append(file_name)

# 파일명을 기반으로 정렬 (파일명에서 숫자만 추출하여 정수로 변환)
file_names.sort(key=lambda x: int(x.split('.')[0]))

# 누락된 frame_id를 찾기
expected_frame_id = 1  # 가정: frame_id가 1부터 시작
missing_frame_ids = []

for file_name in file_names:
    # 파일명에서 숫자 부분만 추출하여 정수로 변환
    current_frame_id = int(file_name.split('.')[0])
    if current_frame_id != expected_frame_id:
        while expected_frame_id < current_frame_id:
            missing_frame_ids.append(f"{expected_frame_id:08d}")  # 8자리 숫자 형태의 문자열로 저장
            expected_frame_id += 1
    expected_frame_id += 1  # 다음 예상 frame_id 업데이트

# 누락된 frame_id 출력
if missing_frame_ids:
    print("누락된 frame_id:", missing_frame_ids)
    print(len(missing_frame_ids))
else:
    print("모든 frame_id가 연속적입니다.")

# import pandas as pd
# import os
# import json
#
# # JSON 파일들이 위치한 디렉토리 경로
# directory_path = '/Volumes/DAIR/자율주행드론 비행 영상/Training/[라벨링][산림지]울산_작천정/202008261115_60m_45도_2/'
#
# # 최종적으로 병합할 데이터를 저장할 리스트
# data_list = []
#
# # 디렉토리 내의 모든 파일명을 순회하며 JSON 파일에서 특정 필드만 추출
# for file_name in sorted(os.listdir(directory_path)):
#     if file_name.endswith('.json') and file_name.split('.')[0].isdigit():
#         file_path = os.path.join(directory_path, file_name)
#         with open(file_path, 'r') as file:
#             data = json.load(file)
#             info = data['info']
#             # 필요한 필드만 추출하여 리스트에 추가
#             # JSON 구조에 따라 접근 방식을 조정할 수 있습니다
#             selected_data = {
#                 'Date': info.get('date', ''),
#                 'Frame_id': info.get('frame_id', ''),
#                 'Latitude': info.get('latitude', ''),
#                 'Longitude': info.get('longitude', ''),
#                 'Altitude (m)': info.get('altitude', '')
#             }
#             data_list.append(selected_data)
#
# # 추출된 데이터로부터 DataFrame 생성
# combined_df = pd.DataFrame(data_list)
#
# # 병합된 DataFrame을 CSV 파일로 저장
# csv_file_path = 'data/Drone_data/울산_작천정_202008261115_60m_45도_2.csv'
# combined_df.to_csv(csv_file_path, index=False)
#
# print(f"CSV 파일이 저장되었습니다: {csv_file_path}")
