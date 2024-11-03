import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# 원본 데이터 폴더 경로
source_folder = 'data/Drone_data/original_data/'

# 결과 데이터를 저장할 폴더 경로
destination_folder = 'data/Drone_data/interpolation_data/'

# 원본 데이터 폴더 내의 모든 CSV 파일 목록을 가져옴
csv_files = [file for file in os.listdir(source_folder) if file.endswith('.csv')]

# 각 파일에 대하여 반복 처리
for file_name in csv_files:
    # CSV 파일 읽기
    original_df = pd.read_csv(source_folder + file_name)

    df = original_df[original_df['Altitude (m)'] != 0].reset_index(drop=True)

    # 전체 Frame_id 범위 생성
    full_frame_ids = np.arange(df['Frame_id'].min(), df['Frame_id'].max() + 1)

    # 누락된 Frame_id 찾기
    missing_frame_ids = np.setdiff1d(full_frame_ids, df['Frame_id'])

    if len(missing_frame_ids) > 0:
        # 보간 함수 생성
        interp_func_lat = interp1d(df['Frame_id'], df['Latitude'], kind='cubic', fill_value="extrapolate")
        interp_func_lon = interp1d(df['Frame_id'], df['Longitude'], kind='cubic', fill_value="extrapolate")
        interp_func_alt = interp1d(df['Frame_id'], df['Altitude (m)'], kind='cubic', fill_value="extrapolate")

        # 누락된 Frame_id에 대한 데이터 보간
        new_latitudes = interp_func_lat(missing_frame_ids)
        new_longitudes = interp_func_lon(missing_frame_ids)
        new_altitudes = interp_func_alt(missing_frame_ids)

        # 보간된 데이터를 포함하는 새로운 데이터프레임 생성
        interpolated_df = pd.DataFrame({
            'Date': df['Date'][0],
            'Frame_id': missing_frame_ids,
            'Latitude': new_latitudes,
            'Longitude': new_longitudes,
            'Altitude (m)': new_altitudes
        })

        # 보간된 데이터와 원본 데이터를 결합
        new_df = pd.concat([df, interpolated_df], ignore_index=True).sort_values('Frame_id').reset_index(drop=True)
        print('new ...')
        print('new df length : ',len(new_df))
        print('0 delete length : ',len(df))
        print('original df length : ',len(original_df))
    else:
        new_df = df.copy()
        print('original ...')
        print('new df length : ',len(new_df))
        print('0 delete length : ',len(df))
        print('original df length : ',len(original_df))
    new_df['Frame_id'] = new_df['Frame_id'].apply(lambda x: f'{x:08d}')

    # 결과 저장
    new_file_path = destination_folder + 'inter_' + file_name
    new_df.to_csv(new_file_path, index=False, columns=['Date', 'Frame_id', 'Latitude', 'Longitude', 'Altitude (m)'])

    print(f'File processed and saved: {new_file_path}')




#
# ########################
# import numpy as np
# import pandas as pd
#
#
# df = pd.read_csv('data/Drone_data/original_data/대구_수성못_202008291824_60m_45도_1.csv')
#
# # 전체 Frame_id 범위 생성 (최소 Frame_id에서 최대 Frame_id까지)
# full_frame_ids = np.arange(df['Frame_id'].min(), df['Frame_id'].max() + 1)
#
# # 누락된 Frame_id 찾기
# missing_frame_ids = np.setdiff1d(full_frame_ids, df['Frame_id'])
#
# print("누락된 Frame_id:", missing_frame_ids)
# print(len(missing_frame_ids))
