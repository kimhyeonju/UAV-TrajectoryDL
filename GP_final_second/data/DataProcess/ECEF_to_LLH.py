import csv
from datetime import datetime
from pyproj import Transformer

# ECEF 좌표계 -> WGS84 (위도, 경도) 변환 함수 (고도는 테이블에서 직접 사용)
def ecef_to_latlon(x, y, z):
    # Transformer 인스턴스 생성 (EPSG:4978 -> EPSG:4326 변환)
    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)

    # ECEF 좌표를 WGS84 좌표로 변환
    lon, lat, alt = transformer.transform(x, y, z)
    return lat, lon

# ft 단위를 m로 변환하는 함수
def feet_to_meters(feet):
    return feet * 0.3048

# kt 단위를 m/s로 변환하는 함수
def knots_to_mps(knots):
    return knots * 0.514444

# 입력 파일 경로
input_file = "/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/AircraftRoute/Flight QFA14 Buenos Aires to Darwin.txt"

# 출력 파일 경로
output_file = "/Users/admin/PycharmProjects/GP_test/GP_final_second/data/DataProcess/LLH_AircraftRoute/LLH_Flight QFA14 Buenos Aires to Darwin.csv"

# 새 CSV 파일을 생성
with open(output_file, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # CSV 헤더 작성
    csv_writer.writerow(['Date', 'Time', 'Latitude', 'Longitude', 'Altitude (m)', 'Speed (m/s)'])

    # 원본 데이터를 읽어 새로운 형식으로 변환
    with open(input_file, mode='r') as infile:
        for line in infile:
            parts = line.strip().split(',')

            # 날짜 및 시간 정보 추출
            date_time_str = parts[0]
            date_time = datetime.strptime(date_time_str, '%Y-%m-%dT%H:%M:%SZ')
            date = date_time.date()
            time = date_time.time()

            # ECEF X, Y, Z 좌표 추출
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])

            # 헤딩, 속도, 고도 추출
            info_parts = parts[4].split()
            speed_str = info_parts[1].replace('Spd:', '')
            altitude_str = info_parts[0].replace('Alt:', '')

            # ECEF 좌표를 위도, 경도로 변환
            lat, lon = ecef_to_latlon(x, y, z)

            # 고도(ft)를 m로 변환하고, 속도(kt)를 m/s로 변환
            altitude_m = feet_to_meters(float(altitude_str))
            speed_mps = knots_to_mps(float(speed_str))

            # 변환된 데이터를 새 파일에 기록
            csv_writer.writerow([date, time, lat, lon, altitude_m, speed_mps])

print(f"변환된 데이터를 {output_file}에 저장했습니다.")