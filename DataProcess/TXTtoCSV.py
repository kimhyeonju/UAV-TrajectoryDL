import csv

file_name = 'GreaterSapiens Flight Leg 9 SAL SFO'
# 입력 TXT 파일 이름
input_filename = 'data/data_txt/'+file_name+'.txt'

# 출력 CSV 파일 이름
csv_filename = 'data/data_csv/'+file_name+'.csv'

# TXT 파일을 읽고 CSV 파일로 변환
with open(input_filename, 'r') as infile, open(csv_filename, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    # CSV 헤더 쓰기
    writer.writerow(['Date', 'Time', 'X', 'Y', 'Z'])

    # 입력 파일의 각 줄을 읽어 처리
    for line in infile:
        # 줄 끝의 줄바꿈 문자 제거 및 T와 Z 처리
        line = line.strip().replace('T', ' ').replace('Z', '')
        # 쉼표로 구분하여 날짜와 시간, X, Y, Z로 분리
        date_time, x, y, z = line.split(',')
        # 날짜와 시간 분리
        date, time = date_time.split(' ')
        # CSV 파일에 쓰기
        writer.writerow([date, time, x, y, z])

print(f"{csv_filename} 파일이 생성되었습니다.")
