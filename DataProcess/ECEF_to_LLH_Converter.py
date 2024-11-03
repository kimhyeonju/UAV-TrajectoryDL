import csv
import math

def ecef_to_llh(x, y, z):
    a = 6378137.0  # 지구의 장반경(m)
    e = 8.1819190842622e-2  # 지구의 이심률
    esq = e ** 2
    b = math.sqrt(a**2 * (1 - esq))
    ep = math.sqrt((a**2 - b**2) / b**2)
    p = math.sqrt(x**2 + y**2)
    th = math.atan2(a * z, b * p)
    lon = math.atan2(y, x)
    lat = math.atan2((z + ep**2 * b * math.sin(th)**3), (p - esq * a * math.cos(th)**3))
    n = a / math.sqrt(1 - esq * math.sin(lat)**2)
    alt = p / math.cos(lat) - n
    alt_ft = alt * 3.28084  # Convert altitude to feet

    # Convert radians to degrees
    lon = math.degrees(lon)
    lat = math.degrees(lat)

    return lat, lon, alt, alt_ft

file_name = 'GreaterSapiens Flight Leg 9 SAL SFO'

input_filename = 'data/data_txt/'+file_name+'.txt'
output_filename = 'data/LLH_data_csv/'+file_name+'.csv'

with open(input_filename, 'r') as infile, open(output_filename, 'w', newline='') as outfile:
    reader = infile.readlines()
    writer = csv.writer(outfile)
    writer.writerow(['Date', 'Time', 'Latitude', 'Longitude', 'Altitude (m)', 'Altitude (ft)'])

    for line in reader:
        parts = line.strip().split(',')
        date_time = parts[0].split('T')
        date = date_time[0]
        time = date_time[1].replace('Z', '')
        x, y, z = map(float, parts[1:])
        lat, lon, alt_m, alt_ft = ecef_to_llh(x, y, z)
        writer.writerow([date, time, lat, lon, alt_m, alt_ft])

print(f"Data has been converted and saved to '{output_filename}'")
