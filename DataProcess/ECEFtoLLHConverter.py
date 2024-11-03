import math
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def ecef_to_llh(x, y, z):
    # WGS-84 타원체 상수
    a = 6378137.0  # 지구의 장반경(m)
    e = 8.1819190842622e-2  # 지구의 이심률

    asq = a ** 2
    esq = e ** 2

    b = math.sqrt(asq * (1 - esq))
    bsq = b ** 2

    ep = math.sqrt((asq - bsq) / bsq)
    p = math.sqrt(x ** 2 + y ** 2)
    th = math.atan2(a * z, b * p)

    lon = math.atan2(y, x)
    lat = math.atan2((z + ep ** 2 * b * math.sin(th) ** 3), (p - esq * a * math.cos(th) ** 3))
    n = a / math.sqrt(1 - esq * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - n
    alt_ft = alt * 3.28084

    # radian을 degree로 변환
    lon = math.degrees(lon)
    lat = math.degrees(lat)

    return (lat, lon, alt, alt_ft)

# X, Y, Z 좌표 예시
x = 3279352.062
y = 4750451.180
z = 2703892.838

latitude, longitude, altitude, altitude_ft = ecef_to_llh(3279360.973,4750436.956,2703908.567)

print("Latitude:", latitude, "degrees")
print("Longitude:", longitude, "degrees")
print("Altitude_m:", altitude, "meters")
print("Altitude_ft:", altitude_ft, "ft")
