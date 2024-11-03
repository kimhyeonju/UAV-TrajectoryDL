
import matplotlib.pyplot as plt
import pandas as pd

# txt 파일에서 데이터 읽어오기
df = pd.read_csv(r"C:\Users\kimhyeonju\PycharmProjects\GP_test\GP_final_second\model\drone\parameterTuning_result.txt", delimiter=",\s*", engine='python', header=None)

# 컬럼 이름 수동 설정
df.columns = ['Model Type', 'GRU Units', 'BiGRU Units', 'Last Layer Type', 'Num Layers', 'Avg Train MSE', 'Avg Test MSE']

# Model Type에서 불필요한 텍스트를 제거하여 'GRU'만 남기기
df['Model Type'] = df['Model Type'].apply(lambda x: x.split(":")[-1].strip())

# 숫자 컬럼에서 ':' 뒤의 값만 추출하여 변환
df['GRU Units'] = df['GRU Units'].apply(lambda x: x.split(":")[-1].strip()).astype(float)
df['Num Layers'] = df['Num Layers'].apply(lambda x: x.split(":")[-1].strip()).astype(int)  # 정수로 변환
df['Avg Test MSE'] = df['Avg Test MSE'].apply(lambda x: x.split(":")[-1].strip()).astype(float)

# 'Model Type'이 'GRU'인 데이터 필터링
df_gru = df[df['Model Type'] == 'GRU']

# Plot
plt.figure(figsize=(12, 8))

# 각 GRU Units 값에 대해 막대 그래프 그리기
width = 0.1  # 막대 너비
x_ticks = sorted(df_gru['Num Layers'].unique())  # x축 위치 (Num Layers 고유 값)
offsets = {32: -width, 64: 0, 128: width}  # 각 GRU Units에 따른 위치 오프셋
colors = {32: '#003f5c', 64: '#58508d', 128: '#ffa600'}  # 색상 설정

for units in [32, 64, 128]:
    df_units = df_gru[df_gru['GRU Units'] == units]
    if not df_units.empty:
        # x 위치에 오프셋을 추가하여 겹치지 않도록 설정
        x_positions = [x + offsets[units] for x in df_units['Num Layers']]
        plt.bar(x_positions, df_units['Avg Test MSE'], width=width, color=colors[units], edgecolor='black', label=f'GRU Units: {units}')

# x축 눈금을 데이터프레임의 'Num Layers' 고유 값으로 설정
plt.xticks(x_ticks)
plt.xlabel('Number of Layers', fontsize=12)
plt.ylabel('Avg Test MSE', fontsize=12)
plt.title('Test MSE by Number of Layers and GRU Units', fontsize=14)
plt.legend(title='GRU Units', fontsize=10)
plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()
plt.show()