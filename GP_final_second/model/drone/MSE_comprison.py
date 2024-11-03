import matplotlib.pyplot as plt

# 텍스트 파일에서 MSE 추출 함수
def extract_mse_data(file_path):
    mse_data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if "Test MSE for" in line:
                # 'Test MSE for'와 'Prediction'을 기준으로 잘라서 MSE 값만 추출
                parts = line.split('Test MSE for ')[1].split(', Prediction')[0]
                file_name, mse_value = parts.split(': ')
                mse_value = float(mse_value)  # MSE 값을 실수형으로 변환
                mse_data[file_name] = mse_value
    return mse_data

# 두 개의 파일에서 MSE 데이터 추출
file1_mse_data = extract_mse_data('/Users/admin/PycharmProjects/GP_test/GP_final_second/data/drone/GRU_model_v5/look_back=10&forward=0/training_log_general.txt')
file2_mse_data = extract_mse_data('/Users/admin/PycharmProjects/GP_test/GP_final_second/data/drone/GRU_model_v4/look_back=10&forward=0/training_log_general.txt')

# 두 파일에서 추출된 데이터를 결합
combined_mse_data = {**file1_mse_data, **file2_mse_data}

# CSV 파일명과 MSE 수치를 분리
file_names = list(combined_mse_data.keys())
mse_values_file1 = [file1_mse_data.get(file, 0) for file in file_names]
mse_values_file2 = [file2_mse_data.get(file, 0) for file in file_names]

# MSE 값 중 최대값 찾기
max_mse_value = max(max(mse_values_file1), max(mse_values_file2))

# 그래프 생성
plt.figure(figsize=(12, 6))
width = 0.35  # 막대의 너비 설정

# 첫 번째 파일의 MSE 데이터 그래프 (풍향과 풍속을 제외한 모델)
plt.bar([x - width/2 for x in range(len(file_names))], mse_values_file1, width=width, label='Without Wind Parameters Model', color='skyblue')

# 두 번째 파일의 MSE 데이터 그래프 (모든 입력 파라미터를 사용한 모델)
plt.bar([x + width/2 for x in range(len(file_names))], mse_values_file2, width=width, label='With All Parameters Model', color='gray')

# 그래프 설정
plt.xlabel('CSV file')
plt.ylabel('MSE value')
plt.title('Comparison of MSE between Models')
plt.xticks(range(len(file_names)), ['Path ' + str(i+1) for i in range(len(file_names))])  # 파일명을 간단히 표시

# Y축 범위를 MSE 최대값에 맞게 설정 (약간의 여유를 주기 위해 1.1배로 설정)
plt.ylim(0, max_mse_value * 1.1)

plt.legend()
plt.tight_layout()
plt.show()