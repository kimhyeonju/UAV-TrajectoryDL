from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import MeanSquaredError

# 모델 파일 경로
model_path = '/Users/admin/PycharmProjects/GP_test/model/CNN_LSTM_model_aircraft.h5'

# 사용자 정의 손실 함수 및 기타 사용자 정의 객체를 정의
custom_objects = {'mse': MeanSquaredError()}

# 모델 불러오기
model = load_model(model_path, custom_objects=custom_objects)

# 모델 요약 정보 출력
model.summary()

# 모델 구조를 이미지로 저장
plot_model(model, to_file='CNN_LSTM_model_aircraft_structure.png', show_shapes=True, show_layer_names=True)
