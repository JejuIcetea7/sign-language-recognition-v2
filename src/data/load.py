"""
- features.csv를 모델 입력(X, y)으로 변환한다
- features.csv는 왼손 55개 + 오른손 55개 + 양 손 사이 거리 1개 + 라벨 -> 총 112개 열로 이루어졌고 하나의 샘플은 24행으로 이루어져있다.

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

timesteps = 24  # 샘플 하나를 이루는 행 수 (영상 하나당 24행, features/extract_features.py의 frames_per_video와 같아야 함)


# CSV를 불러와서 (샘플 수, 24, 111) 입력과 라벨로 변환 (인식 안 된 손의 NaN은 그대로 둠)
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    assert len(data) % timesteps == 0, f"행 수가 {timesteps}의 배수가 아닙니다 (영상당 {timesteps}행이 깨짐)"

    # 마지막 열인 'Label'을 제외한 나머지 데이터를 24행씩 묶어서 샘플로 만듦
    X = data.iloc[:, :-1].values.astype(np.float32).reshape(-1, timesteps, data.shape[1] - 1)
    y = data['Label'].values[::timesteps]  # 샘플(24행)마다 첫 행의 라벨 하나

    # 라벨 인코딩 (문자열 라벨을 정수로 변환)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return X, y, label_encoder.classes_
