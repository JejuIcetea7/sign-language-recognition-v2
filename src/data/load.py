"""
- features.csv를 모델 입력(X, y)으로 변환한다
- features.csv는 왼손 55개 + 오른손 55개 + 양 손 사이 거리 1개 + 라벨 -> 총 112개 열로 이루어졌고 하나의 샘플은 6행으로 이루어져있다.

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

timesteps = 6  # 샘플 하나를 이루는 행 수 (영상 하나당 6행)


# CSV를 불러와서 (샘플 수, 6, 111) 입력과 라벨로 변환 (인식 안 된 손의 NaN은 그대로 둠)
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    assert len(data) % timesteps == 0, "행 수가 6의 배수가 아닙니다 (영상당 6행이 깨짐)"

    # 마지막 열인 'Label'을 제외한 나머지 데이터를 6행씩 묶어서 샘플로 만듦
    X = data.iloc[:, :-1].values.astype(np.float32).reshape(-1, timesteps, data.shape[1] - 1)
    y = data['Label'].values[::timesteps]  # 샘플(6행)마다 첫 행의 라벨 하나

    # 라벨 인코딩 (문자열 라벨을 정수로 변환)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return X, y, label_encoder.classes_
