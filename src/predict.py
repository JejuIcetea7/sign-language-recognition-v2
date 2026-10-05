"""
- 학습된 모델로 영상 하나의 수어 단어를 예측한다
- 사용법: python src/predict.py 영상경로 [모델파일경로]  (모델파일을 안 주면 outputs/models/gru.joblib)
- 모델은 호출하는 쪽에서 한 번만 load_model로 불러와서 predict_video에 넘긴다 (서버는 시작할 때 한 번만 로드)
"""

import os
import sys

from features.extract_features import extract_video_features, frames_per_video
from models import load_model

default_model_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "models", f"gru_{frames_per_video}f.joblib")


# 영상 하나를 예측해서 단어 이름을 반환
def predict_video(video_path, model, classes):
    features = extract_video_features(video_path)  # 영상 -> (24, 111), 인식 안 된 손은 NaN
    if features.shape != (frames_per_video, 111):  # 프레임을 못 읽은 영상은 학습 때와 입력 형태가 달라서 예측하지 않음
        raise ValueError(f"특징 모양이 ({frames_per_video}, 111)이 아닙니다: {features.shape}")

    predicted = model.predict(features[None])  # 샘플 1개짜리 배치 (1, 24, 111)
    return classes[predicted[0]]  # 라벨 번호를 단어 이름으로 변환


if __name__ == "__main__":
    video_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else default_model_path

    model, classes = load_model(model_path)
    print(predict_video(video_path, model, classes))
