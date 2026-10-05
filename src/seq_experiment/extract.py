"""
- 시퀀스 길이 실험용 추출: 영상당 N프레임을 뽑아 data/processed/seq/features_N.csv로 저장한다
- 기존 extract_features.py / features.csv는 건드리지 않고, 프레임 하나의 특징 계산만 가져다 쓴다
- 샘플링: 앞 3초를 N구간으로 균등 분할 (기존 6프레임과 같은 방식). 24일 때 4번째마다가 기존 6프레임 시점, 2번째마다가 12프레임 시점
- 사용법: python src/seq_experiment/extract.py [프레임수, 기본 24]
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features.extract_features import extract_frame_features

frames_per_video = int(sys.argv[1]) if len(sys.argv) > 1 else 24  # 영상당 뽑을 프레임 수


# 영상 하나에서 N프레임의 특징을 뽑음 (seek 대신 앞에서부터 순차 읽기), 못 읽으면 None
def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window = min(total / fps if fps else 0, 3)
    wanted = [min(round(window * k / frames_per_video * fps), total - 1) for k in range(1, frames_per_video + 1)]

    features, last = [], None
    for idx in range(max(wanted) + 1):
        if not cap.grab():
            break
        if idx in wanted:
            ret, frame = cap.retrieve()
            if not ret:
                break
            last = (idx, extract_frame_features(frame))
        if last and last[0] == idx:  # 같은 프레임을 여러 시점이 가리킬 수 있음 (짧은 영상)
            features.extend([last[1]] * wanted.count(idx))
    cap.release()
    return np.array(features, dtype=float) if len(features) == frames_per_video else None


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    root = os.path.join(base_dir, "data", "raw_videos")
    out_dir = os.path.join(base_dir, "data", "processed", "seq")
    os.makedirs(out_dir, exist_ok=True)

    rows, index, skipped = [], [], []
    for label in sorted(os.listdir(root)):
        folder = os.path.join(root, label)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith((".mp4", ".avi", ".mov")):
                continue
            print(f"Processing {label}/{name}...", flush=True)
            features = extract_video_features(os.path.join(folder, name))
            if features is None:
                skipped.append(f"{label}/{name}")
                continue
            rows.extend(features.tolist())
            index.extend([(label, name)] * frames_per_video)

    columns = [f"Left_Distance_{i}" for i in range(55)] + [f"Right_Distance_{i}" for i in range(55)] + ["Distance_Between_Hands"]
    df = pd.DataFrame(rows, columns=columns)
    df["Label"] = [label for label, _ in index]
    df.to_csv(os.path.join(out_dir, f"features_{frames_per_video}.csv"), index=False)
    pd.DataFrame(index[::frames_per_video], columns=["label", "video"]).to_csv(os.path.join(out_dir, f"index_{frames_per_video}.csv"), index=False)
    print(f"saved {len(df) // frames_per_video} videos, skipped {len(skipped)}: {skipped}")
