"""
- 읽기 방식 비교 (docs/experiments/0005-reading-method.md): seek와 순차 읽기가 같은 프레임(이미지)을 읽는지 확인한다
- 단어마다 영상 10개를 균등 간격으로 골라 3개 프로세스로 나눠 비교하고, 프레임마다 결과를 outputs/reading_experiment/frame_compare.csv에 기록한다
- 이미지가 다른 프레임만 MediaPipe 특징도 두 방식으로 계산해 차이를 기록한다
- 사용법: python src/reading_experiment/compare_frames.py
"""

import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features.extract_features import extract_frame_features, frames_per_video

base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
root = os.path.join(base_dir, "data", "raw_videos")
videos_per_label = 10


# 읽을 프레임 번호 (extract_video_features와 같은 계산)
def frame_indices(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window = min(total / fps if fps else 0, 3)
    return [min(round(window * k / frames_per_video * fps), total - 1) for k in range(1, frames_per_video + 1)]


def read_seek(path, wanted):
    cap = cv2.VideoCapture(path)
    frames = {}
    for idx in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames[idx] = frame if ret else None
    cap.release()
    return frames


def read_sequential(path, wanted):
    cap = cv2.VideoCapture(path)
    frames, targets = {}, set(wanted)
    for idx in range(max(wanted) + 1):
        if not cap.grab():
            break
        if idx in targets:
            ret, frame = cap.retrieve()
            frames[idx] = frame if ret else None
    cap.release()
    return {idx: frames.get(idx) for idx in wanted}


# 영상 하나의 프레임별 비교 결과
def compare_video(args):
    label, name = args
    path = os.path.join(root, label, name)
    cap = cv2.VideoCapture(path)
    wanted = frame_indices(cap)
    cap.release()
    seek, seq = read_seek(path, wanted), read_sequential(path, wanted)

    rows = []
    for k, idx in enumerate(wanted):
        a, b = seek[idx], seq[idx]
        row = {"label": label, "video": name, "k": k, "frame_idx": idx, "seek_ok": a is not None, "seq_ok": b is not None,
               "same_image": False, "max_pixel_diff": np.nan, "max_feature_diff": np.nan}
        if a is not None and b is not None:
            row["same_image"] = bool(np.array_equal(a, b))
            row["max_pixel_diff"] = int(np.abs(a.astype(int) - b.astype(int)).max())
            if not row["same_image"]:
                fa, fb = np.array(extract_frame_features(a), dtype=float), np.array(extract_frame_features(b), dtype=float)
                row["max_feature_diff"] = float(np.nanmax(np.abs(fa - fb))) if not (np.isnan(fa).all() and np.isnan(fb).all()) else 0.0
                row["nan_pattern_same"] = bool((np.isnan(fa) == np.isnan(fb)).all())
        rows.append(row)
    return rows


# 단어마다 영상 10개를 균등 간격으로 선택 (읽기 시간 측정에서도 같은 영상을 씀)
def select_videos():
    targets = []
    for label in sorted(os.listdir(root)):
        folder = os.path.join(root, label)
        if not os.path.isdir(folder):
            continue
        names = sorted(n for n in os.listdir(folder) if n.lower().endswith((".mp4", ".avi", ".mov")))
        targets += [(label, names[i]) for i in np.linspace(0, len(names) - 1, videos_per_label).astype(int)]
    return targets


if __name__ == "__main__":
    targets = select_videos()

    out_dir = os.path.join(base_dir, "outputs", "reading_experiment")
    os.makedirs(out_dir, exist_ok=True)
    with Pool(3) as pool:
        rows = [row for result in pool.imap(compare_video, targets) for row in result]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "frame_compare.csv"), index=False)

    diff = df[~df.same_image]
    print(f"영상 {len(targets)}개, 프레임 {len(df)}개, 이미지가 같은 프레임 {int(df.same_image.sum())}개, 다른 프레임 {len(diff)}개")
    print(f"읽기 실패: seek {int((~df.seek_ok).sum())}, 순차 {int((~df.seq_ok).sum())}")
    if len(diff):
        print(f"다른 프레임의 최대 픽셀 차이 {int(diff.max_pixel_diff.max())}, 최대 특징 차이 {diff.max_feature_diff.max():.2e}, 다른 프레임이 있는 영상 {diff.video.nunique()}개")
