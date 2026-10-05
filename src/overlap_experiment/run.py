"""
- 읽기와 MediaPipe 처리 겹치기 비교 (docs/experiments/0008-overlap-read-and-process.md)
- 현재(순차): 24프레임을 모두 읽은 뒤 MediaPipe로 처리, 겹치기: 읽는 동안 바로 처리 (extract_video_features)
- 영상 220개에서 특징이 완전히 같은지와 특징 추출 시간을 잰다. 하나의 프로세스로 순서대로 실행한다.
- 결과: outputs/overlap_experiment/times.csv
- 사용법: python src/overlap_experiment/run.py [영상 수 제한]
"""

import os
import sys
import time

import numpy as np
import pandas as pd

from features.extract_features import extract_frame_features, extract_video_features, read_frames
from reading_experiment.compare_frames import base_dir, root, select_videos

repeats = 4  # 영상마다 4번 실행하고 첫 번째는 캐시와 초기화 영향을 빼려고 버림


# 현재 방식: 24프레임을 모두 읽은 뒤 처리
def extract_sequential(path):
    return np.array([extract_frame_features(f) for f in read_frames(path)], dtype=float)


methods = {"sequential": extract_sequential, "overlap": extract_video_features}

if __name__ == "__main__":
    targets = select_videos()[: int(sys.argv[1]) if len(sys.argv) > 1 else None]
    rows, same, mismatched = [], 0, []
    for i, (label, name) in enumerate(targets):
        path = os.path.join(root, label, name)
        order = list(methods) if i % 2 == 0 else list(methods)[::-1]  # 실행 순서를 번갈아 바꿈
        features = {}
        for rep in range(repeats):
            for method in order:
                start = time.perf_counter()
                features[method] = methods[method](path)
                seconds = time.perf_counter() - start
                if rep > 0:
                    rows.append({"video": name, "method": method, "rep": rep, "seconds": seconds})
        if np.array_equal(features["sequential"], features["overlap"], equal_nan=True):
            same += 1
        else:
            mismatched.append(name)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(targets)}", flush=True)

    df = pd.DataFrame(rows)
    out_dir = os.path.join(base_dir, "outputs", "overlap_experiment")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "times.csv"), index=False)
    print(f"특징이 완전히 같은 영상: {same}/{len(targets)}  {mismatched[:5]}")
    for method, g in df.groupby("method"):
        print(f"{method:10s} n={len(g)}  p50 {np.percentile(g.seconds, 50) * 1000:7.1f} ms   p95 {np.percentile(g.seconds, 95) * 1000:7.1f} ms")
