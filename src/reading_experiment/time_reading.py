"""
- 읽기 방식 비교 (docs/experiments/0005-reading-method.md): 영상을 열어 24프레임을 읽는 시간을 seek와 순차 읽기로 각각 잰다
- compare_frames.py와 같은 영상 220개, 영상마다 방식별 4번(첫 번째는 버림), 방식의 읽는 순서는 영상마다 번갈아 바꿈
- 하나의 프로세스로 순서대로 측정한다 (병렬이면 서로 영향을 줌). 결과: outputs/reading_experiment/read_times.csv
- 사용법: python src/reading_experiment/time_reading.py
"""

import os
import time

import cv2
import numpy as np
import pandas as pd

from compare_frames import base_dir, frame_indices, read_seek, read_sequential, root, select_videos

repeats = 4  # 첫 번째는 파일 캐시 영향을 빼려고 버림
methods = {"seek": read_seek, "sequential": read_sequential}

if __name__ == "__main__":
    rows = []
    targets = select_videos()
    for i, (label, name) in enumerate(targets):
        path = os.path.join(root, label, name)
        cap = cv2.VideoCapture(path)
        wanted = frame_indices(cap)
        cap.release()
        order = list(methods) if i % 2 == 0 else list(methods)[::-1]  # 읽는 순서를 번갈아 바꿈
        for rep in range(repeats):
            for method in order:
                start = time.perf_counter()
                methods[method](path, wanted)
                seconds = time.perf_counter() - start
                if rep > 0:
                    rows.append({"label": label, "video": name, "method": method, "rep": rep, "seconds": seconds})
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(targets)}", flush=True)

    df = pd.DataFrame(rows)
    out_dir = os.path.join(base_dir, "outputs", "reading_experiment")
    df.to_csv(os.path.join(out_dir, "read_times.csv"), index=False)
    for method, g in df.groupby("method"):
        print(f"{method:10s} n={len(g)}  p50 {np.percentile(g.seconds, 50) * 1000:7.1f} ms   p95 {np.percentile(g.seconds, 95) * 1000:7.1f} ms")
