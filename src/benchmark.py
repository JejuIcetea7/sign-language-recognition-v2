"""
- 지연 시간 측정 (docs/experiments/0006-latency-24f-sequential.md): 영상 220개로 구간별 시간, 엔드투엔드, 동시요청을 잰다
- 사용법: python src/benchmark.py stages          구간별 (영상 읽기 / MediaPipe / 모델), 서버 없이 실행
          python src/benchmark.py http [서버주소]  엔드투엔드와 동시요청 (server.py를 먼저 켜 둘 것, 기본 http://localhost:8000)
- 결과: outputs/latency/ 아래 csv, 요약은 화면에 출력
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd

from features.extract_features import extract_frame_features, read_frames
from models import load_model
from predict import default_model_path
from reading_experiment.compare_frames import base_dir, root, select_videos

repeats = 4  # 영상마다 4번 실행하고 첫 번째는 캐시와 초기화 영향을 빼려고 버림
out_dir = os.path.join(base_dir, "outputs", "latency")


def percentiles(times):
    return f"p50 {np.percentile(times, 50) * 1000:7.1f} ms   p95 {np.percentile(times, 95) * 1000:7.1f} ms"


def video_paths():
    return [os.path.join(root, label, name) for label, name in select_videos()]


# 구간별: 영상 읽기 / MediaPipe / 모델 (하나의 프로세스로 순서대로)
def run_stages():
    model, _ = load_model(default_model_path)
    rows = []
    for i, path in enumerate(video_paths()):
        for rep in range(repeats):
            t0 = time.perf_counter()
            frames = read_frames(path)
            t1 = time.perf_counter()
            features = np.array([extract_frame_features(f) for f in frames], dtype=float)
            t2 = time.perf_counter()
            model.predict(features[None])
            t3 = time.perf_counter()
            if rep > 0:
                rows.append({"video": os.path.basename(path), "rep": rep, "read": t1 - t0, "mediapipe": t2 - t1, "model": t3 - t2})
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/220", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "stages.csv"), index=False)
    df["total"] = df[["read", "mediapipe", "model"]].sum(axis=1)
    print(f"## 구간별 (n={len(df)})")
    for name, label in [("read", "영상 읽기"), ("mediapipe", "MediaPipe"), ("model", "모델"), ("total", "합계")]:
        print(f"{label:10s} {percentiles(df[name])}")


def post_video(client, url, path):
    start = time.perf_counter()
    with open(path, "rb") as f:
        r = client.post(f"{url}/predict", files={"file": (os.path.basename(path), f, "video/mp4")})
    r.raise_for_status()
    return time.perf_counter() - start


# 엔드투엔드(요청 1개씩)와 동시요청
def run_http(url):
    videos = video_paths()
    with httpx.Client(timeout=300) as client:
        client.get(f"{url}/health").raise_for_status()

        rows = []
        for i, path in enumerate(videos):
            for rep in range(repeats):
                seconds = post_video(client, url, path)
                if rep > 0:
                    rows.append({"video": os.path.basename(path), "rep": rep, "seconds": seconds})
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/220", flush=True)
        e2e = pd.DataFrame(rows)
        os.makedirs(out_dir, exist_ok=True)
        e2e.to_csv(os.path.join(out_dir, "e2e.csv"), index=False)
        print(f"## 엔드투엔드 (요청 1개씩, n={len(e2e)})\n{percentiles(e2e.seconds)}")

        print("\n## 동시요청 (접속 수 x 10개 요청)")
        rows, counter = [], 0
        for workers in [1, 2, 4, 8]:
            n = workers * 10
            batch = [videos[(counter + j) % len(videos)] for j in range(n)]  # 220개에서 돌아가며 고름
            counter += n
            start = time.perf_counter()
            with ThreadPoolExecutor(workers) as pool:
                times = list(pool.map(lambda p: post_video(client, url, p), batch))
            elapsed = time.perf_counter() - start
            rows.append({"workers": workers, "requests": n, "throughput": n / elapsed,
                         "p50": np.percentile(times, 50), "p95": np.percentile(times, 95)})
            print(f"동시 {workers}: {n / elapsed:5.2f} req/s   {percentiles(times)}")
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "concurrent.csv"), index=False)


if __name__ == "__main__":
    if sys.argv[1] == "stages":
        run_stages()
    else:
        run_http(sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000")
