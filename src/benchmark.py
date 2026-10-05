"""
- 지연시간 측정: 구간별(영상 읽기 / MediaPipe / 모델)과 엔드투엔드(HTTP /predict), 동시 접속 수별 처리량
- 사용법: python src/benchmark.py 영상경로 [서버주소] [반복횟수]
  서버주소 기본값 http://localhost:8000 (먼저 server.py를 켜 둘 것, 안 켜져 있으면 구간별 측정만 함)
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

from features.extract_features import extract_frame_features, read_frames
from models import load_model
from predict import default_model_path


def percentiles(times):
    return f"p50 {np.percentile(times, 50) * 1000:7.1f} ms   p95 {np.percentile(times, 95) * 1000:7.1f} ms"


# 영상 하나의 읽기 / MediaPipe / 모델 시간을 따로 잰다 (샘플링 시점은 extract_video_features와 같다)
def run_stages(video_path, model):
    t0 = time.perf_counter()
    frames = read_frames(video_path)
    t1 = time.perf_counter()

    features = np.array([extract_frame_features(f) for f in frames], dtype=float)
    t2 = time.perf_counter()

    model.predict(features[None])
    t3 = time.perf_counter()
    return t1 - t0, t2 - t1, t3 - t2


def post_video(client, url, video_path):
    start = time.perf_counter()
    with open(video_path, "rb") as f:
        r = client.post(f"{url}/predict", files={"file": (os.path.basename(video_path), f, "video/mp4")})
    r.raise_for_status()
    return time.perf_counter() - start


if __name__ == "__main__":
    video_path = sys.argv[1]
    url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    repeat = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    model, _ = load_model(default_model_path)
    run_stages(video_path, model)  # 첫 호출은 초기화 비용이 섞이므로 버림
    stages = np.array([run_stages(video_path, model) for _ in range(repeat)])
    print(f"## 구간별 (n={repeat})")
    for name, col in zip(["영상 읽기", "MediaPipe", "모델"], stages.T):
        print(f"{name:10s} {percentiles(col)}")
    print(f"{'합계':10s} {percentiles(stages.sum(axis=1))}")

    try:
        httpx.get(f"{url}/health", timeout=3).raise_for_status()
    except httpx.HTTPError:
        sys.exit(f"\n서버({url})가 꺼져 있어 엔드투엔드 측정은 건너뜀")

    with httpx.Client(timeout=120) as client:
        post_video(client, url, video_path)  # 워밍업
        e2e = [post_video(client, url, video_path) for _ in range(repeat)]
        print(f"\n## 엔드투엔드 (HTTP, 동시 1, n={repeat})\n{percentiles(e2e)}")

        print("\n## 동시 접속 수별 (접속 수 x 10개 요청)")
        for workers in [1, 2, 4, 8]:
            n = workers * 10
            start = time.perf_counter()
            with ThreadPoolExecutor(workers) as pool:
                times = list(pool.map(lambda _: post_video(client, url, video_path), range(n)))
            elapsed = time.perf_counter() - start
            print(f"동시 {workers}: {n / elapsed:5.2f} req/s   {percentiles(times)}")
