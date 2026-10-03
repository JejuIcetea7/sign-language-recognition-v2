"""
- 수어 영상 -> 학습 데이터 csv
- MediaPipe를 이용하여 비디오 프레임에서 손가락 키포인트를 추출
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# 라이브러리

# 유클리드 거리 계산 함수
def calculate_euclidean_distance(point1, point2):
    """ 두 점 사이의 유클리드 거리를 계산하는 함수 """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# MediaPipe Hands 초기화 (옛 solutions API는 삭제됨 -> tasks API로 교체)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")
hands = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE,
    )
)
selected_keypoints = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


# 프레임 하나에서 왼손/오른손 키포인트 간 거리와 두 손 중심 간 거리를 추출
def extract_frame_features(frame):
    """ 하나의 프레임에서 손의 키포인트 간 거리와 두 손 중심 간 거리를 계산하는 함수 """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = hands.detect(mp_image)

    height, width, _ = frame.shape
    left_hand_distances = [None] * 55  # 왼손 거리 기본값을 None으로 설정
    right_hand_distances = [None] * 55  # 오른손 거리 기본값을 None으로 설정
    distance_between_hands = None  # 양손 중심 거리 기본값

    if result.hand_landmarks:
        hand_centers = {}

        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            # 선택된 키포인트만 픽셀 좌표로 변환
            landmarks = [(hand_landmarks[i].x * width, hand_landmarks[i].y * height) for i in selected_keypoints]

            # 선택된 키포인트 간의 모든 쌍에 대해 거리 계산 (총 55개)
            distances = [
                calculate_euclidean_distance(landmarks[i], landmarks[j])
                for i in range(len(landmarks))
                for j in range(i + 1, len(landmarks))
            ]

            # 왼손과 오른손을 구분하여 거리 정보를 저장 (감지 순서 대신 handedness 라벨 사용)
            side = handedness[0].category_name  # "Left" 또는 "Right"
            if side == "Left":
                left_hand_distances = distances
            elif side == "Right":
                right_hand_distances = distances

            # 손의 중심 좌표 계산
            hand_center_x = sum(lm[0] for lm in landmarks) / len(landmarks)
            hand_center_y = sum(lm[1] for lm in landmarks) / len(landmarks)
            hand_centers[side] = (hand_center_x, hand_center_y)

        # 두 손이 모두 인식된 경우에만 중심 간 거리 계산
        if "Left" in hand_centers and "Right" in hand_centers:
            distance_between_hands = calculate_euclidean_distance(hand_centers["Left"], hand_centers["Right"])

    return left_hand_distances + right_hand_distances + [distance_between_hands]


# 비디오를 처리하고 키포인트 간 거리와 두 손 중심 간 거리를 추출
def process_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / frame_rate if frame_rate else 0

    data = []  # 거리 정보를 저장할 리스트

    # 영상 길이(최대 3초)를 6구간으로 균등 분할해서 샘플링 시점을 고정
    # -> fps가 흔들려도, 영상이 3초보다 짧아도 항상 영상당 정확히 6행이 나옴 (영상 경계가 행 6개 단위와 항상 일치)
    window = min(duration, 3)
    sample_times = [window * k / 6 for k in range(1, 7)]

    for t in sample_times:
        frame_idx = min(round(t * frame_rate), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:  # 영상이 깨져서 해당 시점을 못 읽으면 건너뜀
            continue
        row_data = extract_frame_features(frame) + [label]
        data.append(row_data)

    cap.release()
    return data

def process_all_videos(root_directory, output_csv):
    """ 모든 폴더의 동영상을 처리하고 CSV 파일로 저장하는 함수 """
    all_data = []

    # 모든 폴더 탐색
    for label in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, label)
        if os.path.isdir(folder_path):
            # 폴더 내 모든 비디오 파일 탐색
            for video_file in os.listdir(folder_path):
                video_path = os.path.join(folder_path, video_file)
                if video_file.lower().endswith(('.mp4', '.avi', '.mov')):  # 비디오 파일 확장자 체크
                    print(f"Processing {video_path}...")
                    video_data = process_video(video_path, label)
                    all_data.extend(video_data)

    # 데이터프레임 생성 및 CSV 파일로 저장
    if all_data:
        # 컬럼 이름 설정: 왼손 거리 55개 + 오른손 거리 55개 + 두 손 중심 거리 1개 + 라벨 1개
        columns = (
            [f"Left_Distance_{i}" for i in range(55)] +
            [f"Right_Distance_{i}" for i in range(55)] +
            ["Distance_Between_Hands", "Label"]
        )
        df = pd.DataFrame(all_data, columns=columns)
        df.to_csv(output_csv, index=False)
        print(f"Data saved to {output_csv}")
    else:
        print("No data to save.")


if __name__ == "__main__":
    # 루트 디렉토리 및 출력 CSV 파일 설정
    root_directory = os.path.join(os.path.dirname(__file__), "..", "raw_videos")
    output_csv = os.path.join(os.path.dirname(__file__), "..", "artifacts", "features.csv")
    process_all_videos(root_directory, output_csv)
