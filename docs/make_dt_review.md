# legacy_source/make_dt.py 리뷰

영상 하나에서 키포인트 간 거리를 뽑아 CSV로 저장하는 기존 코드. 구조는 괜찮으나 아래 문제로 재사용 전 수정 필요.

1. **안 돌아감** — `mp.solutions.hands` 삭제됨. 현재 설치되는 mediapipe(0.10.30~1.0.1)엔 `solutions` 모듈이 없음. 새 Tasks API(`mediapipe.tasks.vision.HandLandmarker`)로 교체 필요.
2. **왼손/오른손이 틀림** — MediaPipe가 손을 감지한 순서(idx 0, 1)로 좌/우를 나눔. 실제 좌우가 아니라 "먼저 잡힌 손" 기준이라 프레임마다 뒤바뀔 수 있음. `handedness` 결과 값을 안 씀.
3. **프레임 간격이 불안정** — `frame_rate // 2`로 나눠서 0.5초 간격을 잡는데, fps가 30이 아니거나 홀수면 간격이 틀어짐. fps가 1이면 0으로 나눠서 에러.
4. **영상 경계 정보 없음** — 영상마다 몇 행이 나왔는지 기록 안 하고 리스트에 그냥 이어붙임. 학습 시 6행씩 묶는 로직이 영상 경계와 안 맞을 수 있는 원인.
5. **모듈 로드하면 바로 실행됨** — `if __name__ == "__main__":` 가드 없이 파일 맨 아래서 바로 `process_all_videos()` 호출. import해서 함수만 재사용하기 불편함.

## 참고: 데이터셋 관계
- `new.csv`(7542행)와 `final.csv`(7782행, chicken·squid 영상 추가분 +240행 포함)는 둘 다 이 스크립트 계열로 뽑힌 결과로 보임.
- 현재 배포된 `new.pth`는 `new.csv`로 학습됨 (`new.ipynb`가 로드하는 파일, 테스트셋 크기 314개가 `new.csv` 기준 계산과 일치). `final.csv`는 재학습에 쓰인 적 없음.
