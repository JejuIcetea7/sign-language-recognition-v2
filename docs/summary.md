# 프로젝트 요약

22개 수어 단어 영상으로 인식 모델을 학습하고, API로 서빙한다. 기존 프로젝트의 확장·개선 버전이다.


## 목표와 달성 (`.claude/CLAUDE.md` 기준)

| 목표 | 상태 |
|---|---|
| 다양한 모델 비교 후 재선택 | **완료**: 7개 모델을 같은 5-fold로 비교, 튜닝, GRU 선택 |
| 분류 성능 최적화 | **일부**: 약 77% → 약 84% (5-fold, 동일 인물 포함). 추가 개선은 향후 과제 |
| 모델 서빙 | **로컬 구현 완료**: FastAPI로 영상 업로드 → 단어 예측. 홈서버 배포는 향후 과제 |


## 파이프라인

```
영상 (4명 x 22단어, 1272개)
  -> src/features/extract_features.py    MediaPipe 손 키포인트 -> 거리 특징 (영상당 6프레임 x 111)
  -> data/processed/features.csv
  -> src/data/                           6행씩 묶기, 라벨 인코딩, 5-fold 분할
  -> src/run_experiments.py, tune.py     모델 비교, 하이퍼파라미터 튜닝
  -> src/train.py                        선택한 GRU를 전체 데이터로 학습 -> outputs/models/gru.joblib
  -> src/predict.py, server.py           영상 한 개 -> 단어 (명령줄, API)
```


## 결과

| 단계 | 결과 |
|---|---|
| 이전 모델 (한 번 분할) | 76.98% |
| 기본값 비교 (5-fold) | LSTM 79.25%, GRU 77.44%, MLP 73.98%, HistGB 72.17%, RandomForest 70.45%, SVM 65.25% |
| 튜닝 후 (재확인, 다른 seed 폴드) | **GRU 83.96%**, SVM 73.35%, MLP 74.61%, LSTM 79.25% |

최종 모델: GRU (`hidden_size 384, num_layers 2, dropout 0.3, epochs 200, lr 0.001`)  
자세한 과정: [실험 0001](experiments/0001-model-comparison.md), [실험 0002](experiments/0002-model-selection.md)


## 서빙

```
python -m uvicorn --app-dir src server:app --port 8000
curl -F "file=@영상.mp4" http://localhost:8000/predict      ->  {"label": "add"}
```

- `GET /health`, `POST /predict` (브라우저에서 `/docs`로 시험 가능)
- 모델은 서버가 켜질 때 한 번만 로드하고, 요청은 잠금으로 한 번에 하나씩 처리한다.
- 영상이 아니거나 프레임을 못 읽으면 422로 거부한다.
- 영상 한 개 예측에 약 0.5초 (개발 맥에서 측정). 로컬 환경에서만 확인했고 배포는 하지 않았다.


## 재현

```
conda create -n sign-language python=3.12 && conda activate sign-language
pip install -r requirements.txt
python src/features/extract_features.py     # 영상 -> features.csv
python src/run_experiments.py               # 모델 비교
python src/tune.py [--grid] 모델이름        # 튜닝
python src/train.py                         # 최종 모델 학습·저장
```

MediaPipe는 1.0.0으로 고정한다. ([ADR 0002](adr/0002-pin-mediapipe.md))


## 결정 기록
- [ADR 0001](adr/0001-experiment-structure.md): 여러 모델을 실험하기 위한 코드 구조
- [ADR 0002](adr/0002-pin-mediapipe.md): MediaPipe 1.0.0 고정


## 한계와 향후 과제
- **평가:** 같은 사람이 학습과 테스트에 섞여 있어 처음 보는 사람에 대한 성능은 이보다 낮을 수 있다. 사람 단위 평가가 필요하다. (영상 목록 저장과 재추출이 필요)
- **특징:** 손 모양 거리만 사용하고 손의 위치와 움직임은 없다. 손 중심 좌표 추가가 가장 큰 개선 후보다.
- **튜닝:** GRU만 그리드서치까지 했고(27개 중 17개), LSTM은 랜덤서치만 했다. 그래서 두 모델의 비교가 완전히 공정하지는 않다.
- **서빙:** 홈서버 배포(환경, 외부 접속, 자동 재시작)는 하지 않았다. 동시 요청은 직렬 처리한다.
