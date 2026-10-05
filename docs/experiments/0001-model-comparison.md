# 0001. 모델 비교 (기본 하이퍼파라미터)

실행: `python src/run_experiments.py` (원본 결과: `outputs/results.csv`)


## 설정
- 데이터: `features.csv` 1272샘플 (6프레임 x 111특징), 22클래스
- 평가: stratified 5-fold, seed 42, 지표는 테스트 폴드 정확도
- 하이퍼파라미터 튜닝 없음 (기본값)
- 빈칸(미인식 손): LSTM/GRU/MLP/SVM/RandomForest는 -1, HistGB는 NaN 그대로


## 결과

| 순위 | 모델 | 평균 정확도 | 표준편차 |
|---|---|---|---|
| 1 | LSTM | **79.25%** | 2.28% |
| 2 | GRU | 77.44% | 1.87% |
| 3 | MLP | 73.98% | 3.18% |
| 4 | HistGradientBoosting | 72.17% | 2.04% |
| 5 | RandomForest | 70.45% | 3.43% |
| 6 | SVM | 65.25% | 1.93% |



## 한계
- 사람 단위 분할이 아니라 같은 사람이 학습과 테스트에 섞인다. 처음 보는 사람에 대한 점수는 이보다 낮을 수 있다.

