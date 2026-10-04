# 0001. 실험 관리를 위한 코드 구조 변경


## Status  
Proposed  


## Context  
현재 코드 베이스는 하나의 모델로만 실험이 가능하다.  
모델 성능 향상을 위하여 여러 모델을 같은 조건에서 비교할 수 있는 구조로 변경이 요구됨.  


## Decision  
`train_model.py` 하나를 데이터 → 모델 → 평가 → 최종 학습의 네 단계로 나눈다.

```
변경 전
  train_model.py        로딩 + 모델 + 학습 + 평가 + 저장 (전부)

변경 후
  data/processed/features.csv
          ↓
  src/data/             데이터 로딩, 분할
          ↓
  src/models/           모델 정의  ← 모델 추가·교체는 여기만
          ↓
  src/run_experiments.py   모델별 점수 비교  →  outputs/results.csv
          ↓
  src/train.py             선택한 모델 최종 학습  →  outputs/models/
```

규칙: 모든 모델은 `fit` / `predict`를 가지며 `src/models/`의 `MODELS` dict에 이름으로 등록한다.


## Consequences  
- 모델 추가는 `src/models/`에 모델과 `MODELS` dict 한 줄을 추가하는 것으로 끝난다.
- 모든 모델이 같은 데이터와 같은 폴드로 평가되어 점수를 비교할 수 있다.
- 실험 코드와 서빙용 최종 학습 코드가 분리된다.
- LSTM은 `fit` / `predict` 래퍼가 추가로 필요하다.
