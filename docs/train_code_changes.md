# 학습 코드 수정 사항 (legacy `new.ipynb` / `lstm.py` 대비)

## 꼭 고칠 것

1. **시퀀스 묶기 / 라벨**
   - 기존: 시작점 `i`로 `X[i:i+6]`을 자르는 건 맞지만, 라벨을 `y[i+6]`(다음 윈도우 첫 행)에서 가져옴
   - 영향: 옛 `new.csv`에서 1256개 중 21개(1.7%)가 라벨이 틀어짐
   - 수정: `X.reshape(-1, 6, 111)`, `y = y[::6]` + `assert len(d) % 6 == 0`
2. **학습 루프 작성**
   - 노트북엔 없고, `lstm.py` 루프는 버그(매 epoch 옵티마이저 재생성, `zero_grad` 누락)
   - 수정: `zero_grad → forward → loss → backward → step` 표준 루프
3. **사람 단위 train/test split** (⏸ 보류: 이번 `train_model.py`에는 반영하지 않음, 기존 무작위 stratify split 사용)
   - 기존: 무작위 `stratify=y` → 같은 사람이 학습/테스트에 섞임
   - 문제: CSV에 사람/영상 정보가 없고 `os.listdir`도 비정렬이라 복원 불가
   - 결정 필요: 추출 시 영상 경로 목록을 `artifacts/index.csv`로 따로 저장 (CSV 스키마 유지, 재추출 약 20분)
4. **체크포인트에 라벨/설정 같이 저장**
   - 기존: `state_dict`만 저장, `predict.py`에 `LABELS` 수작업 입력
   - 수정: 가중치 + `label_encoder.classes_`, `input_size`, `timesteps`, `hidden_size`를 한 파일에
5. **Colab 경로 제거**: `drive.mount`, `/content/drive/...` → `artifacts/features.csv`

## 권장

- `forward`의 `h_0`, `c_0` 제거 (전역변수 의존, `nn.LSTM`이 기본 0 초기화)
- validation set 분리 (테스트셋 보면서 조정하면 점수가 낙관적)
- 모델 클래스는 학습/예측이 같이 import하는 한 파일에

## 그대로 유지

- `nan_to_num(nan=-1)` (없는 손 패딩, 학습/예측 일관)
- LSTM 구조 (hidden 64, 2층, dropout 0.5)
- 알파벳순 라벨 인코딩
