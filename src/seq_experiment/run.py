"""
- 시퀀스 길이(6 / 12 / 24 / 48 / 60프레임) 비교 실험 (계획: docs/experiments/0004-sequence-length.md)
- seed 하나가 폴드 분할과 모델 초기화를 같이 정하고, 같은 seed 안에서 모든 길이를 같은 분할로 평가한다
- 6, 12프레임은 features_24.csv에서 4번째, 2번째마다 골라 쓰고, 24 / 48 / 60프레임은 각 파일을 그대로 쓴다
- 결과: outputs/seq_experiment/results_seed{seed}.csv (길이별 Accuracy, Macro-F1), predictions_seed{seed}.csv (폴드별 예측값)
- 사용법: python src/seq_experiment/run.py seed  (이미 끝난 길이는 건너뜀)
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import data.load as load
from data import load_data, make_folds
from models import MODELS
from train import params  # 최종 GRU 하이퍼파라미터

base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
seq_dir = os.path.join(base_dir, "data", "processed", "seq")
default_out_dir = os.path.join(base_dir, "outputs", "seq_experiment")

lengths = [6, 12, 24, 48, 60]
source_of = {6: 24, 12: 24, 24: 24, 48: 48, 60: 60}  # 길이마다 어느 추출 파일에서 고르는지
num_classes = 22

# fit()이 torch.manual_seed(42)로 고정해 두므로, seed마다 초기화가 달라지게 바꿈 (모델 코드는 건드리지 않음)
current_seed = 42
_manual_seed = torch.manual_seed
torch.manual_seed = lambda _: _manual_seed(current_seed)


# 추출 파일을 (샘플 수, 프레임 수, 111)로 불러옴
def load_source(source):
    load.timesteps = source
    return load_data(os.path.join(seq_dir, f"features_{source}.csv"))


# seed 하나로 모든 길이를 평가해서 결과를 파일에 기록
def run_seed(seed, out_dir=default_out_dir, lengths=lengths, epochs=None):
    global current_seed
    current_seed = seed
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, f"results_seed{seed}.csv")
    preds_path = os.path.join(out_dir, f"predictions_seed{seed}.csv")
    done = set(pd.read_csv(results_path).frames) if os.path.exists(results_path) else set()
    model_params = {**params, **({"epochs": epochs} if epochs else {})}

    data_by_source, y = {}, None
    for source in sorted({source_of[n] for n in lengths}):
        data_by_source[source], y_source, _ = load_source(source)
        y = y_source if y is None else y
        assert (y == y_source).all(), "추출 파일마다 영상 순서가 같아야 같은 폴드가 됨"
    folds = make_folds(y, seed=seed)  # 모든 길이가 같은 분할을 씀

    for n in lengths:
        if n in done:
            continue
        source = source_of[n]
        X = data_by_source[source][:, source // n - 1::source // n]
        assert X.shape[1] == n
        print(f"[{n}f] seed {seed}, X {X.shape}", flush=True)

        start = time.time()
        accs, f1s, pred_rows = [], [], []
        for fold, (train_idx, test_idx) in enumerate(folds):
            model = MODELS["gru"](**model_params)
            model.fit(X[train_idx], y[train_idx])
            predicted = model.predict(X[test_idx])
            accs.append((predicted == y[test_idx]).mean())
            f1s.append(f1_score(y[test_idx], predicted, average="macro", labels=range(num_classes), zero_division=0))
            pred_rows.append(pd.DataFrame({"frames": n, "seed": seed, "fold": fold, "sample": test_idx,
                                           "y_true": y[test_idx], "y_pred": predicted}))
        seconds = time.time() - start

        print(f"  Accuracy {np.mean(accs) * 100:.2f}%  Macro-F1 {np.mean(f1s) * 100:.2f}%  ({seconds:.0f}초)", flush=True)
        pd.concat(pred_rows).to_csv(preds_path, mode="a", header=not os.path.exists(preds_path), index=False)
        pd.DataFrame([{"run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "frames": n, "seed": seed,
                       "accuracy": round(np.mean(accs), 4), "macro_f1": round(np.mean(f1s), 4),
                       "fold_accuracies": ";".join(f"{s:.4f}" for s in accs),
                       "fold_macro_f1": ";".join(f"{s:.4f}" for s in f1s),
                       "train_seconds": round(seconds)}]).to_csv(results_path, mode="a", header=not os.path.exists(results_path), index=False)


if __name__ == "__main__":
    torch.set_num_threads(3)  # seed마다 프로세스를 따로 띄워 동시에 돌리므로 코어를 나눠 씀
    run_seed(int(sys.argv[1]))
