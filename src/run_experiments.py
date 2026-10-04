"""
- MODELS에 등록된 모델을 같은 폴드로 학습/평가하고 점수를 outputs/results.csv에 기록한다
- 점수는 학습에 쓰지 않은 테스트 폴드의 정확도이다
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd

from data import load_data, make_folds
from models import MODELS


# 모델 하나를 모든 폴드로 학습/평가하고 폴드별 정확도를 반환
def run_experiment(Model, X, y, folds):
    scores = []
    for fold, (train_idx, test_idx) in enumerate(folds):
        model = Model()  # 폴드마다 새 모델로 시작
        model.fit(X[train_idx], y[train_idx])
        predicted = model.predict(X[test_idx])
        accuracy = (predicted == y[test_idx]).mean()
        scores.append(accuracy)
        print(f"  Fold {fold + 1}/{len(folds)}: {accuracy * 100:.2f}%")
    return scores


# 모델별 결과를 results.csv에 한 줄씩 추가
def save_results(results, save_path):
    df = pd.DataFrame(results)
    df.to_csv(save_path, mode="a", header=not os.path.exists(save_path), index=False)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    csv_path = os.path.join(base_dir, "data", "processed", "features.csv")
    save_path = os.path.join(base_dir, "outputs", "results.csv")

    X, y, classes = load_data(csv_path)
    folds = make_folds(y)

    results = []
    for name, Model in MODELS.items():
        print(f"[{name}]")
        scores = run_experiment(Model, X, y, folds)
        print(f"  평균 정확도: {np.mean(scores) * 100:.2f}% (표준편차 {np.std(scores) * 100:.2f}%)")
        results.append({
            "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": name,
            "mean_accuracy": round(np.mean(scores), 4),
            "std_accuracy": round(np.std(scores), 4),
            "fold_accuracies": ";".join(f"{s:.4f}" for s in scores),
        })

    save_results(results, save_path)
