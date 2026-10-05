"""
- 모델별로 하이퍼파라미터 조합을 무작위로 뽑아 5-fold로 평가하고 outputs/tuning.csv에 기록한다
- 가장 좋은 조합은 다른 seed로 만든 폴드로 한 번 더 평가한다 (같은 폴드로 고르면 점수가 낙관적이기 때문)
- 사용법: python src/tune.py [모델 이름 ...]  (이름을 안 주면 search_space의 모든 모델)
          python src/tune.py --grid gru        (grid_space 범위의 모든 조합을 시도)
"""

import itertools
import os
import random
import sys
from datetime import datetime
from functools import partial
import numpy as np

from data import load_data, make_folds
from data.load import timesteps
from models import MODELS
from run_experiments import run_experiment, save_results

n_trials = 20  # 모델마다 시도할 조합 수 (조합이 이보다 적으면 전부 시도)

# 모델별 탐색 범위 (여기에 없는 모델은 튜닝하지 않음)
rnn_space = {
    "hidden_size": [32, 64, 128, 256],
    "num_layers": [1, 2, 3],
    "dropout_prob": [0.2, 0.3, 0.5],
    "epochs": [100, 200, 300, 500],
    "learning_rate": [0.0005, 0.001, 0.005],
}
search_space = {
    "lstm": rnn_space,
    "gru": rnn_space,
    "mlp": {"hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)], "alpha": [0.0001, 0.001, 0.01]},
    "svm": {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.001, 0.01, 0.1]},
    "hist_gb": {"learning_rate": [0.03, 0.1, 0.3], "max_iter": [100, 200, 300], "max_leaf_nodes": [15, 31, 63]},
}

# 그리드서치용 범위: 랜덤서치 최고 조합 주변만 촘촘하게 (--grid 옵션, 없는 값은 모델 기본값을 사용)
grid_space = {
    "gru": {"hidden_size": [192, 256, 384], "dropout_prob": [0.2, 0.3, 0.4], "epochs": [200, 300, 400]},
}


# 탐색 범위에서 조합을 n개 뽑아(범위보다 크면 전부) 각각 5-fold 점수를 내고, 조합마다 바로 기록
def tune_model(name, space, X, y, folds, save_path, n=n_trials):
    grid = [dict(zip(space, values)) for values in itertools.product(*space.values())]
    trials = random.Random(42).sample(grid, min(n, len(grid)))

    results = []
    for i, params in enumerate(trials):
        print(f"[{name}] trial {i + 1}/{len(trials)}: {params}")
        scores = run_experiment(partial(MODELS[name], **params), X, y, folds)
        print(f"  평균 정확도: {np.mean(scores) * 100:.2f}%")
        save_results([make_row(name, params, scores, 42)], save_path)  # 중간에 멈춰도 결과가 남도록
        results.append((params, scores))
    return results


# 결과 한 줄 (csv에는 파라미터를 문자열로 기록)
def make_row(name, params, scores, seed):
    return {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": name,
        "params": str(params),
        "fold_seed": seed,
        "mean_accuracy": round(np.mean(scores), 4),
        "std_accuracy": round(np.std(scores), 4),
    }


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    csv_path = os.path.join(base_dir, "data", "processed", f"features_{timesteps}.csv")
    save_path = os.path.join(base_dir, "outputs", "tuning.csv")

    X, y, classes = load_data(csv_path)
    folds = make_folds(y)

    use_grid = "--grid" in sys.argv
    spaces = grid_space if use_grid else search_space
    n = float("inf") if use_grid else n_trials  # 그리드서치는 범위 안의 모든 조합을 시도

    for name in [a for a in sys.argv[1:] if a != "--grid"] or spaces:
        trials = tune_model(name, spaces[name], X, y, folds, save_path, n)

        # 가장 좋은 조합을 다른 seed의 폴드로 재확인
        best_params, best_scores = max(trials, key=lambda t: np.mean(t[1]))
        print(f"[{name}] 최고 조합: {best_params} ({np.mean(best_scores) * 100:.2f}%) -> 다른 seed 폴드로 재확인")
        recheck_scores = run_experiment(partial(MODELS[name], **best_params), X, y, make_folds(y, seed=7))
        print(f"[{name}] 재확인 평균 정확도: {np.mean(recheck_scores) * 100:.2f}%")
        save_results([make_row(name, best_params, recheck_scores, 7)], save_path)
