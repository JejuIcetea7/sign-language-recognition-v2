"""
- 모든 모델이 같은 조건으로 평가되도록 폴드를 만든다
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold


# 라벨 비율을 유지하면서 (train 인덱스, test 인덱스) 쌍을 n_splits개 만듦
def make_folds(y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))
