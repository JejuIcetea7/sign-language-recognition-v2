"""
- scikit-learn 모델들을 fit / predict 인터페이스로 감싼다
- 입력 X: (샘플 수, 24, 111)를 (샘플 수, 2664)로 펴서 사용한다
- 하이퍼파라미터는 인자로 받고, 주지 않으면 defaults 값을 사용한다 (scikit-learn 모델의 인자 이름 그대로)
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# 24행을 한 줄로 펴서 scikit-learn 모델에 넘기는 공통 클래스
class SklearnClassifier:
    fill_nan = True  # 인식 안 된 손의 NaN을 -1로 채울지 여부 (NaN을 직접 처리하는 모델은 False)
    defaults = {}  # 기본 하이퍼파라미터

    def __init__(self, **params):
        self.params = {**self.defaults, **params}

    def prepare(self, X):
        X = X.reshape(len(X), -1)
        return np.nan_to_num(X, nan=-1.0) if self.fill_nan else X

    def fit(self, X, y):
        self.model = self.build()
        self.model.fit(self.prepare(X), y)

    def predict(self, X):
        return self.model.predict(self.prepare(X))

    # 저장할 상태: 하이퍼파라미터와 학습된 scikit-learn 모델
    def get_state(self):
        return {"params": self.params, "estimator": self.model}

    # 저장된 상태로 모델을 복원
    def set_state(self, state):
        self.params = state["params"]
        self.model = state["estimator"]


class HistGB(SklearnClassifier):
    fill_nan = False  # NaN을 그대로 처리함
    defaults = {"random_state": 42}

    def build(self):
        return HistGradientBoostingClassifier(**self.params)


class RandomForest(SklearnClassifier):
    defaults = {"n_estimators": 300, "random_state": 42, "n_jobs": -1}

    def build(self):
        return RandomForestClassifier(**self.params)


class MLP(SklearnClassifier):
    defaults = {"hidden_layer_sizes": (128, 64), "max_iter": 500, "random_state": 42}

    def build(self):
        return make_pipeline(StandardScaler(), MLPClassifier(**self.params))


class SVM(SklearnClassifier):
    def build(self):
        return make_pipeline(StandardScaler(), SVC(**self.params))
