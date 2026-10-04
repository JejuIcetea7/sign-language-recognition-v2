"""
- scikit-learn 모델들을 fit / predict 인터페이스로 감싼다
- 입력 X: (샘플 수, 6, 111)를 (샘플 수, 666)으로 펴서 사용한다
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# 6행을 한 줄로 펴서 scikit-learn 모델에 넘기는 공통 클래스
class SklearnClassifier:
    fill_nan = True  # 인식 안 된 손의 NaN을 -1로 채울지 여부 (NaN을 직접 처리하는 모델은 False)

    def prepare(self, X):
        X = X.reshape(len(X), -1)
        return np.nan_to_num(X, nan=-1.0) if self.fill_nan else X

    def fit(self, X, y):
        self.model = self.build()
        self.model.fit(self.prepare(X), y)

    def predict(self, X):
        return self.model.predict(self.prepare(X))


class HistGB(SklearnClassifier):
    fill_nan = False  # NaN을 그대로 처리함

    def build(self):
        return HistGradientBoostingClassifier(random_state=42)


class RandomForest(SklearnClassifier):
    def build(self):
        return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)


class MLP(SklearnClassifier):
    def build(self):
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42))


class SVM(SklearnClassifier):
    def build(self):
        return make_pipeline(StandardScaler(), SVC())
