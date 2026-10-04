import joblib

from .lstm import LSTMClassifier
from .gru import GRUClassifier
from .sklearn_models import HistGB, RandomForest, MLP, SVM

# 실험할 모델 등록 (이름: 모델 클래스)
MODELS = {
    "lstm": LSTMClassifier,
    "gru": GRUClassifier,
    "hist_gb": HistGB,
    "random_forest": RandomForest,
    "mlp": MLP,
    "svm": SVM,
}


# 학습된 모델을 라벨 목록과 함께 한 파일로 저장
def save_model(path, name, model, classes):
    joblib.dump({"name": name, "classes": list(classes), "state": model.get_state()}, path)


# 저장된 파일에서 (모델, 라벨 목록)을 복원
def load_model(path):
    saved = joblib.load(path)
    model = MODELS[saved["name"]]()
    model.set_state(saved["state"])
    return model, saved["classes"]
