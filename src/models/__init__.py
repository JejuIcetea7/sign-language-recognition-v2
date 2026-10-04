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
