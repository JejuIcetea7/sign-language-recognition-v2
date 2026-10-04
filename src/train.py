"""
- 실험(tune.py)으로 고른 모델 하나를 전체 데이터로 학습하고 outputs/models/에 저장한다
- 모델 코드는 src/models/에 있고, 여기서는 고른 모델과 하이퍼파라미터만 지정한다
"""

import os

from data import load_data
from models import MODELS, save_model

# 실험으로 고른 모델과 하이퍼파라미터
model_name = "gru"
params = {"hidden_size": 384, "num_layers": 2, "dropout_prob": 0.3, "epochs": 200, "learning_rate": 0.001}


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    csv_path = os.path.join(base_dir, "data", "processed", "features.csv")
    save_path = os.path.join(base_dir, "outputs", "models", f"{model_name}.joblib")

    X, y, classes = load_data(csv_path)

    # 폴드로 나누지 않고 전체 데이터로 학습
    model = MODELS[model_name](**params)
    model.fit(X, y)
    save_model(save_path, model_name, model, classes)
    print(f"Model saved to {save_path}")
