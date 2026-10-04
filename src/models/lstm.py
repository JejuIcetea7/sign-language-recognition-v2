"""
- LSTM 모델 정의와 학습
- 모든 모델이 같은 방식(fit / predict)으로 쓰이도록 LSTMClassifier로 감싼다
- 입력 X: (샘플 수, 6, 111), 출력 predict: 샘플마다 라벨 번호 하나
- GRU, CNN은 이 클래스를 상속하고 신경망 구조(build)만 바꿔서 사용한다
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.5, rnn=nn.LSTM):
        super(LSTMModel, self).__init__()
        self.rnn = rnn(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)  # 초기 은닉 상태는 기본값(0)
        out = out[:, -1, :]  # 마지막 time step의 출력 사용
        out = self.dropout(out)
        out = self.fc(out)
        return out


# numpy 데이터를 PyTorch Tensor로 변환 (NaN을 -1로 대체)
def to_tensor(X):
    X = torch.tensor(X, dtype=torch.float32)
    return torch.nan_to_num(X, nan=-1.0)  # 인식 안 된 손은 -1로 채움 (predict에서도 동일하게 -1 사용)


# 모델 학습과 예측을 fit / predict로 감싼 클래스
class LSTMClassifier:
    rnn = nn.LSTM  # 순환 층 종류 (GRU는 이 값만 바꿔서 사용)
    defaults = {"hidden_size": 64, "num_layers": 2, "dropout_prob": 0.5, "epochs": 500, "learning_rate": 0.001}  # 기본 하이퍼파라미터

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        assert not unknown, f"알 수 없는 하이퍼파라미터: {unknown}"
        self.params = {**self.defaults, **params}

    # 신경망 구조 만들기 (다른 구조는 이 부분만 바꿔서 사용)
    def build(self):
        p = self.params
        return LSTMModel(self.input_size, p["hidden_size"], p["num_layers"], self.num_classes, p["dropout_prob"], self.rnn)

    def fit(self, X, y):
        torch.manual_seed(42)  # 폴드마다 같은 조건으로 학습
        X = to_tensor(X)
        y = torch.tensor(y, dtype=torch.long)

        self.input_size = X.shape[2]
        self.num_classes = len(np.unique(y))
        self.model = self.build()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.params["learning_rate"])

        self.model.train()
        for epoch in range(self.params["epochs"]):
            # 순전파
            outputs = self.model(X)
            loss = criterion(outputs, y)

            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{self.params['epochs']}], Loss: {loss.item():.4f}")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(to_tensor(X))
        return outputs.argmax(dim=1).numpy()

    # 저장할 상태: 모델 구조를 다시 만들 값과 가중치(state_dict)만 저장
    def get_state(self):
        return {
            "params": self.params,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "weights": self.model.state_dict(),
        }

    # 저장된 상태로 모델을 복원
    def set_state(self, state):
        self.params = state["params"]
        self.input_size = state["input_size"]
        self.num_classes = state["num_classes"]
        self.model = self.build()
        self.model.load_state_dict(state["weights"])
