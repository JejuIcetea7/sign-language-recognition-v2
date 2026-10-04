"""
- LSTM 모델 정의와 학습
- 모든 모델이 같은 방식(fit / predict)으로 쓰이도록 LSTMClassifier로 감싼다
- 입력 X: (샘플 수, 6, 111), 출력 predict: 샘플마다 라벨 번호 하나
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

    # 하이퍼파라미터 설정 (기본값은 기존 설정과 동일)
    def __init__(self, hidden_size=64, num_layers=2, dropout_prob=0.5, epochs=500, learning_rate=0.001):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob  # 드롭아웃 확률
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X, y):
        torch.manual_seed(42)  # 폴드마다 같은 조건으로 학습
        X = to_tensor(X)
        y = torch.tensor(y, dtype=torch.long)

        self.model = LSTMModel(X.shape[2], self.hidden_size, self.num_layers, len(np.unique(y)), self.dropout_prob, self.rnn)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for epoch in range(self.epochs):
            # 순전파
            outputs = self.model(X)
            loss = criterion(outputs, y)

            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.item():.4f}")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(to_tensor(X))
        return outputs.argmax(dim=1).numpy()
