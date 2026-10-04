"""
- features.csv를 입력으로 모델을 학습한다
- features.csv는 왼손 55개 + 오른손 55개 + 양 손 사이 거리 1개 + 라벨 -> 총 112개 열로 이루어졌고 하나의 샘플은 6행으로 이루어져있다.

"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# 하이퍼파라미터 설정
timesteps = 6  # 샘플 하나를 이루는 행 수 (영상 하나당 6행)
hidden_size = 64
num_layers = 2
dropout_prob = 0.5  # 드롭아웃 확률
epochs = 500
learning_rate = 0.001


# CSV를 불러와서 (샘플 수, 6, 111) 입력과 라벨로 변환
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    assert len(data) % timesteps == 0, "행 수가 6의 배수가 아닙니다 (영상당 6행이 깨짐)"

    # 마지막 열인 'Label'을 제외한 나머지 데이터를 6행씩 묶어서 샘플로 만듦
    X = data.iloc[:, :-1].values.astype(np.float32).reshape(-1, timesteps, data.shape[1] - 1)
    y = data['Label'].values[::timesteps]  # 샘플(6행)마다 첫 행의 라벨 하나

    # 라벨 인코딩 (문자열 라벨을 정수로 변환)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return X, y, label_encoder.classes_


# numpy 데이터를 PyTorch Tensor로 변환 (NaN을 -1로 대체)
def to_tensor(X, y):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    X = torch.nan_to_num(X, nan=-1.0)  # 인식 안 된 손은 -1로 채움 (predict에서도 동일하게 -1 사용)
    return X, y


# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)  # 초기 은닉 상태는 기본값(0)
        out = out[:, -1, :]  # 마지막 time step의 출력 사용
        out = self.dropout(out)
        out = self.fc(out)
        return out


# 모델 학습
def train_model(model, X_train, y_train):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        # 순전파
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


# 모델 평가
def evaluate_model(model, X_test, y_test, classes):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(classification_report(y_test, predicted, target_names=classes))


# 가중치와 함께 예측에 필요한 설정(라벨 목록 등)을 한 파일로 저장
def save_model(model, classes, input_size, save_path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": list(classes),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "timesteps": timesteps,
    }, save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    csv_path = os.path.join(base_dir, "data", "processed", "features.csv")
    save_path = os.path.join(base_dir, "outputs", "models", "lstm_e500.pth")

    X, y, classes = load_data(csv_path)

    # 학습 및 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    X_train, y_train = to_tensor(X_train, y_train)
    X_test, y_test = to_tensor(X_test, y_test)

    input_size = X_train.shape[2]
    model = LSTMModel(input_size, hidden_size, num_layers, len(classes), dropout_prob)

    train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test, classes)
    save_model(model, classes, input_size, save_path)
