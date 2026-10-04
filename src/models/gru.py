"""
- GRU 모델: LSTMClassifier에서 순환 층만 nn.GRU로 바꾼 것
"""

import torch.nn as nn

from .lstm import LSTMClassifier


class GRUClassifier(LSTMClassifier):
    rnn = nn.GRU
