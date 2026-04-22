import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn
from config.settings import LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT


class LSTMClassifier(nn.Module):
    """
    Improved LSTM:
      - BatchNorm1d on the final hidden state for training stability
      - Separate dropout on LSTM output before fc
      - num_classes=2  (binary: DOWN=0, UP=1)
        FIXED: was 3 — caused silent softmax over 3 classes but only 2 labels exist
    """
    def __init__(self, input_size, hidden_size=LSTM_HIDDEN,
                 num_layers=LSTM_LAYERS, dropout=LSTM_DROPOUT, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bn      = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]          # last time-step
        h = self.bn(h)
        h = self.dropout(h)
        return self.fc(h)