import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from config.settings import LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=LSTM_HIDDEN,
                 num_layers=LSTM_LAYERS, dropout=LSTM_DROPOUT, num_classes=2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        attn_hidden = max(hidden_size // 2, 16)
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.input_norm(x)
        out, _ = self.lstm(x)
        weights = torch.softmax(self.attn(out), dim=1)
        h = (out * weights).sum(dim=1)
        h = self.dropout(h)
        return self.fc(h)
