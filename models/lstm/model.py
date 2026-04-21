import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch.nn as nn
from config.settings import LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=LSTM_HIDDEN,
                 num_layers=LSTM_LAYERS, dropout=LSTM_DROPOUT, num_classes=3):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))