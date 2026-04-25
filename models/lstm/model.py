# =============================================================================
# models/lstm/model.py  (FIXED v9)
#
# Fixes vs v8:
#   1. ARCHITECTURE SHRUNK: hidden 128→64, layers 2→1.
#      For 40 NIFTY50 stocks × ~2700 days, a 2-layer 128-unit LSTM has
#      ~800K trainable parameters. The train set has ~57K sequences of
#      length 15 — roughly 14× parameter-to-sample ratio. That's why
#      val_loss was diverging at epoch 3. 1-layer 64-unit has ~100K params
#      (570× better ratio).
#
#   2. BATCH NORM: Replaced with LayerNorm (more stable for variable-length
#      time-series; BN depends on batch statistics which fluctuate at end
#      of epoch when batch sizes vary).
#
#   3. ATTENTION KEPT: Attention pooling is beneficial — it lets the model
#      focus on the most informative timestep in the 15-day window.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from config.settings import LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT


class LSTMClassifier(nn.Module):
    """
    Lightweight attention-pooled LSTM for 5-day direction classification.

    Architecture:
      1. LayerNorm normalizes each time-step's features.
      2. Single LSTM layer processes the sequence (64 hidden units).
      3. Learned attention pools across time steps.
      4. Dropout → Linear classification head.

    Deliberate design choices:
      - Single layer: prevents overfitting on 40-stock universe.
      - LayerNorm not BatchNorm: stable with variable-size batches.
      - No residual projection: only beneficial when input_size ≈ hidden_size.
    """

    def __init__(self, input_size, hidden_size=LSTM_HIDDEN,
                 num_layers=LSTM_LAYERS, dropout=LSTM_DROPOUT, num_classes=2):
        super().__init__()

        # Normalize each feature across the time dimension
        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention: scalar weight per time-step
        attn_hidden = max(hidden_size // 2, 16)
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_norm(x)           # LayerNorm over features per timestep
        out, _ = self.lstm(x)            # (batch, seq_len, hidden)

        # Attention pooling
        weights = torch.softmax(self.attn(out), dim=1)  # (batch, seq_len, 1)
        h = (out * weights).sum(dim=1)                  # (batch, hidden)

        h = self.dropout(h)
        return self.fc(h)                               # (batch, num_classes)
