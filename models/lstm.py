import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)  # (B, L, H)
        out = self.fc(lstm_out)     # (B, L, 1)
        return out.squeeze(-1)
