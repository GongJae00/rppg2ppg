import torch
import torch.nn as nn
import torch.nn.functional as F


def disentangle(signal: torch.Tensor):
    """Split signal into normalized trend and amplitude."""
    s = signal.squeeze(-1)
    diff = s[:, 1:] - s[:, :-1]
    trend = diff / torch.sqrt(diff ** 2 + 1.0)
    amp = diff.abs()
    trend = torch.cat([trend, trend[:, -1:]], dim=1)
    amp = torch.cat([amp, amp[:, -1:]], dim=1)
    return trend.unsqueeze(-1), amp.unsqueeze(-1)


class PhysDiffModel(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        # x: (B, L, 1)
        trend, amp = disentangle(x)
        cond = torch.cat([x, trend, amp], dim=-1)  # (B, L, 3)
        h = F.relu(self.input_proj(cond))
        h = self.transformer(h)
        out = self.output_fc(h).squeeze(-1)        # (B, L)
        return out
