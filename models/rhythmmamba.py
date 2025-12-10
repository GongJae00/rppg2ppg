import torch
import torch.nn as nn
from mamba_ssm import Mamba


class RhythmMambaModel(nn.Module):
    def __init__(self, d_model: int = 160, d_state: int = 64, d_conv: int = 12, n_blocks: int = 8, dropout: float = 0.1, conv_kernel: int = 9):
        super().__init__()
        self.pre_conv = nn.Conv1d(1, d_model, kernel_size=conv_kernel, padding=conv_kernel // 2)
        self.pre_act = nn.GELU()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2),
                nn.Dropout(dropout),
            )
            for _ in range(n_blocks)
        ])
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x = self.pre_conv(x)
        x = self.pre_act(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            res = x
            x = block(x)
            x = x + res
        x = self.output_proj(x)
        return x.squeeze(-1)
