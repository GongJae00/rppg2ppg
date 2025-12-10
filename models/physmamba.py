import torch
import torch.nn as nn
from mamba_ssm import Mamba


class BiMambaBlock(nn.Module):
    def __init__(self, d_model: int = 128, d_state: int = 32, d_conv: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_f = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.mamba_b = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor):
        x_norm = self.norm(x)
        x_rev = torch.flip(x_norm, dims=[1])
        f_out = self.mamba_f(x_norm)
        b_out = self.mamba_b(x_rev)
        b_out = torch.flip(b_out, dims=[1])
        fused = torch.cat([f_out, b_out], dim=-1)
        gated = self.gate(fused)
        return x + self.dropout(gated)


class PhysMambaModel(nn.Module):
    def __init__(self, d_model: int = 128, d_state: int = 32, d_conv: int = 8, n_blocks: int = 6, dropout: float = 0.1, conv_kernel: int = 7):
        super().__init__()
        self.pre_conv = nn.Conv1d(1, d_model, kernel_size=conv_kernel, padding=conv_kernel // 2)
        self.pre_act = nn.GELU()

        self.blocks = nn.ModuleList([
            BiMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, dropout=dropout)
            for _ in range(n_blocks)
        ])

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model // 4, 1),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, d_model, 1),
            nn.Sigmoid(),
        )

        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x = self.pre_act(self.pre_conv(x))
        x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)

        x_att = x.permute(0, 2, 1)
        attn = self.se(x_att)
        x = (x_att * attn).permute(0, 2, 1)

        x = self.output_proj(x)
        return x.squeeze(-1)


def neg_pearson_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
    y_pred = y_pred - y_pred.mean(dim=1, keepdim=True)
    y_true = y_true - y_true.mean(dim=1, keepdim=True)
    num = (y_pred * y_true).sum(dim=1)
    den = torch.sqrt((y_pred ** 2).sum(dim=1) * (y_true ** 2).sum(dim=1)) + 1e-8
    corr = num / den
    return 1 - corr.mean()
