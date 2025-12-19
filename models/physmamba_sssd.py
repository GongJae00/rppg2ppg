"""
PhysMamba_SSSD: Synergistic State Space Duality Model for Remote Physiological Measurement.

Paper: https://arxiv.org/abs/2408.01077
This implementation keeps the rppg2ppg 1D rPPG->PPG pipeline and adapts
Frame Stem, SSSD, MQ, FDF, and rPPG Predictor to 1D signals.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from src.registry import register_model


class ChannelAttention1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x))


class Stem1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrameStem1D(nn.Module):
    """
    Frame Stem adapted to 1D signals.

    X_stem = Stem(X_origin) + Stem(X_origin + X_diff)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 64, kernel_size: int = 7):
        super().__init__()
        self.stem = Stem1D(in_channels, out_channels, kernel_size=kernel_size)
        self.attn = ChannelAttention1D(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, 1, L)
        x_diff = x[:, :, 1:] - x[:, :, :-1]
        x_diff = F.pad(x_diff, (1, 0), mode="replicate")
        x_stem = self.stem(x) + self.stem(x + x_diff)
        x_stem = x_stem * self.attn(x_stem)
        return x_stem.transpose(1, 2)


class MultiScaleQuery(nn.Module):
    def __init__(self, d_model: int, scales: tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.scales = scales
        self.proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in scales])
        self.fuse = nn.Linear(d_model * len(scales), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length, dim = x.shape
        queries = []
        for scale, proj in zip(self.scales, self.proj):
            if scale <= 1 or length < scale:
                q = proj(x)
            else:
                x_down = F.avg_pool1d(x.transpose(1, 2), kernel_size=scale, stride=scale)
                x_down = x_down.transpose(1, 2)
                q_down = proj(x_down)
                q = F.interpolate(
                    q_down.transpose(1, 2),
                    size=length,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            queries.append(q)
        return self.fuse(torch.cat(queries, dim=-1))


class StructuredMask(nn.Module):
    def __init__(self, num_heads: int, init_tau: float = 16.0):
        super().__init__()
        self.log_tau = nn.Parameter(torch.log(torch.full((num_heads,), init_tau)))

    def forward(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        idx = torch.arange(length, device=device)
        dist = (idx.view(-1, 1) - idx.view(1, -1)).clamp(min=0).float()
        tau = torch.exp(self.log_tau).view(-1, 1, 1)
        mask = torch.exp(-dist / tau)
        return mask.to(dtype)


class StructuredSSD(nn.Module):
    def __init__(self, d_model: int, head_dim: int = 16, dropout: float = 0.1, normalize: bool = True):
        super().__init__()
        if d_model % head_dim != 0:
            raise ValueError(f"d_model must be divisible by head_dim, got {d_model} and {head_dim}")
        self.num_heads = d_model // head_dim
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.normalize = normalize

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.mask = StructuredMask(self.num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, q_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, length, dim = x.shape
        q = q_override if q_override is not None else self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, length, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = self.mask(length, device=x.device, dtype=scores.dtype)
        scores = scores * mask.unsqueeze(0)
        if self.normalize:
            denom = mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            scores = scores / denom.unsqueeze(0)

        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(bsz, length, dim)
        out = self.out_proj(out)
        return self.dropout(out)


class SSSDBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        head_dim: int,
        mq_scales: tuple[int, ...],
        dropout: float,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mq = MultiScaleQuery(d_model, scales=mq_scales)
        self.ssm = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.ssd = StructuredSSD(d_model, head_dim=head_dim, dropout=dropout)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, q_override: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm(x)
        q = q_override if q_override is not None else self.mq(x_norm)
        ssm_out = self.ssm(x_norm)
        ssd_out = self.ssd(x_norm, q_override=q)
        gate = self.gate(torch.cat([ssm_out, ssd_out], dim=-1))
        fused = gate * ssm_out + (1.0 - gate) * ssd_out
        x = x + self.dropout(fused)
        return x, q


class SelfAttentionPathway(nn.Module):
    def __init__(
        self,
        depth: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        head_dim: int,
        mq_scales: tuple[int, ...],
        dropout: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SSSDBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    head_dim=head_dim,
                    mq_scales=mq_scales,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        queries = []
        for block in self.blocks:
            x, q = block(x)
            queries.append(q)
        return x, queries


class CrossAttentionPathway(nn.Module):
    def __init__(
        self,
        depth: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        head_dim: int,
        mq_scales: tuple[int, ...],
        dropout: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SSSDBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    head_dim=head_dim,
                    mq_scales=mq_scales,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, queries: list[torch.Tensor]) -> torch.Tensor:
        for idx, block in enumerate(self.blocks):
            q = queries[idx] if idx < len(queries) else None
            x, _ = block(x, q_override=q)
        return x


class FrequencyDomainFeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        self.scale = 0.02
        self.dim = dim * mlp_ratio
        self.r = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.i = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.rb = nn.Parameter(self.scale * torch.randn(self.dim))
        self.ib = nn.Parameter(self.scale * torch.randn(self.dim))

        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Sequential(
            nn.Conv1d(dim, dim * mlp_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm1d(dim * mlp_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(dim * mlp_ratio, dim, 1, 1, 0, bias=False),
            nn.BatchNorm1d(dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)

        fft_in = x
        if fft_in.dtype == torch.bfloat16:
            fft_in = fft_in.float()
        x_fre = torch.fft.fft(fft_in, dim=1, norm="ortho")
        x_real = F.relu(
            torch.einsum("bnc,cc->bnc", x_fre.real, self.r)
            - torch.einsum("bnc,cc->bnc", x_fre.imag, self.i)
            + self.rb
        )
        x_imag = F.relu(
            torch.einsum("bnc,cc->bnc", x_fre.imag, self.r)
            + torch.einsum("bnc,cc->bnc", x_fre.real, self.i)
            + self.ib
        )

        x_fre = torch.stack([x_real, x_imag], dim=-1).float()
        x_fre = torch.view_as_complex(x_fre)
        x = torch.fft.ifft(x_fre, dim=1, norm="ortho").real
        x = x.to(torch.float32)
        if x.dtype != residual.dtype:
            x = x.to(residual.dtype)
        x = self.fc2(x.transpose(1, 2)).transpose(1, 2)
        return residual + self.dropout(x)


@register_model(
    name="physmamba_sssd",
    requires="mamba-ssm",
    default_params={
        "d_model": 64,
        "d_state": 64,
        "d_conv": 4,
        "head_dim": 16,
        "n_blocks": 24, # 12,8,6 조절
        "dropout": 0.1,
        "stem_kernel": 7,
        "mq_scales": (1, 2, 4, 8),
    },
    description="PhysMamba_SSSD (dual-pathway SSSD + FDF) for rPPG to PPG conversion",
)
class PhysMamba_SSSD(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 64,
        d_conv: int = 4,
        head_dim: int = 16,
        n_blocks: int = 24,
        dropout: float = 0.1,
        stem_kernel: int = 7,
        mq_scales: tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        if not 2 <= d_conv <= 4:
            raise ValueError(f"d_conv must be between 2 and 4 for causal_conv1d, got {d_conv}")
        if d_model % head_dim != 0:
            raise ValueError(f"d_model must be divisible by head_dim, got {d_model} and {head_dim}")

        self.frame_stem = FrameStem1D(in_channels=1, out_channels=d_model, kernel_size=stem_kernel)
        self.sa_path = SelfAttentionPathway(
            depth=n_blocks,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            head_dim=head_dim,
            mq_scales=mq_scales,
            dropout=dropout,
        )
        self.ca_path = CrossAttentionPathway(
            depth=n_blocks,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            head_dim=head_dim,
            mq_scales=mq_scales,
            dropout=dropout,
        )

        self.fdf_sa = FrequencyDomainFeedForward(dim=d_model, mlp_ratio=2, dropout=dropout)
        self.fdf_ca = FrequencyDomainFeedForward(dim=d_model, mlp_ratio=2, dropout=dropout)
        self.predictor = nn.Conv1d(2 * d_model, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frame_stem(x)

        sa_out, queries = self.sa_path(x)
        ca_out = self.ca_path(x, queries)

        sa_out = self.fdf_sa(sa_out)
        ca_out = self.fdf_ca(ca_out)

        fused = torch.cat([sa_out, ca_out], dim=-1)
        fused = fused.transpose(1, 2)
        out = self.predictor(fused).squeeze(1)
        return out


def neg_pearson_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y_pred = y_pred - y_pred.mean(dim=1, keepdim=True)
    y_true = y_true - y_true.mean(dim=1, keepdim=True)
    num = (y_pred * y_true).sum(dim=1)
    den = torch.sqrt((y_pred ** 2).sum(dim=1) * (y_true ** 2).sum(dim=1) + eps)
    corr = num / den
    return 1.0 - corr.mean()


def frequency_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    pred_in = y_pred
    true_in = y_true
    if pred_in.dtype == torch.bfloat16:
        pred_in = pred_in.float()
        true_in = true_in.float()
    pred_fft = torch.fft.rfft(pred_in, dim=1)
    true_fft = torch.fft.rfft(true_in, dim=1)
    return F.mse_loss(torch.abs(pred_fft), torch.abs(true_fft))


def physmamba_sssd_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return neg_pearson_loss(y_pred, y_true) + frequency_loss(y_pred, y_true)
