"""
Loss functions used across models.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysMambaSSSDLoss(nn.Module):
    """
    PhysMamba SSSD loss: L_time + L_freq.

    - L_time: negative Pearson correlation
    - L_freq: magnitude MSE in frequency domain
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred - pred.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)

        num = (pred * target).sum(dim=1)
        den = torch.sqrt((pred ** 2).sum(dim=1) * (target ** 2).sum(dim=1) + self.eps)
        time_loss = 1.0 - (num / den).mean()

        pred_fft_in = pred
        target_fft_in = target
        if pred_fft_in.dtype == torch.bfloat16:
            pred_fft_in = pred_fft_in.float()
            target_fft_in = target_fft_in.float()
        pred_fft = torch.fft.rfft(pred_fft_in, dim=1)
        target_fft = torch.fft.rfft(target_fft_in, dim=1)
        freq_loss = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))

        return time_loss + freq_loss
