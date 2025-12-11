import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaModel(nn.Module):
    # NOTE: Origin 코드에서는 d_conv=8 이었으나, 현재 환경의 causal_conv1d CUDA 커널이
    #       width 2~4만 지원하여 d_conv=4로 변경함. (TODO: 환경 업그레이드 후 d_conv=8로 복원 검토)
    def __init__(self, d_model: int = 128, d_state: int = 32, d_conv: int = 4, n_blocks: int = 6, dropout: float = 0.1, conv_kernel: int = 7):
        super().__init__()
        if not 2 <= d_conv <= 4:
            # causal_conv1d CUDA 커널이 width 2~4만 지원한다.
            raise ValueError(f"d_conv must be between 2 and 4 for causal_conv1d, got {d_conv}")        
        self.pre_conv = nn.Conv1d(1, d_model, kernel_size=conv_kernel, padding=conv_kernel // 2)
        self.pre_act = nn.GELU()

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(d_model),
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2),
                nn.Dropout(dropout),
            ))

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
