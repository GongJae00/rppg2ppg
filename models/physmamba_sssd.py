"""
PhysMamba_SSSD: Synergistic State Space Duality Model for Remote Physiological Measurement

논문: "PhysMamba: State Space Duality Model for Remote Physiological Measurement"
      https://arxiv.org/abs/2408.01077

핵심 구성요소:
1. Frame Stem: 시간적 특징 추출
2. Dual-Pathway (SA + CA): Self-Attention 경로 + Cross-Attention 경로
3. SSSD Module: Synergistic State Space Duality with Multi-Scale Query
4. FDF: Frequency Domain Feed-Forward
5. rPPG Predictor: 최종 예측

1D rPPG → PPG 변환에 맞게 적응된 구현
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba

from src.registry import register_model


class FrameStem1D(nn.Module):
    """
    Frame Stem: 입력 신호에서 시간적 특징 추출

    원본은 2D CNN이지만, 1D 신호 처리를 위해 1D CNN으로 적응
    시간적 차이(Temporal Difference)를 계산하여 생리학적 신호 강조
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 64, kernel_size: int = 7):
        super().__init__()
        # Temporal difference convolution
        self.diff_conv = nn.Conv1d(in_channels, out_channels // 2, kernel_size=3, padding=1)

        # Direct feature extraction
        self.feat_conv = nn.Conv1d(in_channels, out_channels // 2, kernel_size=kernel_size, padding=kernel_size // 2)

        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

        # Self-attention for informative region focus
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 1) -> (B, 1, L)
        x = x.permute(0, 2, 1)

        # Temporal difference (frame 간 차이)
        x_diff = x[:, :, 1:] - x[:, :, :-1]
        x_diff = F.pad(x_diff, (1, 0), mode='replicate')  # 길이 맞추기

        # 두 경로의 특징 추출
        diff_feat = self.diff_conv(x_diff)
        direct_feat = self.feat_conv(x)

        # 특징 결합
        feat = torch.cat([diff_feat, direct_feat], dim=1)
        feat = self.act(self.bn(feat))

        # Self-attention weighting
        attn = self.attention(feat).unsqueeze(-1)
        feat = feat * attn

        return feat.permute(0, 2, 1)  # (B, L, C)


class MultiScaleQuery(nn.Module):
    """
    Multi-Scale Query (MQ) Mechanism

    다양한 시간 스케일에서 정보를 처리하여 모션 아티팩트와
    불일치한 조명에 대한 적응성 향상
    """
    def __init__(self, d_model: int, scales: list = [1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.query_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in scales
        ])
        self.fusion = nn.Linear(d_model * len(scales), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        B, L, C = x.shape
        queries = []

        for scale, proj in zip(self.scales, self.query_projections):
            if scale == 1:
                q = proj(x)
            else:
                # Downsample -> project -> upsample
                x_down = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=scale, stride=scale)
                x_down = x_down.permute(0, 2, 1)
                q_down = proj(x_down)
                q = F.interpolate(q_down.permute(0, 2, 1), size=L, mode='linear', align_corners=False)
                q = q.permute(0, 2, 1)
            queries.append(q)

        # Multi-scale fusion
        multi_scale = torch.cat(queries, dim=-1)
        return self.fusion(multi_scale)


class SSSDBlock(nn.Module):
    """
    Synergistic State Space Duality (SSSD) Block

    State space model과 attention mechanism을 통합한 블록
    - State equation: h_t = A * h_{t-1} + B * x_t
    - Output equation: y_t = C^T * h_t
    """
    def __init__(self, d_model: int = 64, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        # State Space Model (Mamba)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # Attention mechanism for duality
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Fusion gate
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)

        # SSM pathway
        ssm_out = self.mamba(x_norm)

        # Attention pathway
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)

        # Gated fusion (State Space Duality)
        combined = torch.cat([ssm_out, attn_out], dim=-1)
        gate = self.gate(combined)
        fused = gate * ssm_out + (1 - gate) * attn_out

        return residual + self.dropout(fused)


class SelfAttentionPathway(nn.Module):
    """
    Self-Attention (SA) Pathway

    개별 특징 내의 자체 의존성에 집중
    """
    def __init__(self, d_model: int, n_blocks: int = 3, d_state: int = 16, d_conv: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mq = MultiScaleQuery(d_model)
        self.blocks = nn.ModuleList([
            SSSDBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, dropout=dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> tuple:
        # Multi-scale query
        query = self.mq(x)

        # Process through SSSD blocks
        for block in self.blocks:
            x = block(x)

        return x, query


class CrossAttentionPathway(nn.Module):
    """
    Cross-Attention (CA) Pathway

    SA 경로로부터 multi-scale query를 받아 특징 간 상호작용 촉진
    """
    def __init__(self, d_model: int, n_blocks: int = 3, d_state: int = 16, d_conv: int = 4, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            SSSDBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # Cross-attention for query from SA pathway
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # Cross-attention with query from SA pathway
        x_norm = self.norm(x)
        x_cross, _ = self.cross_attention(query, x_norm, x_norm)
        x = x + x_cross

        # Process through SSSD blocks
        for block in self.blocks:
            x = block(x)

        return x


class FrequencyDomainFeedForward(nn.Module):
    """
    Frequency Domain Feed-Forward (FDF)

    FFT를 적용하여 시간 신호를 주파수 영역으로 변환
    심박수 신호와 같은 주기적 성분 강조
    """
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        # Frequency domain processing
        self.freq_linear1 = nn.Linear(d_model, d_model * expansion)
        self.freq_linear2 = nn.Linear(d_model * expansion, d_model)

        # Time domain residual
        self.time_linear = nn.Linear(d_model, d_model)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        # FFT: Time -> Frequency domain
        x_freq = torch.fft.rfft(x, dim=1)

        # Process in frequency domain (real and imaginary separately)
        x_freq_real = x_freq.real
        x_freq_imag = x_freq.imag

        # Frequency processing
        x_freq_real = self.freq_linear2(self.dropout(self.act(self.freq_linear1(x_freq_real))))
        x_freq_imag = self.freq_linear2(self.dropout(self.act(self.freq_linear1(x_freq_imag))))

        # Reconstruct complex tensor
        x_freq_processed = torch.complex(x_freq_real, x_freq_imag)

        # IFFT: Frequency -> Time domain
        x_time = torch.fft.irfft(x_freq_processed, n=x.shape[1], dim=1)

        # Time domain residual path
        x_residual = self.time_linear(x)

        return residual + self.dropout(x_time + x_residual)


@register_model(
    name="physmamba_sssd",
    requires="mamba-ssm",
    default_params={"d_model": 64, "d_state": 16, "d_conv": 4, "n_blocks": 3, "dropout": 0.1, "stem_kernel": 7},
    description="PhysMamba_SSSD (Synergistic State Space Duality) for rPPG to PPG conversion",
)
class PhysMamba_SSSD(nn.Module):
    """
    PhysMamba with Synergistic State Space Duality (SSSD)

    이중 경로 시간-주파수 상호작용 모델
    - SA Pathway: 자체 의존성 학습
    - CA Pathway: 경로 간 상호작용
    - FDF: 주파수 영역 강화

    입력: (B, L, 1) - rPPG 신호
    출력: (B, L) - PPG 신호
    """
    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        n_blocks: int = 3,
        dropout: float = 0.1,
        stem_kernel: int = 7
    ):
        super().__init__()

        # Frame Stem: 특징 추출
        self.frame_stem = FrameStem1D(
            in_channels=1,
            out_channels=d_model,
            kernel_size=stem_kernel
        )

        # Dual-Pathway Network
        self.sa_pathway = SelfAttentionPathway(
            d_model=d_model,
            n_blocks=n_blocks,
            d_state=d_state,
            d_conv=d_conv,
            dropout=dropout
        )

        self.ca_pathway = CrossAttentionPathway(
            d_model=d_model,
            n_blocks=n_blocks,
            d_state=d_state,
            d_conv=d_conv,
            dropout=dropout
        )

        # Frequency Domain Feed-Forward
        self.fdf_sa = FrequencyDomainFeedForward(d_model=d_model, dropout=dropout)
        self.fdf_ca = FrequencyDomainFeedForward(d_model=d_model, dropout=dropout)

        # Pathway fusion
        self.fusion_conv = nn.Conv1d(2 * d_model, d_model, kernel_size=1)
        self.fusion_norm = nn.LayerNorm(d_model)

        # rPPG Predictor
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frame Stem
        x = self.frame_stem(x)  # (B, L, d_model)

        # Self-Attention Pathway
        sa_out, query = self.sa_pathway(x)
        sa_out = self.fdf_sa(sa_out)

        # Cross-Attention Pathway (uses query from SA)
        ca_out = self.ca_pathway(x, query)
        ca_out = self.fdf_ca(ca_out)

        # Pathway Fusion
        # Concatenate along channel dimension
        fused = torch.cat([sa_out, ca_out], dim=-1)  # (B, L, 2*d_model)
        fused = fused.permute(0, 2, 1)  # (B, 2*d_model, L)
        fused = self.fusion_conv(fused)  # (B, d_model, L)
        fused = fused.permute(0, 2, 1)  # (B, L, d_model)
        fused = self.fusion_norm(fused)

        # rPPG Prediction
        out = self.predictor(fused)  # (B, L, 1)

        return out.squeeze(-1)  # (B, L)


# Negative Pearson Loss (논문에서 사용하는 시간 영역 손실)
def neg_pearson_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Negative Pearson Correlation Loss

    시간 영역에서 예측값과 실제값의 상관관계 최대화
    """
    y_pred = y_pred - y_pred.mean(dim=1, keepdim=True)
    y_true = y_true - y_true.mean(dim=1, keepdim=True)

    num = (y_pred * y_true).sum(dim=1)
    den = torch.sqrt((y_pred ** 2).sum(dim=1) * (y_true ** 2).sum(dim=1)) + 1e-8

    corr = num / den
    return 1 - corr.mean()


# Frequency Domain Loss (논문에서 사용하는 주파수 영역 손실)
def frequency_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Frequency Domain Loss

    주파수 영역에서 예측값과 실제값의 정렬
    """
    # FFT
    pred_fft = torch.fft.rfft(y_pred, dim=1)
    true_fft = torch.fft.rfft(y_true, dim=1)

    # Magnitude loss
    pred_mag = torch.abs(pred_fft)
    true_mag = torch.abs(true_fft)

    return F.mse_loss(pred_mag, true_mag)


# Combined Loss (시간 + 주파수)
def physmamba_sssd_loss(y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """
    PhysMamba SSSD Combined Loss

    시간 영역 손실과 주파수 영역 손실의 가중 조합
    """
    time_loss = neg_pearson_loss(y_pred, y_true)
    freq_loss = frequency_loss(y_pred, y_true)

    return alpha * time_loss + (1 - alpha) * freq_loss
