import torch
import torch.nn as nn

from src.registry import register_model


class SignalEmbedding(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(seq_len, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))

    def forward(self, x: torch.Tensor):
        B, L, F = x.shape
        x = x.permute(0, 2, 1)  # (B, F, L)
        embeds = [self.proj(x[:, i, :])[:, None, :] for i in range(F)]
        tokens = torch.cat(embeds, dim=1)
        cls = self.cls_token.expand(B, -1, -1).to(x.device)
        pos = self.pos_embed.to(x.device)
        return torch.cat([cls, tokens], dim=1) + pos


class RadiantTransformer(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int = 768, num_heads: int = 12, depth: int = 6, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(embed_dim),
                "attn": nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                "norm2": nn.LayerNorm(embed_dim),
                "mlp": nn.Sequential(
                    nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                    nn.Dropout(dropout),
                ),
            }))
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, seq_len),
        )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x_norm = layer["norm1"](x)
            attn_out, _ = layer["attn"](x_norm, x_norm, x_norm)
            x = x + attn_out
            x = x + layer["mlp"](layer["norm2"](x))
        cls_feat = x[:, 0, :]
        return self.head(cls_feat).unsqueeze(-1).squeeze(-1)


@register_model(
    name="transformer",
    default_params={"dropout": 0.1},
    description="RadiantModel (ViT-based) for rPPG to PPG conversion",
)
class RadiantModel(nn.Module):
    def __init__(self, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = SignalEmbedding(seq_len, dropout=dropout)
        self.transformer = RadiantTransformer(seq_len, dropout=dropout)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        return self.transformer(x)
