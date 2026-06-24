"""
Multi-View Position Fusion Network (MVPF) for SHL 2026.

Key insight: all four sensor positions (Bag, Hand, Hips, Torso) are recorded
simultaneously.  Processing them jointly lets the model learn cross-position
correlations — e.g. "high Bag vibration + smooth Torso motion → Car" — that
single-position models with post-hoc pooling can never see.

Architecture
------------
  Input : (B, 4, 9, 500)   — batch × positions × channels × timesteps

  1. PositionEncoder (shared weights across all 4 positions):
       (B×4, 9, 500) → 3-stage compact ResNet1D → GlobalAvgPool → (B×4, feat_dim)

  2. Reshape + learned position embeddings:
       (B, 4, feat_dim)  — position embeddings teach the transformer which
                            sensor location each token comes from

  3. CrossPositionTransformer (2-layer):
       (B, 4, feat_dim) → MHA + FFN × 2 → mean-pool → (B, feat_dim)

  4. Head:
       Dropout → Linear(feat_dim, n_classes)

Parameter count: ≈ 1.45 M  (compact; ResNet1D stage 15 had 8.7 M)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared per-position encoder
# ---------------------------------------------------------------------------

class _ResBlock1d(nn.Module):
    """Two Conv1d layers with BN, ReLU, and residual shortcut (odd kernel only)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 1) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        pad = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        sc  = self.shortcut(x)
        L   = min(out.shape[-1], sc.shape[-1])
        return self.relu(out[..., :L] + sc[..., :L])


class PositionEncoder(nn.Module):
    """
    Compact 3-stage ResNet1D.  Shared across all sensor positions.

    9 channels × 500 timesteps → 256-d feature vector.
    Intentionally shallower than Stage-15 ResNet1D (8.7 M params) because
    it runs 4× per sample.
    """

    def __init__(
        self,
        n_channels: int = 9,
        feat_dim: int = 256,
        base_filters: int = 64,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        f   = base_filters
        pad = (kernel_size - 1) // 2

        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, f, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(f),
            nn.ReLU(inplace=True),
        )
        self.stage1 = _ResBlock1d(f,     f,     kernel_size, stride=1)
        self.stage2 = _ResBlock1d(f,     f * 2, kernel_size, stride=2)
        self.stage3 = _ResBlock1d(f * 2, f * 4, kernel_size, stride=2)

        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.proj    = nn.Linear(f * 4, feat_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 9, 500) → (B, feat_dim)"""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(self.dropout(x))


# ---------------------------------------------------------------------------
# Cross-position transformer
# ---------------------------------------------------------------------------

class _TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → MHA → residual, LN → FFN → residual."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class CrossPositionTransformer(nn.Module):
    """
    Transformer encoder over the 4 position tokens.

    Learned position embeddings (not temporal) distinguish which sensor
    location each token came from before cross-attention runs.
    """

    def __init__(
        self,
        n_positions: int = 4,
        feat_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_mult: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, n_positions, feat_dim) * 0.02)
        self.blocks    = nn.ModuleList([
            _TransformerBlock(feat_dim, n_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 4, feat_dim) → (B, feat_dim)"""
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x.mean(dim=1)   # average over 4 position tokens → (B, feat_dim)


# ---------------------------------------------------------------------------
# Full MVPF model
# ---------------------------------------------------------------------------

class MVPF(nn.Module):
    """
    Multi-View Position Fusion Network.

    Takes all 4 sensor positions simultaneously; learns which cross-position
    patterns discriminate locomotion modes.
    """

    def __init__(
        self,
        n_positions: int = 4,
        n_channels: int = 9,
        n_classes: int = 8,
        feat_dim: int = 256,
        base_filters: int = 64,
        kernel_size: int = 7,
        n_heads: int = 4,
        n_tf_layers: int = 2,
        dropout: float = 0.3,
        encoder_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_positions = n_positions
        self.feat_dim    = feat_dim

        self.encoder = PositionEncoder(
            n_channels=n_channels, feat_dim=feat_dim,
            base_filters=base_filters, kernel_size=kernel_size,
            dropout=encoder_dropout,
        )
        self.fusion = CrossPositionTransformer(
            n_positions=n_positions, feat_dim=feat_dim,
            n_heads=n_heads, n_layers=n_tf_layers,
            ffn_mult=2, dropout=dropout * 0.5,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_classes),
        )
        nn.init.xavier_uniform_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)

        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, P, C, T)  where P=4 positions, C=9 channels, T=500 timesteps
        returns: (B, n_classes) logits
        """
        B, P, C, T = x.shape
        x     = x.view(B * P, C, T)          # (B×P, C, T)
        feats = self.encoder(x)               # (B×P, feat_dim)
        feats = feats.view(B, P, -1)          # (B, P, feat_dim)
        fused = self.fusion(feats)            # (B, feat_dim)
        return self.head(fused)               # (B, n_classes)
