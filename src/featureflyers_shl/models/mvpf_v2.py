"""
Multi-View Position Fusion Network v2 (MVPF-v2) for SHL 2026.

Improvements over MVPF v1
--------------------------
1. Temporal attention pooling  — replaces global average pool; the model
   learns *which* 62-step temporal bins are most discriminative rather than
   treating all equally.
2. Deeper encoder (4 ResNet stages vs 3) — extra stride-2 stage gives
   62-step resolution at the bottleneck, halving noise and doubling the
   receptive field.
3. Deeper cross-position transformer (3 layers, 8 heads vs 2/4) — more
   capacity to learn cross-sensor correlations.
4. Gated position fusion — a learned sigmoid gate modulates each position's
   contribution before mean-pooling; useful when one sensor position is
   occluded or corrupted (matches position-dropout training augmentation).

Augmentation (in training script, not here)
-------------------------------------------
- IMU rotation augmentation (channels 0-2 and 3-5)
- Magnitude warping (smooth time-varying amplitude variation)
- Jitter + position dropout (carried over from v1)
- Stochastic Weight Averaging (SWA) over last N epochs

Architecture
------------
  Input : (B, 4, 9, 500)

  1. PositionEncoderV2 (shared):
       Stem: Conv1d(9→f, k=15) + BN + ReLU
       Stage1: ResBlock(f, f, stride=1)
       Stage2: ResBlock(f, 2f, stride=2)  → T/2
       Stage3: ResBlock(2f, 4f, stride=2) → T/4
       Stage4: ResBlock(4f, 4f, stride=2) → T/8  [new]
       TemporalAttnPool(4f) → (B, 4f)
       Dropout + Linear(4f, feat_dim)

  2. CrossPositionTransformerV2:
       learned pos_embed(1, 4, feat_dim)
       3 × PreNormTransformerBlock(feat_dim, 8 heads)
       LayerNorm
       PositionGate (sigmoid gate per position) → weighted mean → (B, feat_dim)

  3. Head: Dropout(dropout) → Linear(feat_dim, n_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ResBlock1d(nn.Module):
    """Conv1d residual block with BN, ReLU, and optional shortcut."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 1) -> None:
        super().__init__()
        assert kernel_size % 2 == 1
        pad = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act   = nn.ReLU(inplace=True)

        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.BatchNorm1d(out_ch))
            if stride != 1 or in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        sc  = self.shortcut(x)
        L   = min(out.shape[-1], sc.shape[-1])
        return self.act(out[..., :L] + sc[..., :L])


class TemporalAttentionPool(nn.Module):
    """
    Learned attention-weighted temporal pooling.

    Computes a per-timestep scalar importance weight via a lightweight linear
    layer, then produces a weighted sum instead of a uniform average.  This
    lets the model focus on the most discriminative temporal segments (e.g.
    a brief transition period) rather than smoothing over the entire window.

    x : (B, C, T) → (B, C)
    """

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(feat_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)"""
        xt   = x.transpose(1, 2)           # (B, T, C)
        w    = self.score(xt)              # (B, T, 1)
        w    = torch.softmax(w, dim=1)    # (B, T, 1)
        out  = (xt * w).sum(dim=1)        # (B, C)
        return out


# ---------------------------------------------------------------------------
# Position encoder v2
# ---------------------------------------------------------------------------

class PositionEncoderV2(nn.Module):
    """
    4-stage compact ResNet1D with temporal attention pooling.

    9 ch × 500 ts → feat_dim-d feature vector.
    Stage depths double the receptive field; 4th stride-2 stage reduces
    the bottleneck to T/8 ≈ 62 time steps (fewer attention keys → less noise).
    """

    def __init__(
        self,
        n_channels: int = 9,
        feat_dim: int = 256,
        base_filters: int = 64,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        f   = base_filters
        pad = (kernel_size - 1) // 2

        self.stem   = nn.Sequential(
            nn.Conv1d(n_channels, f, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(f), nn.ReLU(inplace=True),
        )
        self.stage1 = _ResBlock1d(f,     f,     kernel_size, stride=1)
        self.stage2 = _ResBlock1d(f,     f * 2, kernel_size, stride=2)
        self.stage3 = _ResBlock1d(f * 2, f * 4, kernel_size, stride=2)
        self.stage4 = _ResBlock1d(f * 4, f * 4, kernel_size, stride=2)

        self.attn_pool = TemporalAttentionPool(f * 4)
        self.dropout   = nn.Dropout(dropout)
        self.proj      = nn.Linear(f * 4, feat_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 9, 500) → (B, feat_dim)"""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)                 # (B, 4f, ~62)
        x = self.attn_pool(x)              # (B, 4f)
        return self.proj(self.dropout(x))  # (B, feat_dim)


# ---------------------------------------------------------------------------
# Cross-position transformer v2
# ---------------------------------------------------------------------------

class _TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class CrossPositionTransformerV2(nn.Module):
    """
    3-layer, 8-head transformer over the 4 sensor-position tokens.

    Position gating: after transformer, each of the 4 position tokens is
    scaled by a sigmoid gate.  A gate value near 0 means "this position
    contributes little"; a gate near 1 means "this position is important".
    Combined with position dropout in training, the model learns robust
    multi-view fusion that degrades gracefully when a position is missing.
    """

    def __init__(
        self,
        n_positions: int = 4,
        feat_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, n_positions, feat_dim) * 0.02)
        self.blocks    = nn.ModuleList([
            _TransformerBlock(feat_dim, n_heads, ffn_mult, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(feat_dim)
        # Position gating: project each position token to a scalar gate
        self.gate = nn.Linear(feat_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, P, feat_dim) → (B, feat_dim)"""
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Gated mean pooling: weight each position by its learned importance
        g = torch.sigmoid(self.gate(x))   # (B, P, 1)
        return (x * g).sum(dim=1) / g.sum(dim=1).clamp(min=1e-6)   # (B, feat_dim)


# ---------------------------------------------------------------------------
# Full MVPF v2
# ---------------------------------------------------------------------------

class MVPFv2(nn.Module):
    """
    Multi-View Position Fusion Network v2.

    Processes all 4 sensor positions simultaneously with stronger temporal
    modeling (attention pool), deeper encoder, and gated position fusion.
    """

    def __init__(
        self,
        n_positions: int = 4,
        n_channels: int = 9,
        n_classes: int = 8,
        feat_dim: int = 256,
        base_filters: int = 64,
        kernel_size: int = 7,
        n_heads: int = 8,
        n_tf_layers: int = 3,
        ffn_mult: int = 4,
        dropout: float = 0.4,
        encoder_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_positions = n_positions
        self.feat_dim    = feat_dim

        self.encoder = PositionEncoderV2(
            n_channels=n_channels, feat_dim=feat_dim,
            base_filters=base_filters, kernel_size=kernel_size,
            dropout=encoder_dropout,
        )
        self.fusion = CrossPositionTransformerV2(
            n_positions=n_positions, feat_dim=feat_dim,
            n_heads=n_heads, n_layers=n_tf_layers,
            ffn_mult=ffn_mult, dropout=dropout * 0.5,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_classes),
        )
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, P, C, T)  P=4 positions, C=9 channels, T=500 timesteps
        returns: (B, n_classes) logits
        """
        B, P, C, T = x.shape
        x     = x.view(B * P, C, T)           # (B×P, C, T)
        feats = self.encoder(x)               # (B×P, feat_dim)
        feats = feats.view(B, P, -1)          # (B, P, feat_dim)
        fused = self.fusion(feats)            # (B, feat_dim)
        return self.head(fused)               # (B, n_classes)
