"""
IMUFormer — Multi-Branch Multi-Scale Transformer for IMU locomotion classification.

Architecture overview
---------------------
Three parallel multi-scale 1-D CNN branches, one per physical sensor group
(Accelerometer, Gyroscope, Magnetometer).  Each branch independently learns
motion patterns at four temporal scales before the three branches are fused
and processed by a lightweight temporal Transformer.

```
Input (B, 9, 500)
      │
  ┌───┴──────────────────────────────────────┐
  │  Split on channel dim                    │
  ├── Acc (B, 3, 500) ──► SensorBranch ──►  │
  ├── Gyr (B, 3, 500) ──► SensorBranch ──►  │  each → (B, D, T')
  └── Mag (B, 3, 500) ──► SensorBranch ──►  │
  └───────────────────────────────────────   │
        │  concat → (B, 3D, T')              │
        │  reshape → (B, T', 3D)             │
        ▼                                    │
  TransformerEncoder (n_layers, 3D, n_heads) │
        │  global avg-pool → (B, 3D)         │
        ▼
  LayerNorm → Dropout → Linear → (B, n_classes)
```

SensorBranch
------------
  Input: (B, in_ch, T)
  Parallel depthwise-separable convs with kernels k ∈ {5, 13, 33, 65}
  → each outputs (B, D//4, T) with same-padding
  → concat → (B, D, T)
  → stride-4 conv → (B, D, T//4)
  → stride-2 conv → (B, D, T//8)       [T//8 ≈ 62 for T=500]

Inductive biases
----------------
- Explicit Acc/Gyr/Mag branch split: physical sensor groups behave differently
  (Acc: gravity + linear motion; Gyr: rotation; Mag: heading)
- Multi-scale kernels: k=5 (50 ms bursts), k=13 (130 ms steps), k=33 (330 ms
  gait sub-cycles), k=65 (650 ms full stride)
- Shared Transformer across time: captures the 5-second rhythm of each mode
- No pre-trained weights: learns everything from the SHL data directly

Parameter count (default d=128, n_tf_layers=2): ~2.8 M params
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Depthwise-separable 1-D conv block
# ---------------------------------------------------------------------------

class _DSConv1d(nn.Module):
    """Depthwise + pointwise conv → BN → GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int) -> None:
        super().__init__()
        pad = kernel // 2   # same-size output
        self.dw  = nn.Conv1d(in_ch, in_ch, kernel, padding=pad, groups=in_ch, bias=False)
        self.pw  = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


# ---------------------------------------------------------------------------
# Per-sensor-group multi-scale branch
# ---------------------------------------------------------------------------

class _SensorBranch(nn.Module):
    """
    Multi-scale feature extractor for one 3-axis sensor group.

    Four parallel depthwise-separable convs (kernels 5, 13, 33, 65) capture
    motion dynamics at 50 ms → 650 ms temporal scales.  Two strided convs
    then downsample T=500 → T//8 ≈ 62 for the Transformer.
    """

    KERNELS = (5, 13, 33, 65)

    def __init__(self, in_ch: int = 3, d: int = 128) -> None:
        super().__init__()
        assert d % len(self.KERNELS) == 0, "d must be divisible by 4"
        d_per = d // len(self.KERNELS)

        self.branches = nn.ModuleList([
            _DSConv1d(in_ch, d_per, k) for k in self.KERNELS
        ])

        # T=500 → 125 (stride 4) → 62 (stride 2)
        self.down1 = nn.Sequential(
            nn.Conv1d(d, d, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm1d(d),
            nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(d, d, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm1d(d),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_ch, T) → (B, d, T//8)"""
        parts = [br(x) for br in self.branches]  # each (B, d//4, T)
        out = torch.cat(parts, dim=1)             # (B, d, T)
        return self.down2(self.down1(out))        # (B, d, T//8)


# ---------------------------------------------------------------------------
# Positional encoding (sinusoidal, fixed)
# ---------------------------------------------------------------------------

class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        return x + self.pe[:, :x.size(1)]


# ---------------------------------------------------------------------------
# IMUFormer
# ---------------------------------------------------------------------------

class IMUFormer(nn.Module):
    """
    Multi-Branch IMU-Former for locomotion classification.

    Parameters
    ----------
    n_classes    : number of output classes (default 8 for SHL)
    d            : per-branch hidden dim; Transformer sees 3*d (default 128 → 384)
    n_heads      : attention heads in Transformer (must divide 3*d)
    n_tf_layers  : number of Transformer encoder layers
    dropout      : applied before the final linear classifier
    in_channels  : total input channels (default 9: Acc+Gyr+Mag, 3 each)
    """

    def __init__(
        self,
        n_classes: int = 8,
        d: int = 128,
        n_heads: int = 4,
        n_tf_layers: int = 2,
        dropout: float = 0.1,
        in_channels: int = 9,
    ) -> None:
        super().__init__()
        assert in_channels % 3 == 0, "in_channels must be divisible by 3 (Acc/Gyr/Mag)"

        ch_per_group = in_channels // 3
        self.acc_branch = _SensorBranch(ch_per_group, d)
        self.gyr_branch = _SensorBranch(ch_per_group, d)
        self.mag_branch = _SensorBranch(ch_per_group, d)

        d_model = 3 * d
        self.pe  = _SinusoidalPE(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,       # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_tf_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        self._d = d
        self._d_model = d_model
        self._n_classes = n_classes
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d) and not isinstance(m, _DSConv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T)  — C=9 channels, T=500 timesteps
        returns (B, n_classes) logits
        """
        B, C, T = x.shape
        ch = C // 3

        acc = self.acc_branch(x[:, :ch,      :])   # (B, d, T')
        gyr = self.gyr_branch(x[:, ch:2*ch,  :])   # (B, d, T')
        mag = self.mag_branch(x[:, 2*ch:,    :])   # (B, d, T')

        # (B, 3d, T') → (B, T', 3d)
        fused = torch.cat([acc, gyr, mag], dim=1).permute(0, 2, 1)

        fused = self.pe(fused)
        fused = self.transformer(fused)              # (B, T', 3d)
        pooled = fused.mean(dim=1)                   # (B, 3d)
        return self.head(pooled)                     # (B, n_classes)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_imu_former(
    n_classes: int = 8,
    d: int = 128,
    n_heads: int = 4,
    n_tf_layers: int = 2,
    dropout: float = 0.1,
) -> IMUFormer:
    return IMUFormer(
        n_classes=n_classes,
        d=d,
        n_heads=n_heads,
        n_tf_layers=n_tf_layers,
        dropout=dropout,
    )
