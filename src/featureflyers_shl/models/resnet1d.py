"""
1D ResNet for SHL 2026 locomotion classification.

Input  : (batch, 9, 500)  — 9 sensor channels × 500 time steps
Output : (batch, 8)       — logits over 8 locomotion classes

Architecture:
  4 residual stages with progressively wider channels and stride-2 downsampling.
  Conv1d residual blocks with BatchNorm + ReLU.
  Global average pooling → Dropout → Linear(512, n_classes).

Padding rule: kernel_size must be odd; padding = (kernel_size-1)//2 gives
exact same-length output for stride=1 and ceil(L/2) for stride=2,
matching the kernel_size=1 shortcut on any input length.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ResBlock1d(nn.Module):
    """Two Conv1d layers with BatchNorm, ReLU, and residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for same-length output"
        pad = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1,
                               padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)

        # Shortcut: 1×1 conv with same stride, no extra padding needed
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        sc  = self.shortcut(x)
        # Guard against ±1 length mismatch from integer division on odd lengths
        min_len = min(out.shape[-1], sc.shape[-1])
        return self.relu(out[..., :min_len] + sc[..., :min_len])


class ResNet1D(nn.Module):
    """
    Four-stage 1D ResNet.

    Channels: 9 → f → 2f → 4f → 8f  (default f=64 → 64/128/256/512)
    Temporal : 500 → 500 → 250 → 125 → 63  (stride-2 at stages 2–4)
    """

    def __init__(
        self,
        n_channels: int = 9,
        n_classes: int = 8,
        base_filters: int = 64,
        kernel_size: int = 7,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        f   = base_filters
        pad = (kernel_size - 1) // 2

        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, f, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(f),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            _ResBlock1d(f,     f,     kernel_size, stride=1),
            _ResBlock1d(f,     f,     kernel_size, stride=1),
        )
        self.stage2 = nn.Sequential(
            _ResBlock1d(f,     f * 2, kernel_size, stride=2),
            _ResBlock1d(f * 2, f * 2, kernel_size, stride=1),
        )
        self.stage3 = nn.Sequential(
            _ResBlock1d(f * 2, f * 4, kernel_size, stride=2),
            _ResBlock1d(f * 4, f * 4, kernel_size, stride=1),
        )
        self.stage4 = nn.Sequential(
            _ResBlock1d(f * 4, f * 8, kernel_size, stride=2),
            _ResBlock1d(f * 8, f * 8, kernel_size, stride=1),
        )

        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(f * 8, n_classes)

        self._init_weights()
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.head(x)
