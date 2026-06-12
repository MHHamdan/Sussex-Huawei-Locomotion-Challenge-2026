"""
1D CNN baseline for SHL 2026 locomotion classification.

Input  : (batch, 9, 500)  — 9 sensor channels × 500 time steps
Output : (batch, 8)       — logits over 8 locomotion classes

Architecture
------------
Four conv blocks (Conv1d → BatchNorm1d → ReLU → optional MaxPool → Dropout),
followed by global average pooling and a linear classifier.

Channel progression: 9 → 32 → 64 → 128 → 256
Temporal resolution: 500 → 250 → 125 → 25 → 1 (via pooling + adaptive avg)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(
    in_ch: int,
    out_ch: int,
    kernel: int,
    dropout: float,
    pool: int | None,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool1d(pool))
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class CNN1D(nn.Module):
    """
    Parameters
    ----------
    n_channels : number of input sensor channels (9 for SHL)
    n_classes  : number of output classes (8 for SHL)
    dropout    : dropout probability applied after each block (except block 1)
    """

    def __init__(
        self,
        n_channels: int = 9,
        n_classes: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1: wide receptive field, no dropout
            _conv_block(n_channels, 32,  kernel=7, dropout=0.0,     pool=2),   # → 250
            # Block 2
            _conv_block(32,         64,  kernel=5, dropout=dropout,  pool=2),   # → 125
            # Block 3
            _conv_block(64,         128, kernel=3, dropout=dropout,  pool=5),   # →  25
            # Block 4: deep features, no further pooling
            _conv_block(128,        256, kernel=3, dropout=dropout,  pool=None),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)   # (B, 256, T) → (B, 256, 1)
        self.classifier = nn.Linear(256, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 9, 500) → logits (B, 8)."""
        x = self.encoder(x)           # (B, 256, 25)
        x = self.pool(x).squeeze(-1)  # (B, 256)
        return self.classifier(x)     # (B, 8)

    @staticmethod
    def n_params(model: "CNN1D") -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
