"""
InceptionTime for SHL 2026 locomotion classification.

Input  : (batch, 9, 500)  — 9 sensor channels × 500 time steps
Output : (batch, 8)       — logits over 8 locomotion classes

Architecture (Fawaz et al., 2020 — InceptionTime):
  N_MODULES inception modules, grouped into residual blocks of 3.
  Each inception module: 3 parallel Conv1d (kernels 40/20/10) + MaxPool branch,
  concatenated → 4*nb_filters channels, then BN + ReLU.
  Residual shortcut: Conv1d(1x1) + BN when channels change.
  Global average pool → Linear(4*nb_filters, n_classes).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _InceptionModule(nn.Module):
    def __init__(self, in_channels: int, nb_filters: int = 32, bottleneck: int = 32) -> None:
        super().__init__()
        # Bottleneck to reduce channels before parallel convolutions
        use_bottleneck = in_channels > 1
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1, bias=False) if use_bottleneck else None
        conv_in = bottleneck if use_bottleneck else in_channels

        # Parallel convolutions with different receptive fields
        self.conv_large  = nn.Conv1d(conv_in, nb_filters, kernel_size=40, padding=20, bias=False)
        self.conv_medium = nn.Conv1d(conv_in, nb_filters, kernel_size=20, padding=10, bias=False)
        self.conv_small  = nn.Conv1d(conv_in, nb_filters, kernel_size=10, padding=5,  bias=False)

        # MaxPool branch applied to original (not bottlenecked) input
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_mp = nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)

        self.bn   = nn.BatchNorm1d(4 * nb_filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottlenecked = self.bottleneck(x) if self.bottleneck is not None else x
        # Trim to input length (padding may add 1 extra step for even kernels)
        T = x.shape[-1]
        out_large  = self.conv_large(bottlenecked)[..., :T]
        out_medium = self.conv_medium(bottlenecked)[..., :T]
        out_small  = self.conv_small(bottlenecked)[..., :T]
        out_mp     = self.conv_mp(self.maxpool(x))
        out = torch.cat([out_large, out_medium, out_small, out_mp], dim=1)
        return self.relu(self.bn(out))


class InceptionTime(nn.Module):
    """
    Parameters
    ----------
    n_channels  : input sensor channels (9 for SHL)
    n_classes   : output classes (8 for SHL)
    nb_filters  : filters per parallel branch in each inception module
    depth       : total number of inception modules (must be multiple of 3 for residuals)
    bottleneck  : bottleneck width before the parallel convolutions
    dropout     : dropout applied before the final classifier
    """

    def __init__(
        self,
        n_channels: int = 9,
        n_classes:  int = 8,
        nb_filters: int = 32,
        depth:      int = 6,
        bottleneck: int = 32,
        dropout:    float = 0.3,
    ) -> None:
        super().__init__()
        assert depth % 3 == 0, "depth must be divisible by 3 (one residual block per 3 modules)"

        self.modules_list = nn.ModuleList()
        self.residuals    = nn.ModuleList()

        in_ch = n_channels
        out_ch = 4 * nb_filters
        for i in range(depth):
            self.modules_list.append(_InceptionModule(in_ch, nb_filters, bottleneck))
            if (i + 1) % 3 == 0:
                # Residual shortcut: match channels at input of this residual block
                shortcut_in = n_channels if i < 3 else out_ch
                self.residuals.append(nn.Sequential(
                    nn.Conv1d(shortcut_in, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_ch),
                ))
            in_ch = out_ch

        self.gap        = nn.AdaptiveAvgPool1d(1)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_ch, n_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 9, 500) → logits (B, 8)."""
        residual_input = x
        res_idx = 0
        for i, mod in enumerate(self.modules_list):
            x = mod(x)
            if (i + 1) % 3 == 0:
                shortcut = self.residuals[res_idx](residual_input)
                x = torch.relu(x + shortcut)
                residual_input = x
                res_idx += 1
        x = self.gap(x).squeeze(-1)   # (B, 4*nb_filters)
        x = self.dropout(x)
        return self.classifier(x)

    @staticmethod
    def n_params(model: "InceptionTime") -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
