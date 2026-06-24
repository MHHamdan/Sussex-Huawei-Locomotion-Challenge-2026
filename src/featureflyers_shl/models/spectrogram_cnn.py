"""
Spectrogram CNN for SHL 2026 locomotion classification.

Pipeline
--------
1. torch.stft applied to all channels in a vectorised batch (no Python loops)
2. Log-magnitude spectrograms: (B, 9, n_freq, n_frames)
3. Modified ResNet-18 (first Conv2d accepts 9 channels instead of 3)
4. Global avg pool → Dropout → Linear(512, n_classes)

STFT parameters for 500-sample windows:
  n_fft=64  (win_length=64, center=True)
    → n_freq  = n_fft // 2 + 1 = 33
    → n_frames = ceil(500 / 16)  = 32   (hop_length=16)

Why spectrograms?
  Bus (~20 Hz engine rumble), Train (rail-periodicity ~50–100 Hz) and Metro
  (distinct tunnel resonance) are separable in frequency space where their raw
  time-domain acceleration profiles look nearly identical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_spectrograms(
    x: torch.Tensor,
    n_fft: int = 64,
    hop_length: int = 16,
    win_length: int = 64,
    n_frames: int = 32,
) -> torch.Tensor:
    """
    Vectorised batched STFT: no Python loop over batch items.

    Parameters
    ----------
    x : (B, C, T)  float32  raw windows, channels-first
    Returns
    -------
    log_mag : (B, C, n_freq, n_frames)  float32
    """
    B, C, T = x.shape
    n_freq = n_fft // 2 + 1   # 33

    window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)

    # Reshape to (B*C, T) for a single batched stft call
    x_flat = x.reshape(B * C, T)

    # torch.stft with center=True pads n_fft//2 each side → n_frames = ceil(T/hop_length)
    S = torch.stft(
        x_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
        onesided=True,
    )  # (B*C, n_freq, n_frames_actual)

    # Crop or pad to exact n_frames for consistent tensor shape
    nf_actual = S.shape[-1]
    if nf_actual >= n_frames:
        S = S[:, :, :n_frames]
    else:
        S = F.pad(S, (0, n_frames - nf_actual))

    log_mag = torch.log1p(S.abs())          # (B*C, n_freq, n_frames)
    return log_mag.reshape(B, C, n_freq, n_frames)


class SpectrogramCNN(nn.Module):
    """
    Modified ResNet-18 accepting 9-channel log-spectrogram input.

    Input : (B, 9, 500)  raw windows channels-first
    Output: (B, n_classes) logits
    """

    def __init__(
        self,
        n_channels: int = 9,
        n_classes: int = 8,
        n_fft: int = 64,
        hop_length: int = 16,
        win_length: int = 64,
        n_frames: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_frames   = n_frames

        from torchvision.models import resnet18
        backbone = resnet18(weights=None)

        # First conv: change from 3 → n_channels input channels, same spatial settings
        backbone.conv1 = nn.Conv2d(
            n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")

        # Feature extractor = everything up to (and including) avgpool
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,   # AdaptiveAvgPool2d(1, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(512, n_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 9, 500) raw z-scored windows."""
        specs = compute_spectrograms(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_frames=self.n_frames,
        )                               # (B, 9, 33, 32)
        feats = self.features(specs)    # (B, 512, 1, 1)
        feats = feats.flatten(1)        # (B, 512)
        return self.head(self.dropout(feats))
