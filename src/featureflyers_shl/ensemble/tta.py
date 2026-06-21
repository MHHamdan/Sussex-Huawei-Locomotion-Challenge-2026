"""
Test-time augmentation for (B, C, T) IMU tensors.

All functions operate on torch.Tensors already on the target device.
Normalization is always applied first (same order as training); augmentation
is applied on top of the normalized signal so it perturbs the z-scored values.
"""

from __future__ import annotations
import torch


def norm_perwindow(x: torch.Tensor) -> torch.Tensor:
    """Per-window z-score: zero-mean, unit-std each (channel, window) pair."""
    x = x.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    mu  = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (x - mu) / std


def augment_batch(
    x: torch.Tensor,
    jitter_std: float = 0.02,
    scale_lo: float   = 0.9,
    scale_hi: float   = 1.1,
) -> torch.Tensor:
    """
    In-place-safe batch augmentation for TTA.

    1. Gaussian jitter  : add i.i.d. N(0, jitter_std) to every element.
    2. Per-channel scale: multiply each (sample, channel) by U(scale_lo, scale_hi).

    Both transforms are applied after norm_perwindow so the noise is relative
    to the normalised magnitude — consistent with IMUFormer training (--augment).
    """
    if jitter_std > 0.0:
        x = x + torch.randn_like(x) * jitter_std
    if scale_lo != 1.0 or scale_hi != 1.0:
        B, C, T = x.shape
        scale = x.new_empty(B, C, 1).uniform_(scale_lo, scale_hi)
        x = x * scale
    return x
