"""
Multi-class focal loss with optional per-class alpha weighting and label smoothing.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma : float
        Focusing exponent.  0 recovers standard cross-entropy.
        Typical: 0.5–2.0.  Use 2.0 for severe class imbalance.
    weight : Tensor | None
        Per-class alpha weights, shape (C,).  Equivalent to the ``weight``
        argument of ``nn.CrossEntropyLoss``.  Pass inverse-frequency weights
        to combine alpha-weighting with focal down-weighting.
    label_smoothing : float
        Smoothing coefficient ε ∈ [0, 1).  The per-sample CE target is
        ``(1 - ε) * one_hot + ε / C``, which translates to:
            ce = (1 - ε) * (-log p_t) + ε * mean(-log p_k)
        The focal weight is computed from the un-smoothed p_t.
    reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight: torch.Tensor | None = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (N, C)  raw unnormalised scores (pre-softmax)
        targets : (N,)    integer class indices in [0, C)
        """
        log_prob = F.log_softmax(logits, dim=1)          # (N, C)
        log_pt   = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
        pt       = log_pt.exp()                           # (N,)

        # Focal down-weighting factor
        focal_w = (1.0 - pt).pow(self.gamma)             # (N,)

        # Per-class alpha
        if self.weight is not None:
            alpha_t = self.weight[targets]                # (N,)
            focal_w = focal_w * alpha_t

        # Per-sample CE with optional label smoothing
        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            # Smoothed CE = (1-eps)*(-log_pt) + eps*mean_k(-log_p_k)
            smooth_ce = -log_prob.mean(dim=1)             # (N,)
            ce = (1.0 - eps) * (-log_pt) + eps * smooth_ce
        else:
            ce = -log_pt                                  # (N,)

        loss = focal_w * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return (f"gamma={self.gamma}, label_smoothing={self.label_smoothing}, "
                f"reduction={self.reduction!r}, "
                f"alpha={'yes' if self.weight is not None else 'no'}")


def build_criterion(
    loss_type: str,
    class_weights: str,
    counts: "torch.Tensor | None",
    n_classes: int,
    focal_gamma: float,
    label_smoothing: float,
    device: "torch.device",
) -> nn.Module:
    """
    Factory: build the training loss from CLI arguments.

    Parameters
    ----------
    loss_type      : 'ce' | 'focal'
    class_weights  : 'none' | 'balanced'  (inverse-frequency weighting)
    counts         : per-class sample counts tensor (C,) for balanced weighting
    n_classes      : C
    focal_gamma    : γ for focal loss (ignored when loss_type='ce')
    label_smoothing: ε (applies to both CE and focal)
    device         : target device for weight tensor
    """
    alpha: torch.Tensor | None = None
    if class_weights == "balanced" and counts is not None:
        counts_f = counts.float().to(device)
        inv_freq = counts_f.sum() / (n_classes * counts_f.clamp(min=1.0))
        alpha = inv_freq / inv_freq.sum() * n_classes  # normalised to sum=n_classes

    if loss_type == "focal":
        return FocalLoss(
            gamma=focal_gamma,
            weight=alpha,
            label_smoothing=label_smoothing,
        )

    # Standard cross-entropy
    return nn.CrossEntropyLoss(
        weight=alpha,
        label_smoothing=label_smoothing,
    )
