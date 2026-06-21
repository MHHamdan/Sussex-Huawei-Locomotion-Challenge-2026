"""
Temperature scaling for post-hoc classifier calibration.

Given raw logits and ground-truth labels from a validation set, find scalar T > 0
that minimises the negative log-likelihood (NLL).  Then apply softmax(logits / T)
to obtain calibrated probability estimates.

Calibrating before ensembling improves the weighted average because it aligns each
model's confidence scale — an over-confident model no longer dominates the mixture.
"""

from __future__ import annotations
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize_scalar


def find_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Find T ∈ [0.1, 10] that minimises NLL(softmax(logits / T), labels).

    Parameters
    ----------
    logits : (N, K) float32 — raw pre-softmax scores
    labels : (N,)   int    — ground-truth class indices (0-based)

    Returns
    -------
    T : float — optimal temperature (T < 1 sharpens, T > 1 softens)
    """
    def nll(T: float) -> float:
        probs = softmax(logits / T, axis=1)
        probs = np.clip(probs, 1e-12, 1.0)
        return -float(np.mean(np.log(probs[np.arange(len(labels)), labels])))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded",
                             options={"xatol": 1e-4})
    return float(result.x)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Return softmax(logits / T) as float32 array of shape (N, K)."""
    return softmax(logits / T, axis=1).astype(np.float32)
