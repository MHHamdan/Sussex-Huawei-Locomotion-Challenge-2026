"""
Probability combination and ensemble evaluation utilities.

All functions operate on lists of (N, K) float32 probability arrays
(already calibrated via temperature scaling).
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, classification_report
from scipy.optimize import minimize


def weighted_average(
    probs_list: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """
    Weighted average of probability arrays.

    Parameters
    ----------
    probs_list : list of (N, K) float32 arrays — calibrated softmax probs
    weights    : list of non-negative floats (normalised internally); None = uniform

    Returns
    -------
    (N, K) float32 — averaged probability distribution
    """
    if weights is None:
        weights = [1.0] * len(probs_list)
    w = np.array(weights, dtype=np.float64)
    w /= w.sum()
    out = np.zeros_like(probs_list[0], dtype=np.float64)
    for p, wi in zip(probs_list, w):
        out += p.astype(np.float64) * wi
    return out.astype(np.float32)


def evaluate_probs(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """
    Compute macro-F1 and accuracy from probability array.

    Returns (macro_f1, accuracy).
    """
    preds = probs.argmax(axis=1)
    f1  = float(f1_score(labels, preds, average="macro", zero_division=0))
    acc = float((preds == labels).mean())
    return f1, acc


def find_optimal_weights(
    probs_list: list[np.ndarray],
    labels: np.ndarray,
    n_models: int | None = None,
) -> np.ndarray:
    """
    Find non-negative weights w that maximise val macro-F1 via L-BFGS-B.

    Uses softmax parameterisation (w = softmax(θ)) to enforce w > 0, sum = 1.
    Returns (n_models,) float64 normalised weight vector.
    """
    n = n_models or len(probs_list)

    def neg_f1(theta: np.ndarray) -> float:
        w = np.exp(theta - theta.max())
        w /= w.sum()
        avg = weighted_average(probs_list, weights=w.tolist())
        f1, _ = evaluate_probs(avg, labels)
        return -f1

    theta0 = np.zeros(n)
    res = minimize(neg_f1, theta0, method="L-BFGS-B",
                   options={"maxiter": 200, "ftol": 1e-6})
    w = np.exp(res.x - res.x.max())
    w /= w.sum()
    return w.astype(np.float64)


def print_ensemble_report(
    probs: np.ndarray,
    labels: np.ndarray,
    label_map: dict[int, str],
    title: str = "Ensemble",
) -> None:
    """Print classification report and macro-F1 for a probability array."""
    preds = probs.argmax(axis=1)
    f1, acc = evaluate_probs(probs, labels)
    n_classes = len(label_map)
    target_names = [label_map[i] for i in range(n_classes)]
    print(f"\n{'='*60}")
    print(f"{title}  —  Macro-F1: {f1:.4f}   Accuracy: {acc:.4f}")
    print(f"{'='*60}")
    print(classification_report(
        labels, preds,
        labels=list(range(n_classes)),
        target_names=target_names,
        zero_division=0,
    ))
