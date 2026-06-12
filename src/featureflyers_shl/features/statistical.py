"""
Statistical and spectral feature extraction for SHL sensor windows.

Input: (N, 9) array of a single window — columns are
    [Acc_x, Acc_y, Acc_z, Gyr_x, Gyr_y, Gyr_z, Mag_x, Mag_y, Mag_z]

Output: 1-D feature vector of length `n_features`.

Call `feature_names()` to get the ordered list of feature names.
"""

from __future__ import annotations

import numpy as np

_SENSORS = ["Acc_x", "Acc_y", "Acc_z", "Gyr_x", "Gyr_y", "Gyr_z",
            "Mag_x", "Mag_y", "Mag_z"]

_SENSOR_GROUPS = {
    "Acc": [0, 1, 2],
    "Gyr": [3, 4, 5],
    "Mag": [6, 7, 8],
}


def _stat_per_axis(w: np.ndarray) -> np.ndarray:
    """Per-axis stats: mean, std, min, max, median, energy, zcr → (9×7,)."""
    feats = []
    for i in range(w.shape[1]):
        x = w[:, i]
        mean = x.mean()
        std = x.std()
        feats.extend([
            mean,
            std,
            x.min(),
            x.max(),
            np.median(x),
            np.mean(x ** 2),                           # energy (mean power)
            np.mean(np.diff(np.sign(x - mean)) != 0),  # zero-crossing rate around mean
        ])
    return np.array(feats, dtype=np.float32)


def _magnitude_stats(w: np.ndarray, indices: list[int]) -> np.ndarray:
    """Stats of the Euclidean magnitude of a 3-axis group → (5,)."""
    axes = w[:, indices]
    mag = np.sqrt((axes ** 2).sum(axis=1))
    return np.array([mag.mean(), mag.std(), mag.min(), mag.max(),
                     np.mean(mag ** 2)], dtype=np.float32)


def extract(window: np.ndarray) -> np.ndarray:
    """
    Extract features from a single window.

    Parameters
    ----------
    window : np.ndarray, shape (W, 9)
        One window of sensor data. W = number of time steps.

    Returns
    -------
    np.ndarray, shape (n_features,)
    """
    if window.ndim != 2 or window.shape[1] != 9:
        raise ValueError(f"Expected (W, 9), got {window.shape}")

    parts = [_stat_per_axis(window)]
    for group_idxs in _SENSOR_GROUPS.values():
        parts.append(_magnitude_stats(window, group_idxs))

    return np.concatenate(parts)


def extract_batch(windows: np.ndarray) -> np.ndarray:
    """
    Extract features from a batch of windows.

    Parameters
    ----------
    windows : np.ndarray, shape (B, W, 9) or (W, 9) for a single window

    Returns
    -------
    np.ndarray, shape (B, n_features)
    """
    if windows.ndim == 2:
        return extract(windows)[np.newaxis, :]
    return np.vstack([extract(w) for w in windows])


def feature_names() -> list[str]:
    """Return ordered list of feature names matching `extract()` output."""
    names: list[str] = []
    stat_labels = ["mean", "std", "min", "max", "median", "energy", "zcr"]
    for sensor in _SENSORS:
        for stat in stat_labels:
            names.append(f"{sensor}_{stat}")
    for group in _SENSOR_GROUPS:
        for stat in ["mean", "std", "min", "max", "energy"]:
            names.append(f"{group}_mag_{stat}")
    return names


def n_features() -> int:
    return len(feature_names())
