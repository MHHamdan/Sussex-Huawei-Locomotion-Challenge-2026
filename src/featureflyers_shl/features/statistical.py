"""
Statistical and spectral feature extraction for SHL sensor windows.

Input: (W, 9) array for a single window — columns are
    [Acc_x, Acc_y, Acc_z, Gyr_x, Gyr_y, Gyr_z, Mag_x, Mag_y, Mag_z]

Output: 1-D float32 feature vector.  Use `feature_names()` for column labels.

Feature groups (in order):
  1. Per-axis statistics   9 axes × 7 stats         = 63
  2. Group magnitude stats 3 groups × 5 stats        = 15
  3. Per-axis spectral     9 axes × (fft_top_k+3)   = 9 × (K+3)
  4. Group magnitude FFT   3 groups × (fft_top_k+3) = 3 × (K+3)

Default K=20 → total = 63 + 15 + 9×23 + 3×23 = 63 + 15 + 207 + 69 = 354 features.
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

_DEFAULT_FFT_K = 20


# ---------------------------------------------------------------------------
# Statistical features
# ---------------------------------------------------------------------------

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
            float(x.min()),
            float(x.max()),
            float(np.median(x)),
            float(np.mean(x ** 2)),                           # mean power
            float(np.mean(np.diff(np.sign(x - mean)) != 0)), # zcr around mean
        ])
    return np.array(feats, dtype=np.float32)


def _magnitude_stats(w: np.ndarray, indices: list[int]) -> np.ndarray:
    """Euclidean magnitude stats for a 3-axis group → (5,)."""
    axes = w[:, indices]
    mag = np.sqrt((axes ** 2).sum(axis=1))
    return np.array([mag.mean(), mag.std(), float(mag.min()), float(mag.max()),
                     float(np.mean(mag ** 2))], dtype=np.float32)


# ---------------------------------------------------------------------------
# Spectral features
# ---------------------------------------------------------------------------

def _fft_features(signal: np.ndarray, top_k: int) -> np.ndarray:
    """
    FFT-based features for a 1-D signal → (top_k + 3,).

    Features:
        - top_k dominant frequency magnitudes (sorted descending)
        - spectral energy  : sum of squared magnitudes
        - spectral entropy : normalised Shannon entropy of the power spectrum
        - spectral rolloff : frequency index below which 85 % energy is contained
    """
    n = len(signal)
    half = n // 2
    mags = np.abs(np.fft.rfft(signal - signal.mean()))[:half].astype(np.float32)

    # top-K magnitudes
    if top_k >= len(mags):
        top_mags = np.pad(mags, (0, top_k - len(mags)))
    else:
        top_mags = np.partition(mags, -top_k)[-top_k:]
        top_mags = np.sort(top_mags)[::-1]   # descending

    power = mags ** 2
    total_power = power.sum() + 1e-12

    # spectral energy (normalised)
    spec_energy = float(total_power / n)

    # spectral entropy
    p = power / total_power
    spec_entropy = float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p) + 1))

    # spectral rolloff at 85 %
    cumsum = np.cumsum(power)
    rolloff_idx = float(np.searchsorted(cumsum, 0.85 * total_power)) / max(len(mags), 1)

    return np.concatenate([
        top_mags.astype(np.float32),
        np.array([spec_energy, spec_entropy, rolloff_idx], dtype=np.float32),
    ])


def _spectral_per_axis(w: np.ndarray, top_k: int) -> np.ndarray:
    """Per-axis spectral features → (9 × (top_k+3),)."""
    parts = [_fft_features(w[:, i], top_k) for i in range(w.shape[1])]
    return np.concatenate(parts)


def _spectral_magnitude(w: np.ndarray, indices: list[int], top_k: int) -> np.ndarray:
    """Spectral features of the magnitude signal for one sensor group → (top_k+3,)."""
    axes = w[:, indices]
    mag = np.sqrt((axes ** 2).sum(axis=1))
    return _fft_features(mag, top_k)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(window: np.ndarray, fft_top_k: int = _DEFAULT_FFT_K) -> np.ndarray:
    """
    Extract statistical + spectral features from a single window.

    Parameters
    ----------
    window    : (W, 9) float32 sensor data
    fft_top_k : number of top FFT magnitude bins to include (default 20)

    Returns
    -------
    np.ndarray, shape (n_features(fft_top_k),)
    """
    if window.ndim != 2 or window.shape[1] != 9:
        raise ValueError(f"Expected (W, 9), got {window.shape}")
    # Guard against NaN/Inf from recording gaps in raw sensor data
    if not np.isfinite(window).all():
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

    parts = [
        _stat_per_axis(window),
        *[_magnitude_stats(window, idx) for idx in _SENSOR_GROUPS.values()],
        _spectral_per_axis(window, fft_top_k),
        *[_spectral_magnitude(window, idx, fft_top_k)
          for idx in _SENSOR_GROUPS.values()],
    ]
    return np.concatenate(parts).astype(np.float32)


def extract_batch(windows: np.ndarray,
                  fft_top_k: int = _DEFAULT_FFT_K) -> np.ndarray:
    """
    Extract features from a batch of windows.

    Parameters
    ----------
    windows   : (B, W, 9) or (W, 9)
    fft_top_k : FFT top-K bins

    Returns
    -------
    np.ndarray, shape (B, n_features(fft_top_k))
    """
    if windows.ndim == 2:
        return extract(windows, fft_top_k)[np.newaxis, :]
    return np.vstack([extract(w, fft_top_k) for w in windows])


def feature_names(fft_top_k: int = _DEFAULT_FFT_K) -> list[str]:
    """Return ordered feature names matching `extract(fft_top_k=fft_top_k)`."""
    names: list[str] = []

    # Statistical per axis
    stat_labels = ["mean", "std", "min", "max", "median", "energy", "zcr"]
    for sensor in _SENSORS:
        for stat in stat_labels:
            names.append(f"{sensor}_{stat}")

    # Magnitude stats per group
    for group in _SENSOR_GROUPS:
        for stat in ["mean", "std", "min", "max", "energy"]:
            names.append(f"{group}_mag_{stat}")

    # Spectral per axis
    fft_labels = [f"fft_top{i}" for i in range(fft_top_k)]
    fft_labels += ["spec_energy", "spec_entropy", "spec_rolloff"]
    for sensor in _SENSORS:
        for fl in fft_labels:
            names.append(f"{sensor}_{fl}")

    # Spectral magnitude per group
    for group in _SENSOR_GROUPS:
        for fl in fft_labels:
            names.append(f"{group}_mag_{fl}")

    return names


def n_features(fft_top_k: int = _DEFAULT_FFT_K) -> int:
    """Total feature count for a given fft_top_k."""
    return len(feature_names(fft_top_k))
