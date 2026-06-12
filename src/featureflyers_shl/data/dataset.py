"""
PyTorch Dataset wrapper around the SHL 2026 HDF5 file.

Train/val splits are stored as flat (N, 9) sensor streams with (N,) int8 labels.
Windows are computed at __init__ time.

Preload mode (default when sample_limit is set):
  All selected windows are read in sorted, chunk-aligned batches during __init__
  and kept in a (K, 500, 9) float32 numpy array.  __getitem__ is then an O(1)
  memory lookup — no HDF5 access at training time.  Fast for smoke tests.

Lazy mode (default for full-dataset runs, or when preload=False):
  Only window start indices are stored at __init__ time.  __getitem__ opens the
  HDF5 file lazily per DataLoader worker and reads each window on demand.  Safe
  for datasets that don't fit in RAM; slower per batch due to random HDF5 seeks.

Test split is pre-windowed: (92726, 500, 9).  Always preloaded if sample_limit
is set; otherwise lazy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

WIN_SIZE = 500
HOP_SIZE = 250
_LABEL_OFFSET = 1   # HDF5 labels are 1-8; Dataset returns 0-7
N_CLASSES = 8


class SHLWindowDataset(Dataset):
    """
    Parameters
    ----------
    hdf5_path    : path to shl2026.hdf5
    split        : 'train' | 'validation' | 'test'
    position     : 'Bag' | 'Hand' | 'Hips' | 'Torso'  (ignored for test)
    sample_limit : stratified-sample this many windows (None = full dataset)
    seed         : RNG seed for stratified sampling
    preload      : if True, read all windows into RAM at init time (fast training)
                   defaults to True when sample_limit is set, False otherwise
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        split: str,
        position: str = "Bag",
        sample_limit: int | None = None,
        seed: int = 42,
        preload: bool | None = None,
    ) -> None:
        self.hdf5_path = str(hdf5_path)
        self.split = split
        self.position = position
        self._hf = None          # lazy handle for non-preloaded access

        if preload is None:
            preload = sample_limit is not None  # default: preload when a limit is set

        if split in ("train", "validation"):
            self._init_windowed(sample_limit, seed, preload)
        elif split == "test":
            self._init_test(sample_limit, seed, preload)
        else:
            raise ValueError(f"split must be train/validation/test, got {split!r}")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_windowed(self, sample_limit: int | None, seed: int, preload: bool) -> None:
        import h5py

        path = f"{self.split}/{self.position}"
        with h5py.File(self.hdf5_path, "r") as hf:
            if path not in hf:
                raise FileNotFoundError(
                    f"HDF5 path '{path}' not found — "
                    f"check split ({self.split}) and position ({self.position})."
                )
            ds = hf[path]["data"]
            N = ds.shape[0]
            hdf5_chunk_rows = ds.chunks[0] if ds.chunks else 500_000
            labels_raw = hf[path]["labels"][:]  # (N,) int8 — one fast read

            # Window centres at every HOP_SIZE step within bounds
            centres = np.arange(WIN_SIZE // 2, N - WIN_SIZE // 2, HOP_SIZE,
                                dtype=np.int64)
            centre_labels = labels_raw[centres].astype(np.int64)

            if sample_limit is not None:
                rng = np.random.default_rng(seed)
                classes = np.unique(centre_labels)
                k_per = max(1, sample_limit // len(classes))
                chosen = []
                for cls in classes:
                    idx = np.where(centre_labels == cls)[0]
                    chosen.append(
                        rng.choice(idx, size=min(k_per, len(idx)), replace=False)
                    )
                sel = np.concatenate(chosen)[:sample_limit]
                centres = centres[sel]
                centre_labels = centre_labels[sel]

            win_starts = (centres - WIN_SIZE // 2).astype(np.int64)
            self._labels = (centre_labels - _LABEL_OFFSET).astype(np.int64)

            if preload:
                self._X = _read_windows_batched(ds, win_starts, N, hdf5_chunk_rows)
                self._win_starts = None
            else:
                self._X = None
                self._win_starts = win_starts

        self._n = len(self._labels)
        self._is_test = False
        self._path_key = f"{self.split}/{self.position}/data"

    def _init_test(self, sample_limit: int | None, seed: int, preload: bool) -> None:
        import h5py

        with h5py.File(self.hdf5_path, "r") as hf:
            if "test" not in hf or "data" not in hf["test"]:
                raise FileNotFoundError("HDF5 has no 'test/data' dataset.")
            ds = hf["test"]["data"]
            total = ds.shape[0]

            if sample_limit is not None:
                rng = np.random.default_rng(seed)
                indices = rng.choice(total, size=min(sample_limit, total), replace=False)
                indices = np.sort(indices)
            else:
                indices = np.arange(total, dtype=np.int64)

            if preload:
                self._X = ds[indices]           # (K, 500, 9) — direct slice
                self._test_indices = indices
            else:
                self._X = None
                self._test_indices = indices

        self._labels = None
        self._win_starts = None
        self._n = len(indices)
        self._is_test = True
        self._path_key = "test/data"

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        if self._X is not None:
            # Preloaded — pure memory access
            window = self._X[idx]  # (500, 9) numpy
        else:
            # Lazy HDF5 access
            hf = self._open_hf()
            if self._is_test:
                raw_idx = int(self._test_indices[idx])
                window = hf["test"]["data"][raw_idx]
            else:
                start = int(self._win_starts[idx])
                window = hf[self._path_key][start : start + WIN_SIZE]

        x = torch.from_numpy(window.astype(np.float32)).T  # (9, 500)

        if self._is_test:
            return x, idx

        y = int(self._labels[idx])
        return x, y

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def class_counts(self) -> np.ndarray:
        """Per-class sample counts (length N_CLASSES) for loss weighting."""
        if self._labels is None:
            raise RuntimeError("class_counts() not available for test split.")
        return np.bincount(self._labels, minlength=N_CLASSES).astype(np.int64)

    def _open_hf(self):
        """Return file handle, opening lazily (once per DataLoader worker)."""
        if self._hf is None:
            import h5py
            self._hf = h5py.File(self.hdf5_path, "r")
        return self._hf

    def __del__(self) -> None:
        if self._hf is not None:
            try:
                self._hf.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _read_windows_batched(
    ds,                          # open h5py Dataset (N, 9)
    win_starts: np.ndarray,      # (K,) int64 start indices
    N: int,
    hdf5_chunk_rows: int,
) -> np.ndarray:
    """
    Read K windows from an open HDF5 dataset in sorted, chunk-aligned batches.

    Sorts win_starts to minimise HDF5 seeks, reads one HDF5 chunk per group,
    then restores original ordering.  Returns (K, WIN_SIZE, 9) float32 array.
    """
    K = len(win_starts)
    buf = np.empty((K, WIN_SIZE, 9), dtype=np.float32)

    # Sort for sequential access; remember original order to restore
    order = np.argsort(win_starts)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(K)
    sorted_starts = win_starts[order]

    chunk_id = sorted_starts // hdf5_chunk_rows
    boundaries = np.where(np.diff(chunk_id))[0] + 1
    groups = np.split(np.arange(K), boundaries)

    for grp in groups:
        c_start = int(sorted_starts[grp[0]]) // hdf5_chunk_rows * hdf5_chunk_rows
        c_end   = min(int(sorted_starts[grp[-1]]) + WIN_SIZE, N)
        block   = ds[c_start:c_end, :]                  # one HDF5 chunk read
        for gi in grp:
            offset = int(sorted_starts[gi]) - c_start
            buf[inv_order[gi]] = block[offset : offset + WIN_SIZE, :]

    return buf
