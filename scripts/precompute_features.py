#!/usr/bin/env python3
"""
Pre-compute statistical+spectral features for all positions and splits and cache
them to disk as compressed .npz files.

Run this ONCE before any training run. Subsequent runs load cached features in
~1 second instead of re-extracting over 60 seconds every time.

Cache location: dataset/processed/features/{split}_{position}.npz
Each file stores:
  X  -- (N, 354) float32 feature matrix
  y  -- (N,)     int64  labels (1-based, matching HDF5)

Usage
-----
# Full precompute (all positions, train + validation) using 32 parallel workers
python scripts/precompute_features.py

# Single position
python scripts/precompute_features.py --positions Bag --splits train validation

# Force recompute even if cache already exists
python scripts/precompute_features.py --force

# Custom worker count
python scripts/precompute_features.py --n-jobs 16
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HDF5_PATH = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
CACHE_DIR = REPO_ROOT / "dataset" / "processed" / "features"

WIN_SIZE   = 500
HOP_SIZE   = 250
FFT_TOP_K  = 20
POSITIONS  = ["Bag", "Hand", "Hips", "Torso"]
SPLITS     = ["train", "validation"]


def _extract_chunk(chunk: np.ndarray, fft_top_k: int) -> np.ndarray:
    from featureflyers_shl.features.statistical import extract_batch
    return extract_batch(chunk, fft_top_k)


def precompute_one(split: str, position: str, n_jobs: int, fft_top_k: int,
                   force: bool) -> None:
    import h5py
    from featureflyers_shl.data.dataset import _read_windows_batched

    cache_file = CACHE_DIR / f"{split}_{position}.npz"

    if cache_file.exists() and not force:
        size_mb = cache_file.stat().st_size / 1e6
        print(f"  [SKIP] {split}/{position} -- already cached  "
              f"({cache_file.name}, {size_mb:.0f} MB)")
        return

    print(f"  [{split}/{position}] reading HDF5 ...", flush=True)
    t0 = time.time()

    with h5py.File(HDF5_PATH, "r") as hf:
        path = f"{split}/{position}"
        if path not in hf:
            print(f"  [WARN] {path} not found in HDF5 -- skipping")
            return
        ds     = hf[path]["data"]
        N      = ds.shape[0]
        chunk_rows = ds.chunks[0] if ds.chunks else 500_000
        labels_raw = hf[path]["labels"][:]

        centres    = np.arange(WIN_SIZE // 2, N - WIN_SIZE // 2, HOP_SIZE, dtype=np.int64)
        win_starts = (centres - WIN_SIZE // 2).astype(np.int64)
        win_labels = labels_raw[centres]

        t1 = time.time()
        print(f"  [{split}/{position}] {len(win_starts):,} windows -- "
              f"reading {len(win_starts):,} from HDF5 ...", flush=True)
        windows = _read_windows_batched(ds, win_starts, N, chunk_rows)  # (K, 500, 9)

    t2 = time.time()
    print(f"  [{split}/{position}] HDF5 read done in {t2-t1:.1f}s -- "
          f"extracting features with {n_jobs} workers ...", flush=True)

    # Parallel feature extraction across n_jobs chunks
    n_chunks = min(n_jobs, len(windows))
    chunks   = np.array_split(windows, n_chunks)
    results  = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_extract_chunk)(chunk, fft_top_k) for chunk in chunks
    )
    X = np.vstack(results).astype(np.float32)       # (K, 354)
    y = win_labels.astype(np.int64)                  # 1-based, matches load_features()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, X=X, y=y)

    elapsed = time.time() - t0
    size_mb = cache_file.stat().st_size / 1e6
    print(f"  [{split}/{position}] saved {cache_file.name}  "
          f"X={X.shape} y={y.shape}  {size_mb:.0f} MB  ({elapsed:.0f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--positions", nargs="+", default=POSITIONS, choices=POSITIONS)
    parser.add_argument("--splits",    nargs="+", default=SPLITS,
                        choices=["train", "validation"])
    parser.add_argument("--n-jobs",   type=int, default=32,
                        help="Parallel workers for feature extraction (default: 32)")
    parser.add_argument("--fft-top-k", type=int, default=FFT_TOP_K)
    parser.add_argument("--force",    action="store_true",
                        help="Recompute and overwrite existing cache files")
    args = parser.parse_args()

    print(f"Precomputing features")
    print(f"  Splits    : {args.splits}")
    print(f"  Positions : {args.positions}")
    print(f"  n_jobs    : {args.n_jobs}")
    print(f"  fft_top_k : {args.fft_top_k}")
    print(f"  cache_dir : {CACHE_DIR}\n")

    t_total = time.time()
    for split in args.splits:
        for position in args.positions:
            precompute_one(split, position, args.n_jobs, args.fft_top_k, args.force)

    print(f"\nAll done in {time.time()-t_total:.0f}s")
    print("\nCache files:")
    for f in sorted(CACHE_DIR.glob("*.npz")):
        print(f"  {f.name}  {f.stat().st_size/1e6:.0f} MB")


if __name__ == "__main__":
    main()
