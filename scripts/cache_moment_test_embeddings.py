#!/usr/bin/env python3
"""
Cache MOMENT-1-large embeddings for the test split.

Reads the 92 726 pre-windowed test windows from HDF5, applies per-window
z-score normalisation (matching the training pipeline), runs MOMENT in
embedding mode, and saves the result to:

  dataset/processed/embeddings/test_Bag_moment_normperwindow_mean_pool.npz

This file is consumed by run_stage9_ensemble.py to include MOMENT-XGB in the
test ensemble without re-running the full foundation training pipeline.

Usage
-----
CUDA_VISIBLE_DEVICES=2 python -u scripts/cache_moment_test_embeddings.py \\
    --device cuda:0 --batch-size 256 \\
    2>&1 | tee outputs/execution-output/cache_moment_test_embed.log

Expected runtime: ~19 min at 82 windows/s on a single GPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HDF5_PATH  = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
EMB_DIR    = REPO_ROOT / "dataset" / "processed" / "embeddings"
FEAT_DIR   = REPO_ROOT / "dataset" / "processed" / "features"
OUT_EMB    = EMB_DIR  / "test_Bag_moment_normperwindow_mean_pool.npz"
OUT_FEAT   = FEAT_DIR / "test_Bag.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device",     default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit",      type=int, default=None,
                        help="Process only first N windows (smoke test)")
    args = parser.parse_args()

    import torch
    from featureflyers_shl.models.foundation import MomentEncoder

    device = torch.device(args.device)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nMOMENT test embedding + stat feature extraction")
    print(f"  device     : {device}")
    print(f"  batch size : {args.batch_size}")
    print(f"  embed out  : {OUT_EMB.relative_to(REPO_ROOT)}")
    print(f"  feat out   : {OUT_FEAT.relative_to(REPO_ROOT)}\n")

    # Load test windows: (92726, 500, 9) → (N, 9, 500) channels-first
    import h5py
    print("Loading test windows from HDF5 …", flush=True)
    t0 = time.time()
    with h5py.File(HDF5_PATH, "r") as hf:
        raw = hf["test"]["data"][:].astype(np.float32)   # (92726, 500, 9)
    raw = raw.transpose(0, 2, 1)                          # (92726, 9, 500)
    n_total = len(raw)
    n_windows = min(n_total, args.limit) if args.limit else n_total
    raw = raw[:n_windows]
    print(f"  {n_windows:,} windows  ({time.time()-t0:.1f}s)\n")

    # Load MOMENT encoder (frozen, FP16)
    print("Loading MOMENT-1-large …", flush=True)
    t0 = time.time()
    encoder = MomentEncoder(embed_strategy="mean_pool").to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    # Cast to FP16 to fit in 11 GB VRAM at batch=256
    encoder = encoder.half()
    print(f"  Loaded in {time.time()-t0:.1f}s\n")

    # Extract embeddings in batches with per-window z-score normalisation
    print(f"Extracting embeddings (batch={args.batch_size}) …", flush=True)
    embeddings = np.empty((n_windows, 1024), dtype=np.float32)
    t_start = time.time()

    with torch.no_grad():
        for start in range(0, n_windows, args.batch_size):
            end = min(start + args.batch_size, n_windows)
            x = torch.from_numpy(raw[start:end]).to(device).half()  # (B, 9, 500)

            # Per-window z-score (matching training pipeline)
            x = x.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            mu  = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
            x   = (x - mu) / std

            emb = encoder(x).float().cpu().numpy()   # (B, 1024)
            embeddings[start:end] = emb

            if start % (args.batch_size * 10) == 0:
                elapsed = time.time() - t_start
                speed   = (end) / max(elapsed, 1e-3)
                eta     = (n_windows - end) / max(speed, 1)
                print(f"  {end:,}/{n_windows:,}  "
                      f"({100*end/n_windows:.0f}%)  "
                      f"{speed:.0f} win/s  ETA {eta/60:.1f} min",
                      flush=True)

    total = time.time() - t_start
    print(f"\nDone in {total:.1f}s  ({n_windows/total:.0f} windows/s)")

    # Save MOMENT embeddings (no labels for test — only 'X' key)
    np.savez_compressed(OUT_EMB, X=embeddings)
    size_mb = OUT_EMB.stat().st_size / 1024 / 1024
    print(f"Saved → {OUT_EMB.relative_to(REPO_ROOT)}  ({size_mb:.1f} MB)")
    print(f"  shape: {embeddings.shape}  dtype: {embeddings.dtype}\n")

    # Extract and cache statistical/spectral features for test windows
    # test/data is already windowed at (N, 500, 9) — run extract_batch directly
    if not OUT_FEAT.exists() or args.limit:
        print("Extracting test stat features (CPU, ~2 min) …", flush=True)
        from featureflyers_shl.features.statistical import extract_batch
        FFT_TOP_K = 20   # matches _DEFAULT_FFT_K used for all cached train/val features
        t_feat = time.time()
        STAT_BATCH = 2048
        n_feat_cols = None
        feat_chunks = []
        for start in range(0, n_windows, STAT_BATCH):
            end   = min(start + STAT_BATCH, n_windows)
            batch = raw[start:end].transpose(0, 2, 1)   # (B, 9, 500) → (B, 500, 9) for extract_batch
            chunk = extract_batch(batch, fft_top_k=FFT_TOP_K)
            feat_chunks.append(chunk)
            if start % (STAT_BATCH * 10) == 0:
                print(f"  stat features: {end:,}/{n_windows:,}  "
                      f"({100*end/n_windows:.0f}%)", flush=True)
        X_feat = np.vstack(feat_chunks).astype(np.float32)
        np.savez_compressed(OUT_FEAT, X=X_feat)
        size_feat_mb = OUT_FEAT.stat().st_size / 1024 / 1024
        print(f"Saved → {OUT_FEAT.relative_to(REPO_ROOT)}  ({size_feat_mb:.1f} MB)")
        print(f"  shape: {X_feat.shape}  ({time.time()-t_feat:.1f}s)\n")
    else:
        print(f"Stat features already cached at {OUT_FEAT.name} — skipping.")


if __name__ == "__main__":
    main()
