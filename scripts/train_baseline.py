#!/usr/bin/env python3
"""
Train a classical ML baseline (Random Forest or Logistic Regression) on
windowed SHL 2026 data from the HDF5 file.

Usage:
    python scripts/train_baseline.py --sample-limit 5000
    python scripts/train_baseline.py --sample-limit 50000 --model rf
    python scripts/train_baseline.py --model lr                    # full dataset

Steps:
    1. Load windowed samples from HDF5 (train + validation splits)
    2. Extract statistical features per window
    3. Train model
    4. Evaluate on validation set
    5. Save metrics to outputs/baseline/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HDF5_PATH = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
OUT_DIR = REPO_ROOT / "outputs" / "baseline"

LABEL_MAP = {1: "Still", 2: "Walking", 3: "Run", 4: "Bike",
             5: "Car", 6: "Bus", 7: "Train", 8: "Metro"}

POSITIONS = ["Bag", "Hand", "Hips", "Torso"]

SENSORS = ["Acc_x", "Acc_y", "Acc_z", "Gyr_x", "Gyr_y", "Gyr_z",
           "Mag_x", "Mag_y", "Mag_z"]

WIN_SIZE = 500
HOP_SIZE = 250


_HDF5_READ_BLOCK = 5_000_000   # samples per sequential HDF5 read (full-dataset mode)


def load_windows(hf, split: str, positions: list[str],
                 n_samples: int | None, rng: np.random.Generator):
    """
    Return (X_features, y_labels) for one split.

    Sample-limit mode:
        1. Read the label array (small: ~98 MB int8 per position).
        2. Compute window-centre labels at hop intervals.
        3. Stratified-sample k windows equally across all present classes.
        4. Sort selected indices; read HDF5 data in contiguous chunks.

    Full-dataset mode:
        Sequential chunked reads of _HDF5_READ_BLOCK samples to avoid OOM.
    """
    from featureflyers_shl.features.statistical import extract_batch
    from collections import Counter

    X_all, y_all = [], []

    for pos in positions:
        path = f"{split}/{pos}"
        if path not in hf:
            continue

        ds_data   = hf[path]["data"]     # (N, 9) float32
        ds_labels = hf[path]["labels"]   # (N,) int8
        N = ds_data.shape[0]
        k_per_pos = (n_samples // len(positions)) if n_samples else None

        print(f"  {split}/{pos}: N={N:,}", flush=True)
        t0 = time.time()

        if k_per_pos is not None:
            # ---- stratified sample ----
            print(f"    reading labels …", end="", flush=True)
            all_labels = ds_labels[:]   # (N,) int8 — small: ~100 MB
            print(f" done ({time.time()-t0:.1f}s)", flush=True)

            # Valid window-centre indices
            centres = np.arange(WIN_SIZE // 2, N - WIN_SIZE // 2, HOP_SIZE, dtype=np.int64)
            centre_labels = all_labels[centres]
            del all_labels

            classes = np.unique(centre_labels)
            k_per_class = max(1, k_per_pos // len(classes))
            selected = []
            for cls in classes:
                idx = np.where(centre_labels == cls)[0]
                k = min(k_per_class, len(idx))
                selected.append(rng.choice(idx, size=k, replace=False))
            sel_idx = np.concatenate(selected)[:k_per_pos]
            # Map centre indices back to window start indices
            win_starts = centres[sel_idx] - WIN_SIZE // 2
            sel_labels = centre_labels[sel_idx]

            # Group sorted win_starts by HDF5 chunk — one read per chunk
            order = np.argsort(win_starts)
            win_starts = win_starts[order]
            sel_labels = sel_labels[order]

            # HDF5 chunk rows from the dataset's storage chunk
            hdf5_chunk_rows = ds_data.chunks[0] if ds_data.chunks else 500_000

            wins_buf = np.empty((len(win_starts), WIN_SIZE, 9), dtype=np.float32)
            chunk_id = win_starts // hdf5_chunk_rows
            boundaries = np.where(np.diff(chunk_id))[0] + 1
            groups = np.split(np.arange(len(win_starts)), boundaries)

            for grp in groups:
                chunk_start = int(win_starts[grp[0]]) // hdf5_chunk_rows * hdf5_chunk_rows
                chunk_end   = min(int(win_starts[grp[-1]]) + WIN_SIZE, N)
                block = ds_data[chunk_start: chunk_end, :]   # one HDF5 chunk read
                for gi in grp:
                    offset = int(win_starts[gi]) - chunk_start
                    wins_buf[gi] = block[offset: offset + WIN_SIZE, :]

            wins = wins_buf
            lbls = sel_labels.astype(np.int8)
        else:
            # ---- full dataset sequential ----
            wins_list, lbls_list = [], []
            for chunk_start in range(0, N - WIN_SIZE + 1, _HDF5_READ_BLOCK):
                end = min(chunk_start + _HDF5_READ_BLOCK + WIN_SIZE, N)
                block = ds_data[chunk_start: end, :]
                blbls = ds_labels[chunk_start: end]
                n_w = (len(block) - WIN_SIZE) // HOP_SIZE + 1
                for j in range(n_w):
                    s = j * HOP_SIZE
                    wins_list.append(block[s: s + WIN_SIZE, :])
                    lbls_list.append(int(blbls[s + WIN_SIZE // 2]))
            wins = np.stack(wins_list)
            lbls = np.array(lbls_list, dtype=np.int8)

        X = extract_batch(wins)
        X_all.append(X)
        y_all.append(lbls)
        dist = dict(sorted(Counter(lbls.tolist()).items()))
        print(f"    → {len(lbls):,} windows, F={X.shape[1]}, "
              f"classes={dist}  ({time.time()-t0:.1f}s)")

    return np.concatenate(X_all), np.concatenate(y_all)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-limit", type=int, default=None,
                        help="Max windows to draw from train+val (for fast testing)")
    parser.add_argument("--model", choices=["rf", "lr"], default="rf",
                        help="Classifier: rf=RandomForest, lr=LogisticRegression")
    parser.add_argument("--positions", nargs="+", default=["Bag"],
                        choices=POSITIONS,
                        help="Positions to use (default: Bag only for speed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Trees for RF (ignored for LR)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    try:
        import h5py
    except ImportError:
        print("ERROR: h5py not installed"); sys.exit(1)
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (classification_report, confusion_matrix,
                                     f1_score)
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("ERROR: scikit-learn not installed — run: pip install scikit-learn")
        sys.exit(1)

    if not HDF5_PATH.exists():
        print(f"ERROR: {HDF5_PATH} not found. Run scripts/convert_to_hdf5.py first.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nBaseline training — model={args.model}  "
          f"sample_limit={args.sample_limit}  positions={args.positions}")
    print(f"HDF5: {HDF5_PATH.relative_to(REPO_ROOT)}\n")

    t_start = time.time()

    with h5py.File(HDF5_PATH, "r") as hf:
        print("Loading TRAIN windows …")
        X_train, y_train = load_windows(hf, "train", args.positions,
                                        args.sample_limit, rng)
        print(f"\nLoading VALIDATION windows …")
        X_val, y_val = load_windows(hf, "validation", args.positions,
                                    args.sample_limit, rng)

    print(f"\nTrain: {X_train.shape}  Val: {X_val.shape}")
    print(f"NaN in train: {np.isnan(X_train).sum()}, val: {np.isnan(X_val).sum()}")

    # Standardise
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Build model
    if args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            n_jobs=-1,
            random_state=args.seed,
            class_weight="balanced",
        )
    else:
        clf = LogisticRegression(
            max_iter=500,
            n_jobs=-1,
            random_state=args.seed,
            class_weight="balanced",
        )

    print(f"\nFitting {clf.__class__.__name__} …", flush=True)
    t_fit = time.time()
    clf.fit(X_train, y_train)
    print(f"  Done in {time.time()-t_fit:.1f}s")

    # Evaluate
    y_pred = clf.predict(X_val)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    acc = (y_pred == y_val).mean()

    print(f"\n{'='*60}")
    print(f"Validation  Accuracy: {acc:.4f}   Macro-F1: {macro_f1:.4f}")
    print(f"{'='*60}")

    label_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
    present = sorted(set(y_val.tolist()) | set(y_train.tolist()))
    report = classification_report(
        y_val, y_pred,
        labels=present,
        target_names=[LABEL_MAP[l] for l in present],
    )
    print(report)

    # Save metrics
    run_name = (f"{args.model}_s{args.sample_limit or 'full'}"
                f"_pos{'_'.join(args.positions)}")
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": args.model,
        "positions": args.positions,
        "sample_limit": args.sample_limit,
        "seed": args.seed,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_features": int(X_train.shape[1]),
        "val_accuracy": float(acc),
        "val_macro_f1": float(macro_f1),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "classification_report.txt").write_text(report)

    print(f"\nResults saved → {run_dir.relative_to(REPO_ROOT)}/")
    print(f"  macro-F1: {macro_f1:.4f}   accuracy: {acc:.4f}")
    print(f"  elapsed:  {metrics['elapsed_s']}s")


if __name__ == "__main__":
    main()
