#!/usr/bin/env python3
"""
Train a classical ML baseline (Random Forest or Logistic Regression) on
windowed SHL 2026 data from the HDF5 file.

Usage:
    # Fast smoke test — single position, no fusion
    python scripts/train_baseline.py --sample-limit 5000

    # Four-position early fusion, fast mode
    python scripts/train_baseline.py --sample-limit 5000 \\
        --positions Bag Hand Hips Torso --fusion early

    # Single position, full dataset
    python scripts/train_baseline.py --positions Bag --model rf

    # Four-position early fusion, full dataset (slow)
    python scripts/train_baseline.py --positions Bag Hand Hips Torso \\
        --fusion early --model rf

Fusion modes:
    none  — train one classifier per position (default)
    early — concatenate feature vectors from all positions, one classifier

Results and fitted model saved to outputs/baseline/<run-name>/.
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

WIN_SIZE = 500
HOP_SIZE = 250
FFT_TOP_K = 20
_HDF5_READ_BLOCK = 5_000_000   # samples per chunk in full-dataset mode


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_windows_one_position(ds_data, ds_labels, k: int | None,
                                rng: np.random.Generator, label: str) -> tuple:
    """
    Return (windows (K,500,9), labels (K,)) for one HDF5 position dataset.

    Stratified when k is set: reads the label array, picks k//n_classes
    indices per class, then batch-reads sensor data grouped by HDF5 chunk.
    Full-dataset mode: sequential chunked reads.
    """
    N = ds_data.shape[0]
    print(f"    N={N:,}", end="", flush=True)
    t0 = time.time()

    if k is not None:
        # Read labels (tiny: int8 ~100 MB)
        all_labels = ds_labels[:]
        centres = np.arange(WIN_SIZE // 2, N - WIN_SIZE // 2, HOP_SIZE, dtype=np.int64)
        centre_labels = all_labels[centres]
        del all_labels

        classes = np.unique(centre_labels)
        k_per_class = max(1, k // len(classes))
        selected = []
        for cls in classes:
            idx = np.where(centre_labels == cls)[0]
            selected.append(rng.choice(idx, size=min(k_per_class, len(idx)),
                                       replace=False))
        sel_idx = np.concatenate(selected)[:k]

        win_starts = (centres[sel_idx] - WIN_SIZE // 2)
        sel_labels = centre_labels[sel_idx].astype(np.int8)

        # Sort for sequential-ish HDF5 access, batch by chunk
        order = np.argsort(win_starts)
        win_starts = win_starts[order]
        sel_labels = sel_labels[order]

        hdf5_chunk_rows = ds_data.chunks[0] if ds_data.chunks else 500_000
        wins_buf = np.empty((len(win_starts), WIN_SIZE, 9), dtype=np.float32)
        chunk_id = win_starts // hdf5_chunk_rows
        boundaries = np.where(np.diff(chunk_id))[0] + 1
        groups = np.split(np.arange(len(win_starts)), boundaries)
        for grp in groups:
            c_start = int(win_starts[grp[0]]) // hdf5_chunk_rows * hdf5_chunk_rows
            c_end = min(int(win_starts[grp[-1]]) + WIN_SIZE, N)
            block = ds_data[c_start: c_end, :]
            for gi in grp:
                offset = int(win_starts[gi]) - c_start
                wins_buf[gi] = block[offset: offset + WIN_SIZE, :]

        wins = wins_buf
        lbls = sel_labels
    else:
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

    from collections import Counter
    dist = dict(sorted(Counter(lbls.tolist()).items()))
    print(f"  → {len(lbls):,} windows  classes={dist}  ({time.time()-t0:.1f}s)")
    return wins, lbls


def load_features(hf, split: str, positions: list[str],
                  n_samples: int | None, rng: np.random.Generator,
                  fusion: str) -> tuple:
    """
    Return (X, y) for one split.

    fusion='none'  → uses first position only (positions[0])
    fusion='early' → feature vectors concatenated across positions;
                     windows are aligned by label (all positions share labels)
    """
    from featureflyers_shl.features.statistical import extract_batch

    if fusion == "none":
        pos = positions[0]
        path = f"{split}/{pos}"
        if path not in hf:
            raise FileNotFoundError(f"HDF5 path not found: {path}")
        print(f"  {split}/{pos}:", end="")
        wins, lbls = _load_windows_one_position(
            hf[path]["data"], hf[path]["labels"], n_samples, rng, pos)
        X = extract_batch(wins, FFT_TOP_K)
        return X, lbls

    # Early fusion: same window indices for each position
    # Use Bag labels as the reference; all positions share the same label sequence
    ref_pos = positions[0]
    ref_path = f"{split}/{ref_pos}"
    if ref_path not in hf:
        raise FileNotFoundError(f"HDF5 path not found: {ref_path}")

    print(f"  {split}/{ref_pos} (reference):", end="")
    ref_wins, lbls = _load_windows_one_position(
        hf[ref_path]["data"], hf[ref_path]["labels"], n_samples, rng, ref_pos)
    X_parts = [extract_batch(ref_wins, FFT_TOP_K)]

    # For remaining positions, read the same window indices (matched to ref)
    # We rebuild win_starts from the reference labels so they align
    hdf5_chunk_rows = hf[ref_path]["data"].chunks[0] if hf[ref_path]["data"].chunks else 500_000

    for pos in positions[1:]:
        path = f"{split}/{pos}"
        if path not in hf:
            print(f"  [SKIP] {path} not found — omitting from fusion")
            continue
        ds = hf[path]["data"]
        N_pos = ds.shape[0]

        print(f"  {split}/{pos} (fusion):", end="", flush=True)
        t0 = time.time()

        # Rebuild the same win_starts that were used for the reference position
        # (stratified: fixed by same rng state — but rng advanced; we re-derive
        #  from lbls instead)
        # Simpler: re-read same number of windows using same label-stratified selection
        # but from this position's label array (labels are identical across positions)
        pos_labels = hf[path]["labels"][:]
        centres = np.arange(WIN_SIZE // 2, N_pos - WIN_SIZE // 2, HOP_SIZE, dtype=np.int64)
        centre_labels = pos_labels[centres]
        del pos_labels

        # Build per-class pools
        needed_per_class: dict[int, int] = {}
        for l in lbls:
            needed_per_class[int(l)] = needed_per_class.get(int(l), 0) + 1

        selected = []
        for cls, cnt in sorted(needed_per_class.items()):
            idx = np.where(centre_labels == cls)[0]
            chosen = rng.choice(idx, size=min(cnt, len(idx)), replace=False)
            selected.append(chosen)
        sel_idx = np.concatenate(selected)
        win_starts = (centres[sel_idx] - WIN_SIZE // 2)
        order = np.argsort(win_starts)
        win_starts = win_starts[order]

        chunk_id = win_starts // hdf5_chunk_rows
        boundaries = np.where(np.diff(chunk_id))[0] + 1
        groups = np.split(np.arange(len(win_starts)), boundaries)
        wins_buf = np.empty((len(win_starts), WIN_SIZE, 9), dtype=np.float32)
        for grp in groups:
            c_start = int(win_starts[grp[0]]) // hdf5_chunk_rows * hdf5_chunk_rows
            c_end = min(int(win_starts[grp[-1]]) + WIN_SIZE, N_pos)
            block = ds[c_start: c_end, :]
            for gi in grp:
                offset = int(win_starts[gi]) - c_start
                wins_buf[gi] = block[offset: offset + WIN_SIZE, :]

        # Trim / pad to match reference length
        n_ref = len(lbls)
        if len(wins_buf) >= n_ref:
            wins_buf = wins_buf[:n_ref]
        else:
            wins_buf = np.concatenate([
                wins_buf,
                np.zeros((n_ref - len(wins_buf), WIN_SIZE, 9), dtype=np.float32)
            ])

        X_parts.append(extract_batch(wins_buf, FFT_TOP_K))
        print(f"  → {len(wins_buf):,} windows  ({time.time()-t0:.1f}s)")

    X = np.concatenate(X_parts, axis=1)
    return X, lbls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-limit", type=int, default=None,
                        help="Max windows per split (stratified; None = full dataset)")
    parser.add_argument("--model", choices=["rf", "lr"], default="rf")
    parser.add_argument("--positions", nargs="+", default=["Bag"],
                        choices=POSITIONS)
    parser.add_argument("--fusion", choices=["none", "early"], default="none",
                        help="none=single position; early=concatenate features")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.fusion == "early" and len(args.positions) < 2:
        parser.error("--fusion early requires at least 2 positions")

    rng = np.random.default_rng(args.seed)

    try:
        import h5py
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, f1_score
        from sklearn.preprocessing import StandardScaler
        import joblib
    except ImportError as e:
        print(f"ERROR: {e}\nRun: pip install scikit-learn h5py joblib")
        sys.exit(1)

    if not HDF5_PATH.exists():
        print(f"ERROR: {HDF5_PATH} not found. Run scripts/convert_to_hdf5.py first.")
        sys.exit(1)

    pos_str = "_".join(args.positions)
    run_name = (f"{args.model}_s{args.sample_limit or 'full'}"
                f"_pos{pos_str}_fusion{args.fusion}")
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBaseline — model={args.model}  fusion={args.fusion}"
          f"  positions={args.positions}"
          f"  sample_limit={args.sample_limit}  seed={args.seed}")
    print(f"Output → {run_dir.relative_to(REPO_ROOT)}\n")

    t_start = time.time()

    with h5py.File(HDF5_PATH, "r") as hf:
        print("Loading TRAIN …")
        X_train, y_train = load_features(
            hf, "train", args.positions, args.sample_limit, rng, args.fusion)
        print(f"  X_train {X_train.shape}  y_train {y_train.shape}\n")

        print("Loading VALIDATION …")
        X_val, y_val = load_features(
            hf, "validation", args.positions, args.sample_limit, rng, args.fusion)
        print(f"  X_val {X_val.shape}  y_val {y_val.shape}\n")

    # NaN guard
    for name, X in [("train", X_train), ("val", X_val)]:
        n_nan = np.isnan(X).sum()
        if n_nan:
            print(f"  [WARN] {n_nan} NaN in {name} — replacing with 0")
            np.nan_to_num(X, copy=False)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Build model
    if args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators, n_jobs=-1,
            random_state=args.seed, class_weight="balanced")
    else:
        clf = LogisticRegression(
            max_iter=500, n_jobs=-1,
            random_state=args.seed, class_weight="balanced")

    print(f"Fitting {clf.__class__.__name__} …", flush=True)
    t_fit = time.time()
    clf.fit(X_train_s, y_train)
    print(f"  Done in {time.time()-t_fit:.1f}s\n")

    # Evaluate
    y_pred = clf.predict(X_val_s)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    acc = float((y_pred == y_val).mean())

    present = sorted(set(y_train.tolist()) | set(y_val.tolist()))
    report = classification_report(
        y_val, y_pred,
        labels=present,
        target_names=[LABEL_MAP[l] for l in present],
        zero_division=0,
    )
    print(f"{'='*60}")
    print(f"Validation  Accuracy: {acc:.4f}   Macro-F1: {macro_f1:.4f}")
    print(f"{'='*60}")
    print(report)

    # Save artefacts
    metrics = {
        "model": args.model, "fusion": args.fusion,
        "positions": args.positions,
        "sample_limit": args.sample_limit, "seed": args.seed,
        "fft_top_k": FFT_TOP_K,
        "n_train": int(X_train.shape[0]), "n_val": int(X_val.shape[0]),
        "n_features": int(X_train.shape[1]),
        "val_accuracy": round(acc, 4),
        "val_macro_f1": round(macro_f1, 4),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "classification_report.txt").write_text(report)

    # Save model bundle (scaler + classifier)
    model_path = run_dir / "model.joblib"
    joblib.dump({"scaler": scaler, "clf": clf,
                 "positions": args.positions, "fusion": args.fusion,
                 "fft_top_k": FFT_TOP_K, "win_size": WIN_SIZE},
                model_path)
    print(f"Model saved → {model_path.relative_to(REPO_ROOT)}")
    print(f"Metrics    → {(run_dir/'metrics.json').relative_to(REPO_ROOT)}")
    print(f"macro-F1: {macro_f1:.4f}   acc: {acc:.4f}   elapsed: {metrics['elapsed_s']}s")


if __name__ == "__main__":
    main()
