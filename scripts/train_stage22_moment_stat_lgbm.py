#!/usr/bin/env python3
"""
Stage 22 — Frozen MOMENT embeddings + hand-crafted statistical/spectral features
            → LightGBM classifier (pure foundation-model submission path).

Feature vector per window (5,512-d total):
  - MOMENT-1-large frozen embeddings: 4 positions × 1,024-d  = 4,096-d
  - Statistical/spectral hand-crafted: 4 positions × 354-d   = 1,416-d
  Combined: 5,512-d  → LightGBM multiclass (8 classes)

This is a fully compliant SHL 2026 foundation-model submission:
  - Foundation weights: NEVER updated (frozen extraction only)
  - Trained component: LightGBM head (lightweight, ~500 trees)
  - No scratch-trained deep models used

Training strategy:
  - Train LightGBM on full labelled train split (392,142 windows)
  - Evaluate on full labelled val split (57,576 windows) — unbiased F1
  - Predict on unlabelled test split (92,726 windows) → submission file

Usage
-----
python scripts/train_stage22_moment_stat_lgbm.py
python scripts/train_stage22_moment_stat_lgbm.py --n-trees 1000 --lr 0.03
python scripts/train_stage22_moment_stat_lgbm.py --no-test  # skip test prediction
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HDF5_PATH  = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
FEAT_DIR   = REPO_ROOT / "dataset" / "processed" / "features"
EMB_DIR    = REPO_ROOT / "outputs" / "execution-output" / "moment_embeddings"
OUT_DIR    = REPO_ROOT / "outputs" / "execution-output" / "moment_stat_lgbm_stage22"
SUB_DIR    = REPO_ROOT / "outputs" / "execution-output" / "submissions"

POSITIONS  = ["Bag", "Hand", "Hips", "Torso"]
WIN_SIZE   = 500
FFT_TOP_K  = 20


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_moment_embeddings(split: str) -> np.ndarray:
    """Load MOMENT embeddings and return (N, 4096) float32."""
    path = EMB_DIR / f"{split}_embeddings.npz"
    print(f"  Loading MOMENT {split} embeddings from {path.name} …", flush=True)
    d = np.load(path)
    key = list(d.keys())[0]
    emb = d[key].astype(np.float32)            # (N, 4, 1024)
    return emb.reshape(emb.shape[0], -1)        # (N, 4096)


def load_stat_features(split: str, positions=POSITIONS) -> np.ndarray:
    """Load pre-computed statistical features and return (N, 4×354) float32."""
    parts = []
    for pos in positions:
        path = FEAT_DIR / f"{split}_{pos}.npz"
        d = np.load(path)
        parts.append(d["X"].astype(np.float32))   # (N, 354)
    return np.concatenate(parts, axis=1)           # (N, 4×354 = 1416)


def load_labels(split: str) -> np.ndarray:
    """Return (N,) int labels 1-based from any position (all identical)."""
    path = FEAT_DIR / f"{split}_Bag.npz"
    return np.load(path)["y"].astype(np.int64)


def extract_test_stat_features() -> np.ndarray:
    """
    Extract statistical features from the raw test HDF5 on-the-fly.
    Test has a single flat array (no position labels).
    We extract once and repeat ×4 to match the 4-position training dimension.
    Returns (92726, 4×354) float32.
    """
    import h5py
    from featureflyers_shl.features.statistical import extract_batch

    print("  Extracting test statistical features from HDF5 (on-the-fly) …", flush=True)
    t0 = time.time()
    with h5py.File(HDF5_PATH, "r") as hf:
        raw = hf["test"]["data"][:].astype(np.float32)  # (46363000, 9)

    n_win = raw.shape[0] // WIN_SIZE
    windows = raw[: n_win * WIN_SIZE].reshape(n_win, WIN_SIZE, 9)  # (92726, 500, 9)
    del raw

    # Extract in batches to keep memory manageable
    batch = 4096
    feats = []
    for i in range(0, n_win, batch):
        feats.append(extract_batch(windows[i : i + batch], FFT_TOP_K))
        if i % (batch * 8) == 0:
            print(f"    {i:,}/{n_win:,} windows …", flush=True)
    single = np.vstack(feats).astype(np.float32)  # (92726, 354)
    del windows

    # Repeat ×4 to match training dimension (4 positions)
    combined = np.tile(single, 4)                  # (92726, 1416)
    elapsed = time.time() - t0
    print(f"  Test stat features: {combined.shape}  ({elapsed:.0f}s)", flush=True)
    return combined


def build_features(split: str) -> tuple:
    """Return (X, y) where X is (N, 5512) and y is (N,) or None for test."""
    emb = load_moment_embeddings(split)            # (N, 4096)
    if split == "test":
        stat = extract_test_stat_features()        # (N, 1416)
        y = None
    else:
        stat = load_stat_features(split)           # (N, 1416)
        y = load_labels(split) - 1                 # 0-based for LightGBM
    X = np.concatenate([emb, stat], axis=1)        # (N, 5512)
    print(f"  {split}: X={X.shape}  labels={'none' if y is None else y.shape}", flush=True)
    return X, y


# ── Training ─────────────────────────────────────────────────────────────────

def train_lgbm(X_tr, y_tr, X_val, y_val, n_trees: int, lr: float, seed: int):
    import lightgbm as lgb
    from sklearn.metrics import f1_score

    print(f"\nTraining LightGBM: {n_trees} trees, lr={lr} …", flush=True)
    clf = lgb.LGBMClassifier(
        n_estimators=n_trees,
        learning_rate=lr,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    y_pred = clf.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="macro")
    print(f"\nVal Macro-F1 (full val set, unbiased): {f1:.4f}", flush=True)

    # Per-class breakdown
    from sklearn.metrics import classification_report
    labels = ["Still", "Walking", "Run", "Bike", "Car", "Bus", "Train", "Metro"]
    print(classification_report(y_val, y_pred, target_names=labels, digits=4))
    return clf, f1


# ── Submission writer ─────────────────────────────────────────────────────────

def write_submission(labels_1based: np.ndarray, path: Path) -> None:
    """Write 92726 lines each containing 500 copies of the predicted label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(labels_1based)
    print(f"\nWriting submission: {n:,} windows → {path.name} …", flush=True)
    with open(path, "w") as fh:
        for lbl in labels_1based:
            fh.write(",".join([str(lbl)] * WIN_SIZE) + "\n")
    size_mb = path.stat().st_size / 1e6
    print(f"  Written: {path}  ({size_mb:.1f} MB)", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trees", type=int,   default=500)
    parser.add_argument("--lr",      type=float, default=0.05)
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--no-test", action="store_true",
                        help="Skip test prediction (val evaluation only)")
    parser.add_argument("--out-dir", type=Path,  default=OUT_DIR)
    parser.add_argument("--output",  type=Path,
                        default=SUB_DIR / "FeatureFlyers_moment_stat_lgbm_s22.txt")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage 22 — MOMENT (frozen) + Stat features → LightGBM")
    print(f"  Feature dim: 4096 (MOMENT) + 1416 (stat ×4 pos) = 5512")
    print(f"  Trees: {args.n_trees}  LR: {args.lr}  Seed: {args.seed}")
    print("=" * 60)

    # ── Load features ────────────────────────────────────────────────────────
    print("\n[1/4] Loading train features …")
    X_tr, y_tr = build_features("train")

    print("\n[2/4] Loading validation features …")
    X_val, y_val = build_features("validation")

    # ── Train LightGBM ───────────────────────────────────────────────────────
    print("\n[3/4] Training …")
    t0 = time.time()
    clf, val_f1 = train_lgbm(X_tr, y_tr, X_val, y_val,
                              n_trees=args.n_trees, lr=args.lr, seed=args.seed)
    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.0f}s")

    # ── Save model & metrics ─────────────────────────────────────────────────
    import json, joblib
    joblib.dump(clf, args.out_dir / "lgbm_moment_stat.pkl")
    metrics = {"val_macro_f1": round(val_f1, 6), "n_trees": args.n_trees,
               "lr": args.lr, "feature_dim": 5512, "seed": args.seed}
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nModel saved to {args.out_dir}/lgbm_moment_stat.pkl")

    # ── Test prediction ───────────────────────────────────────────────────────
    if not args.no_test:
        print("\n[4/4] Loading test features and predicting …")
        X_test, _ = build_features("test")
        y_test_pred = clf.predict(X_test).astype(np.int64) + 1  # back to 1-based
        write_submission(y_test_pred, args.output)
        print(f"\nSubmission: {args.output}")
    else:
        print("\n[4/4] Skipping test prediction (--no-test)")

    print("\n=== Stage 22 complete ===")
    print(f"  Val Macro-F1 (full val, unbiased): {val_f1:.4f}")
    if not args.no_test:
        print(f"  Submission: {args.output}")


if __name__ == "__main__":
    main()
