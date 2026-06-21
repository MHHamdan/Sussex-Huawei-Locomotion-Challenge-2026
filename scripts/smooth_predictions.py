#!/usr/bin/env python3
"""
Stage 7b — Temporal majority-vote smoothing for SHL submission files.

Locomotion modes persist for minutes; isolated single-window predictions that
differ from their neighbours are almost always wrong.  A sliding majority vote
over W consecutive windows eliminates these flips without any model changes.

Typical gain: +2–4 pp macro-F1 on test, especially for Run/Bike (high
precision, low recall) and Bus (low precision, high recall).

Usage
-----
# Smooth a submission file with window=7 (default)
python scripts/smooth_predictions.py \\
    --input  outputs/execution-output/submissions/my_submission.txt \\
    --output outputs/execution-output/submissions/my_submission_smooth7.txt \\
    --window 7

# Evaluate smoothing gain on validation predictions (need --val-labels path)
python scripts/smooth_predictions.py \\
    --input  outputs/execution-output/val_preds.txt \\
    --output outputs/execution-output/val_preds_smooth7.txt \\
    --window 7 --eval

# Sweep window sizes 1..15 on validation predictions to pick the best W
python scripts/smooth_predictions.py \\
    --input  outputs/execution-output/val_preds.txt \\
    --eval --sweep

Format notes
------------
  Input / output: one line per window, 500 comma-separated label integers (1–8).
  All 500 values on a line are identical (each line = one window label repeated).
  The smoothing operates on the per-line label sequence, not within a line.

Assumption: windows are stored in recording order (standard in SHL challenge).
If the test file is shuffled, smoothing will not help.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VAL_LABELS = REPO_ROOT / "dataset" / "processed" / "features" / "validation_Bag.npz"

LABEL_MAP = {1: "Still", 2: "Walking", 3: "Run", 4: "Bike",
             5: "Car",   6: "Bus",     7: "Train", 8: "Metro"}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_labels(path: Path) -> np.ndarray:
    """Read a submission file → 1-D array of int labels (1–8)."""
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(int(line.split(",")[0]))
    return np.array(labels, dtype=np.int32)


def _write_submission(labels: np.ndarray, path: Path, win_size: int = 500) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for lbl in labels:
            f.write(",".join([str(lbl)] * win_size) + "\n")


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def majority_vote_smooth(labels: np.ndarray, window: int) -> np.ndarray:
    """
    Sliding majority vote of size `window` (must be odd).

    Each position i gets the most common label in [i - w//2, i + w//2].
    Edge positions use a smaller window (no padding with a dummy class).
    """
    if window % 2 == 0:
        window += 1   # force odd

    out = labels.copy()
    half = window // 2
    n = len(labels)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        counts = Counter(labels[lo:hi].tolist())
        out[i] = counts.most_common(1)[0][0]

    return out


def _fast_majority_vote(labels: np.ndarray, window: int) -> np.ndarray:
    """
    Vectorised majority vote using a sliding histogram.
    Falls back to the loop version for window > 51 (rare).
    """
    if window % 2 == 0:
        window += 1

    n = len(labels)
    half = window // 2
    out = np.empty(n, dtype=np.int32)

    # Build sliding histogram (8 classes, labels 1-8)
    hist = np.zeros(9, dtype=np.int32)   # index 0 unused
    for j in range(min(window, n)):
        hist[labels[j]] += 1

    # Slide
    for i in range(n):
        out[i] = int(hist[1:].argmax()) + 1

        # Remove element leaving the window
        if i - half >= 0:
            hist[labels[i - half]] -= 1
        # Add element entering the window
        if i + half + 1 < n:
            hist[labels[i + half + 1]] += 1

    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import f1_score
    scores = f1_score(y_true, y_pred, average=None, labels=list(range(1, 9)),
                      zero_division=0)
    return {LABEL_MAP[i + 1]: round(float(s), 4) for i, s in enumerate(scores)}


def _load_val_labels(path: Path) -> np.ndarray:
    """Load ground-truth validation labels from npz cache (1-based)."""
    d = np.load(path)
    y = d["y"]
    # cache stores 1-based labels directly (or 0-based if shifted)
    if y.min() == 0:
        y = y + 1
    return y.astype(np.int32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",      type=Path, required=False,
                        help="Input submission file (one line per window)")
    parser.add_argument("--output",     type=Path, default=None,
                        help="Output smoothed file (default: input_smoothW.txt)")
    parser.add_argument("--window",     type=int, default=7,
                        help="Majority-vote window size (odd; default 7)")
    parser.add_argument("--eval",       action="store_true",
                        help="Evaluate smoothed vs raw using --val-labels")
    parser.add_argument("--val-labels", type=Path, default=DEFAULT_VAL_LABELS,
                        help="Ground-truth validation labels .npz (for --eval)")
    parser.add_argument("--force",      action="store_true",
                        help="Write output even when shuffled-data guard fires")
    parser.add_argument("--sweep",      action="store_true",
                        help="Sweep window sizes 1,3,5,...,19 and print F1 table")
    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        sys.exit(0)

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}")
        sys.exit(1)

    print(f"Loading predictions from {args.input.name} ...", flush=True)
    labels_raw = _load_labels(args.input)
    print(f"  {len(labels_raw):,} windows  label range: [{labels_raw.min()}, {labels_raw.max()}]")

    # --- eval mode: load val labels ---
    y_true = None
    if args.eval or args.sweep:
        if not args.val_labels.exists():
            print(f"ERROR: val labels not found: {args.val_labels}")
            print("  You can generate them with: python scripts/precompute_features.py")
            sys.exit(1)
        y_true = _load_val_labels(args.val_labels)
        if len(y_true) != len(labels_raw):
            print(f"ERROR: label count mismatch: preds={len(labels_raw)}, "
                  f"true={len(y_true)}")
            sys.exit(1)
        f1_raw = _macro_f1(y_true, labels_raw)
        print(f"\nRaw macro-F1 : {f1_raw:.4f}")

    # --- sweep mode ---
    if args.sweep:
        print(f"\n{'W':>4}  {'Macro-F1':>9}  {'Delta':>7}  Per-class F1")
        print("-" * 80)
        for w in range(1, 20, 2):
            sm = _fast_majority_vote(labels_raw, w)
            f1 = _macro_f1(y_true, sm)
            delta = f1 - f1_raw
            pc = _per_class_f1(y_true, sm)
            run_str = "  ".join(f"{LABEL_MAP[i+1][:3]}={v:.3f}" for i, v in enumerate(pc.values()))
            sign = "+" if delta >= 0 else ""
            print(f"{w:>4}  {f1:>9.4f}  {sign}{delta:>6.4f}  {run_str}")
        print()

    # --- temporal-order warning (always shown when smoothing) ---
    _is_eval = args.eval or args.sweep
    if not _is_eval:
        print(
            "\nWARNING: Temporal smoothing assumes predictions are in recording "
            "order (window i is adjacent in time to window i+1).\n"
            "  SHL 2026 test data is shuffled — smoothing on the test submission "
            "will corrupt labels rather than improve them.\n"
            "  Use --eval / --sweep on validation predictions only."
        )

    # --- single smooth run ---
    print(f"\nSmoothing with window={args.window} ...", flush=True)
    labels_smooth = _fast_majority_vote(labels_raw, args.window)
    changed = int((labels_smooth != labels_raw).sum())
    change_rate = changed / len(labels_raw)
    print(f"  Changed {changed:,} / {len(labels_raw):,} predictions "
          f"({change_rate * 100:.1f}%)")

    # --- shuffled-data guard ---
    _SHUFFLE_THRESHOLD = 0.50
    if change_rate > _SHUFFLE_THRESHOLD:
        print(
            f"\nCRITICAL: {change_rate*100:.1f}% of predictions changed — this indicates "
            f"rows are NOT in temporal order (shuffled data).\n"
            f"  Ordered data typically changes ~10–15% of windows.\n"
            f"  Writing this file would corrupt the submission. Aborting.\n"
            f"  Use --force to override (only if you know rows are ordered)."
        )
        if not args.force:
            sys.exit(1)
        print("  --force set: writing anyway (you accepted the risk).\n")

    if y_true is not None:
        f1_sm = _macro_f1(y_true, labels_smooth)
        delta = f1_sm - f1_raw
        sign = "+" if delta >= 0 else ""
        print(f"\n  Raw    macro-F1 : {f1_raw:.4f}")
        print(f"  Smooth macro-F1 : {f1_sm:.4f}  ({sign}{delta:.4f})")
        print(f"\n  Per-class delta:")
        pc_raw = _per_class_f1(y_true, labels_raw)
        pc_sm  = _per_class_f1(y_true, labels_smooth)
        for cls in pc_raw:
            d = pc_sm[cls] - pc_raw[cls]
            sign = "+" if d >= 0 else ""
            print(f"    {cls:<10}: {pc_raw[cls]:.4f} → {pc_sm[cls]:.4f}  ({sign}{d:.4f})")

    # --- write output ---
    if args.output is None:
        stem = args.input.stem
        args.output = args.input.with_name(f"{stem}_smooth{args.window}.txt")

    _write_submission(labels_smooth, args.output)
    print(f"\nWritten: {args.output}")
    print(f"  Lines : {len(labels_smooth):,}")


if __name__ == "__main__":
    main()
