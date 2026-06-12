#!/usr/bin/env python3
"""
Generate a challenge submission from a trained model.

Reads the pre-windowed test set from HDF5 (92 726 windows × 500 samples × 9 sensors),
extracts features per window, predicts one label per window, then replicates
that label for all 500 samples within the window to match the submission format.

Submission format (required by SHL 2026):
    92 726 lines, each containing 500 comma-separated integer labels.
    Total predictions: 92 726 × 500 = 46 363 000.

Usage:
    python scripts/generate_submission.py \\
        --model-path outputs/baseline/<run>/model.joblib \\
        --output outputs/submissions/FeatureFlyers_predictions.txt

    # Smoke test — first 1000 windows only:
    python scripts/generate_submission.py \\
        --model-path outputs/baseline/<run>/model.joblib \\
        --output outputs/submissions/smoke.txt \\
        --limit 1000

    # Preview only (no file written):
    python scripts/generate_submission.py \\
        --model-path outputs/baseline/<run>/model.joblib \\
        --dry-run
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HDF5_PATH = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
DEFAULT_OUT = REPO_ROOT / "outputs" / "submissions" / "FeatureFlyers_predictions.txt"

N_TEST_WINDOWS = 92_726
WIN_SIZE = 500
BATCH_SIZE = 512   # windows processed at once to control peak RAM


def load_model(model_path: Path):
    try:
        import joblib
    except ImportError:
        print("ERROR: joblib not installed — run: pip install joblib")
        sys.exit(1)
    bundle = joblib.load(model_path)
    return bundle


def extract_test_features(hf, fft_top_k: int, limit: int | None = None) -> np.ndarray:
    """
    Extract features from test windows in batches.

    Test HDF5 shape: (92726, 500, 9)
    Returns: (n_windows, n_features) where n_windows = limit or 92726.
    """
    from featureflyers_shl.features.statistical import extract_batch, n_features

    ds = hf["test"]["data"]   # (N_WINDOWS, WIN_SIZE, 9)
    total_windows = ds.shape[0]
    n_windows = min(total_windows, limit) if limit is not None else total_windows
    nf = n_features(fft_top_k)
    X = np.empty((n_windows, nf), dtype=np.float32)

    print(f"  Test data shape : {total_windows:,} × {WIN_SIZE} × 9 (HDF5)")
    print(f"  Processing      : {n_windows:,} windows  fft_top_k={fft_top_k}", flush=True)
    t0 = time.time()

    for start in range(0, n_windows, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_windows)
        batch = ds[start: end, :, :]   # (B, 500, 9)
        X[start: end] = extract_batch(batch.astype(np.float32), fft_top_k)
        if start % (BATCH_SIZE * 20) == 0:
            pct = end / n_windows * 100
            print(f"    {end:,}/{n_windows:,}  ({pct:.0f}%)  "
                  f"{time.time()-t0:.0f}s", flush=True)

    print(f"  Features done   : {time.time()-t0:.1f}s  shape={X.shape}")
    return X


def write_submission(predictions: np.ndarray, out_path: Path) -> None:
    """
    Write predictions to the submission file.

    predictions : (N_WINDOWS,) integer labels — one per window
    Output      : N_WINDOWS lines, each with WIN_SIZE comma-separated copies
                  of the window prediction.
    """
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Writing {out_path.relative_to(REPO_ROOT)} …", flush=True)
    t0 = time.time()

    with open(out_path, "w") as f:
        for lbl in predictions:
            line = ",".join([str(int(lbl))] * WIN_SIZE)
            f.write(line + "\n")

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved — {size_mb:.1f} MB  ({time.time()-t0:.1f}s)")


def verify_submission(out_path: Path, expected_lines: int) -> bool:
    """Quick format check on the written file."""
    lines = out_path.read_text().splitlines()
    if len(lines) != expected_lines:
        print(f"  [FAIL] Expected {expected_lines} lines, got {len(lines)}")
        return False
    first = lines[0].split(",")
    if len(first) != WIN_SIZE:
        print(f"  [FAIL] Expected {WIN_SIZE} values/line, got {len(first)}")
        return False
    try:
        vals = [int(v) for v in first]
        if not all(1 <= v <= 8 for v in vals):
            print(f"  [FAIL] Label values out of range 1–8: {set(vals)}")
            return False
    except ValueError as e:
        print(f"  [FAIL] Non-integer label: {e}")
        return False
    print(f"  [PASS] Format OK: {expected_lines} lines × {WIN_SIZE} predictions")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to model.joblib saved by train_baseline.py")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT,
                        help=f"Output path (default: {DEFAULT_OUT.relative_to(REPO_ROOT)})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N test windows for smoke testing.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prediction stats without writing file")
    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"ERROR: model not found: {args.model_path}")
        sys.exit(1)

    try:
        import h5py
    except ImportError:
        print("ERROR: h5py not installed"); sys.exit(1)

    if not HDF5_PATH.exists():
        print(f"ERROR: {HDF5_PATH} not found")
        sys.exit(1)

    mode = "dry-run" if args.dry_run else "writing"
    limit_str = str(args.limit) if args.limit is not None else "all"
    print(f"\nGenerating submission")
    print(f"  Model  : {args.model_path}")
    print(f"  HDF5   : {HDF5_PATH.relative_to(REPO_ROOT)}")
    print(f"  Output : {args.output}")
    print(f"  Windows: {limit_str}")
    print(f"  Mode   : {mode}\n")

    t_total = time.time()

    # Load model bundle
    bundle = load_model(args.model_path)
    scaler    = bundle["scaler"]
    clf       = bundle["clf"]
    fft_top_k = bundle.get("fft_top_k", 20)
    fusion    = bundle.get("fusion", "none")
    positions = bundle.get("positions", ["Bag"])
    print(f"  Loaded: {clf.__class__.__name__}  "
          f"fusion={fusion}  positions={positions}  fft_top_k={fft_top_k}\n")

    with h5py.File(HDF5_PATH, "r") as hf:
        if "test" not in hf or "data" not in hf["test"]:
            print("ERROR: test/data not found in HDF5.")
            sys.exit(1)

        X_test = extract_test_features(hf, fft_top_k, limit=args.limit)

    # Scale
    X_test_s = scaler.transform(X_test)

    # Predict
    print("\n  Predicting …", flush=True)
    t_pred = time.time()
    preds = clf.predict(X_test_s)   # (N_WINDOWS,)
    print(f"  Done in {time.time()-t_pred:.1f}s")

    # Distribution
    from collections import Counter
    LABEL_MAP = {1:"Still",2:"Walking",3:"Run",4:"Bike",
                 5:"Car",6:"Bus",7:"Train",8:"Metro"}
    dist = {LABEL_MAP.get(k, k): v for k, v in
            sorted(Counter(preds.tolist()).items())}
    print(f"\n  Prediction distribution: {dist}")

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return

    print()
    write_submission(preds, args.output)
    verify_submission(args.output, expected_lines=len(preds))

    print(f"\nTotal elapsed: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
