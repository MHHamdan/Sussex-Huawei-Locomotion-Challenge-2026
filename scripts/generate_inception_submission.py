#!/usr/bin/env python3
"""
Generate SHL 2026 submission from a trained InceptionTime model.

Reads model.pt + config.json from a Stage 8 run directory, runs batched inference
on the 92 726 pre-windowed test windows, and writes the submission file.

Submission format: 92 726 lines, each with 500 comma-separated integers (1-8).
One label per window, replicated 500 times.

Usage
-----
# Full submission
python scripts/generate_inception_submission.py \
    --run-dir outputs/execution-output/inception_posPool_nb32_d6_sfull_ep100_bs512 \
    --output  outputs/execution-output/submissions/FeatureFlyers_inception_pool_full.txt \
    --device  cuda:0

# Smoke test (first 1000 windows only)
python scripts/generate_inception_submission.py \
    --run-dir outputs/execution-output/inception_posPool_nb32_d6_sfull_ep100_bs512 \
    --output  outputs/execution-output/submissions/FeatureFlyers_inception_smoke.txt \
    --limit   1000 \
    --device  cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HDF5_PATH  = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
WIN_SIZE   = 500
N_CLASSES  = 8
LABEL_MAP  = {0:"Still",1:"Walking",2:"Run",3:"Bike",4:"Car",5:"Bus",6:"Train",7:"Metro"}
INFER_BATCH = 512


def _norm_perwindow(x: "torch.Tensor") -> "torch.Tensor":
    x = x.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    mu  = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (x - mu) / std


def write_submission(predictions: np.ndarray, out_path: Path) -> None:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Writing {len(predictions):,} lines → {out_path.relative_to(REPO_ROOT)} …", flush=True)
    t0 = time.time()
    with open(out_path, "w") as f:
        for lbl in predictions:
            f.write(",".join([str(int(lbl))] * WIN_SIZE) + "\n")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved — {size_mb:.1f} MB  ({time.time()-t0:.1f}s)", flush=True)


def verify_submission(out_path: Path, expected_lines: int) -> bool:
    lines = out_path.read_text().splitlines()
    if len(lines) != expected_lines:
        print(f"  [FAIL] Expected {expected_lines} lines, got {len(lines)}")
        return False
    first = lines[0].split(",")
    if len(first) != WIN_SIZE:
        print(f"  [FAIL] Expected {WIN_SIZE} values/line, got {len(first)}")
        return False
    vals = [int(v) for v in first]
    if not all(1 <= v <= 8 for v in vals):
        print(f"  [FAIL] Label out of 1–8: {set(vals)}")
        return False
    print(f"  [PASS] Format OK: {expected_lines} lines × {WIN_SIZE} predictions, labels 1–8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Path to Stage 8 run dir containing model.pt + config.json")
    parser.add_argument("--output",  type=Path,
                        default=REPO_ROOT / "outputs" / "execution-output" / "submissions"
                                / "FeatureFlyers_inception_pool_full.txt")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Process only first N windows (smoke test)")
    parser.add_argument("--device",  default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=INFER_BATCH)
    args = parser.parse_args()

    args.run_dir = args.run_dir.resolve() if not args.run_dir.is_absolute() else args.run_dir
    args.output  = args.output.resolve()  if not args.output.is_absolute()  else args.output
    model_pt   = args.run_dir / "model.pt"
    config_json = args.run_dir / "config.json"

    for p in (model_pt, config_json, HDF5_PATH):
        if not p.exists():
            print(f"ERROR: not found: {p}"); sys.exit(1)

    cfg = json.loads(config_json.read_text())
    limit_str = str(args.limit) if args.limit is not None else "all"

    print(f"\nInceptionTime test inference")
    print(f"  run dir  : {args.run_dir.relative_to(REPO_ROOT)}")
    print(f"  model    : nb_filters={cfg['nb_filters']}  depth={cfg['depth']}  "
          f"bottleneck={cfg['bottleneck']}  params={cfg['n_params']:,}")
    print(f"  device   : {args.device}")
    print(f"  windows  : {limit_str}")
    print(f"  output   : {args.output.relative_to(REPO_ROOT)}\n")

    import torch
    from featureflyers_shl.models.inception import InceptionTime

    device = torch.device(args.device)

    model = InceptionTime(
        n_channels=9, n_classes=N_CLASSES,
        nb_filters=cfg["nb_filters"], depth=cfg["depth"],
        bottleneck=cfg["bottleneck"], dropout=0.0,
    )
    state = torch.load(model_pt, map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(device)
    print(f"  Loaded model.pt  ({cfg['n_params']:,} params)\n")

    import h5py
    print("Loading test data from HDF5 …", flush=True)
    t0 = time.time()
    with h5py.File(HDF5_PATH, "r") as hf:
        raw = hf["test"]["data"][:]   # (92726, 500, 9)  float32
    n_total = raw.shape[0]
    n_windows = min(n_total, args.limit) if args.limit is not None else n_total
    raw = raw[:n_windows]             # (N, 500, 9)
    # Permute to channels-first: (N, 9, 500)
    data = raw.transpose(0, 2, 1).astype(np.float32)
    print(f"  {n_windows:,} windows loaded  shape={data.shape}  ({time.time()-t0:.1f}s)\n")

    print("Running inference …", flush=True)
    t_infer = time.time()
    all_preds: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, n_windows, args.batch_size):
            end = min(start + args.batch_size, n_windows)
            x = torch.from_numpy(data[start:end]).to(device)
            x = _norm_perwindow(x)
            logits = model(x)
            preds  = logits.argmax(1).cpu().numpy()   # 0-based
            all_preds.append(preds)
            if start % (args.batch_size * 20) == 0:
                pct = end / n_windows * 100
                print(f"  {end:,}/{n_windows:,}  ({pct:.0f}%)  "
                      f"{time.time()-t_infer:.0f}s", flush=True)

    preds_0based = np.concatenate(all_preds)           # (N,) labels 0-7
    preds_1based = preds_0based.astype(np.int64) + 1  # submission labels 1-8
    elapsed_infer = time.time() - t_infer
    print(f"  Inference done in {elapsed_infer:.1f}s  "
          f"({n_windows / elapsed_infer:.0f} windows/s)\n")

    dist = {LABEL_MAP.get(k-1, str(k)): int(v)
            for k, v in sorted(Counter(preds_1based.tolist()).items())}
    print(f"  Prediction distribution: {dist}\n")

    write_submission(preds_1based, args.output)
    verify_submission(args.output, expected_lines=n_windows)
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
