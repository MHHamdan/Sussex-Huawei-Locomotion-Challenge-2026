#!/usr/bin/env python3
"""
Stage 9 — Ensemble: InceptionTime + IMUFormer + MOMENT-hybrid XGB.

Pipeline
--------
1. Load validation windows from HDF5 (Bag position, 57 576 windows).
2. For each model, generate val logits / probabilities:
   - InceptionTime  (model.pt)       : GPU forward pass, per-window z-norm + TTA
   - IMUFormer      (best_model.pt)  : GPU forward pass, per-window z-norm + TTA
   - MOMENT+stat XGB (model.joblib)  : cached embeddings + stat features → predict_proba
3. Temperature-calibrate each model's logits to align confidence scales.
4. Weighted-average calibrated probs (uniform then weight-optimised).
5. Report per-model and ensemble validation macro-F1.
6. Optionally generate a test submission (--predict-test).

GPU usage
---------
Set CUDA_VISIBLE_DEVICES=2 before launch; inside the process the GPU appears as cuda:0.

Monitoring
----------
  GPU memory  : nvidia-smi -i 2 --query-gpu=memory.used,memory.total --format=csv,noheader
  Process ID  : ps aux | grep run_stage9
  Log file    : outputs/execution-output/stage9_ensemble.log
  Val F1 best : InceptionTime 0.7265, IMUFormer 0.7125, MOMENT-XGB 0.7098

Usage
-----
# Validate ensemble on GPU 2 (with TTA, weight optimisation):
CUDA_VISIBLE_DEVICES=2 python -u scripts/run_stage9_ensemble.py \\
    --tta-n 5 --device cuda:0 \\
    2>&1 | tee outputs/execution-output/stage9_ensemble.log

# Smoke test — val only, no TTA (fast):
CUDA_VISIBLE_DEVICES=2 python -u scripts/run_stage9_ensemble.py \\
    --tta-n 1 --device cuda:0

# Generate test submission after val calibration:
CUDA_VISIBLE_DEVICES=2 python -u scripts/run_stage9_ensemble.py \\
    --tta-n 5 --device cuda:0 --predict-test \\
    --output outputs/execution-output/submissions/FeatureFlyers_ensemble_s9.txt \\
    2>&1 | tee outputs/execution-output/stage9_ensemble_test.log

Expected runtimes (GPU 2, batch=512, TTA n=5)
----------------------------------------------
  Val  (57 576 windows × 3 models × 5 TTA passes) : ~8 min
  Test (92 726 windows × 3 models × 5 TTA passes) : ~13 min
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HDF5_PATH   = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
EMB_DIR     = REPO_ROOT / "dataset" / "processed" / "embeddings"
FEAT_DIR    = REPO_ROOT / "dataset" / "processed" / "features"
OUT_DIR     = REPO_ROOT / "outputs" / "execution-output"
N_CLASSES   = 8
WIN_SIZE    = 500
LABEL_MAP   = {0:"Still",1:"Walking",2:"Run",3:"Bike",
               4:"Car",5:"Bus",6:"Train",7:"Metro"}

# ---------------------------------------------------------------------------
# Default model paths (relative to REPO_ROOT)
# ---------------------------------------------------------------------------
DEFAULT_INCEPTION = OUT_DIR / "inception_posPool_nb32_d6_sfull_ep100_bs512"
DEFAULT_IMUFORMER = OUT_DIR / "imuformer_d128tf2_posPool_sfull_strat40000_ep60"
DEFAULT_MOMENT_XGB = OUT_DIR / "foundation_moment_posPool_mean_pool_normperwindow_xgb_hybrid_sfull_strat40000"


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _infer_pytorch_model(
    model,
    raw_windows: np.ndarray,
    device,
    norm_fn,
    batch_size: int = 512,
    tta_n: int = 1,
    jitter_std: float = 0.02,
    scale_lo: float = 0.9,
    scale_hi: float = 1.1,
) -> np.ndarray:
    """
    Run batched inference and return raw logits (N, K).

    raw_windows : (N, 9, 500) float32 numpy, channels-first
    norm_fn     : callable (tensor → tensor), e.g. norm_perwindow
    tta_n       : 1 = no TTA; >1 = average over tta_n augmented passes
    Returns     : (N, K) float32 logits (not softmax-ed — needed for temperature scaling)
    """
    import torch
    from featureflyers_shl.ensemble.tta import augment_batch

    model.eval()
    N = len(raw_windows)
    all_logits = np.zeros((N, N_CLASSES), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end  = min(start + batch_size, N)
            x_np = raw_windows[start:end]
            x    = torch.from_numpy(x_np).to(device)
            x    = norm_fn(x)

            batch_logits = np.zeros((end - start, N_CLASSES), dtype=np.float64)
            for _ in range(tta_n):
                xi = augment_batch(x, jitter_std, scale_lo, scale_hi) if tta_n > 1 else x
                logits = model(xi).float().cpu().numpy()
                batch_logits += logits
            all_logits[start:end] = batch_logits / tta_n

    return all_logits.astype(np.float32)


def load_inception(run_dir: Path, device) -> tuple:
    """Load InceptionTime model. Returns (model, norm_fn, 'inception')."""
    import torch
    from featureflyers_shl.models.inception import InceptionTime
    from featureflyers_shl.ensemble.tta import norm_perwindow

    cfg = json.loads((run_dir / "config.json").read_text())
    model = InceptionTime(
        n_channels=9, n_classes=N_CLASSES,
        nb_filters=cfg["nb_filters"], depth=cfg["depth"],
        bottleneck=cfg["bottleneck"], dropout=0.0,
    )
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(device)
    print(f"  [InceptionTime] loaded — {cfg['n_params']:,} params  F1={cfg.get('best_f1','?')}")
    return model, norm_perwindow, "InceptionTime"


def load_imuformer(run_dir: Path, device) -> tuple:
    """Load IMUFormer model. Returns (model, norm_fn, 'IMUFormer')."""
    import torch
    from featureflyers_shl.models.imu_former import IMUFormer
    from featureflyers_shl.ensemble.tta import norm_perwindow

    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu")
    cfg = checkpoint["config"]
    model = IMUFormer(
        n_classes=N_CLASSES,
        d=cfg.get("d", 128),
        n_heads=cfg.get("n_heads", 4),
        n_tf_layers=cfg.get("n_tf_layers", 2),
        dropout=0.0,   # disable at inference
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(device)
    n_params = model.n_params
    print(f"  [IMUFormer    ] loaded — {n_params:,} params")
    return model, norm_perwindow, "IMUFormer"


def load_moment_xgb_probs(run_dir: Path, split: str) -> np.ndarray | None:
    """
    Generate probabilities from the MOMENT+stat XGB using cached embeddings/features.

    Requires:
      EMB_DIR / f"{split}_Bag_moment_normperwindow_mean_pool.npz"  (1024-dim)
      FEAT_DIR / f"{split}_Bag.npz"                                (354-dim)

    Returns (N, 8) float32 predict_proba, or None if cache is missing.
    """
    import joblib
    from scipy.special import softmax

    emb_path  = EMB_DIR  / f"{split}_Bag_moment_normperwindow_mean_pool.npz"
    feat_path = FEAT_DIR / f"{split}_Bag.npz"

    for p in (emb_path, feat_path, run_dir / "model.joblib"):
        if not p.exists():
            print(f"  [MOMENT-XGB   ] SKIP — missing {p.name}")
            return None

    bundle = joblib.load(run_dir / "model.joblib")
    clf    = bundle["sklearn_model"]

    X_emb  = np.load(emb_path)["X"].astype(np.float32)   # (N, 1024)
    X_feat = np.load(feat_path)["X"].astype(np.float32)  # (N, 354)
    X      = np.hstack([X_emb, X_feat])                  # (N, 1378)

    print(f"  [MOMENT-XGB   ] loaded — {X.shape[1]}-dim features  N={X.shape[0]:,}")
    probs = clf.predict_proba(X).astype(np.float32)       # (N, 8)
    return probs


def write_submission(predictions: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {len(predictions):,} lines → {out_path} …", flush=True)
    t0 = time.time()
    with open(out_path, "w") as f:
        for lbl in predictions:
            f.write(",".join([str(int(lbl))] * WIN_SIZE) + "\n")
    mb = out_path.stat().st_size / 1024 / 1024
    print(f"Saved — {mb:.1f} MB  ({time.time()-t0:.1f}s)", flush=True)


def verify_submission(out_path: Path, n: int) -> bool:
    lines = out_path.read_text().splitlines()
    ok = (len(lines) == n
          and len(lines[0].split(",")) == WIN_SIZE
          and all(1 <= int(v) <= 8 for v in lines[0].split(",")))
    print(f"  [{'PASS' if ok else 'FAIL'}] {n} lines × {WIN_SIZE} predictions, labels 1–8")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device",         default="cuda:0")
    parser.add_argument("--batch-size",     type=int,   default=512)
    parser.add_argument("--tta-n",          type=int,   default=5,
                        help="TTA passes for PyTorch models (1=no TTA)")
    parser.add_argument("--jitter-std",     type=float, default=0.02)
    parser.add_argument("--scale-lo",       type=float, default=0.9)
    parser.add_argument("--scale-hi",       type=float, default=1.1)
    parser.add_argument("--inception-dir",  type=Path,  default=DEFAULT_INCEPTION)
    parser.add_argument("--imuformer-dir",  type=Path,  default=DEFAULT_IMUFORMER)
    parser.add_argument("--moment-xgb-dir", type=Path,  default=DEFAULT_MOMENT_XGB)
    parser.add_argument("--no-inception",   action="store_true")
    parser.add_argument("--no-imuformer",   action="store_true")
    parser.add_argument("--no-moment-xgb",  action="store_true")
    parser.add_argument("--predict-test",   action="store_true",
                        help="After val calibration, run inference on test set")
    parser.add_argument("--output", type=Path,
                        default=OUT_DIR / "submissions" / "FeatureFlyers_ensemble_s9.txt")
    args = parser.parse_args()

    import torch
    import h5py  # used for test inference path
    from featureflyers_shl.ensemble.calibration import find_temperature, apply_temperature
    from featureflyers_shl.ensemble.combine import (
        weighted_average, evaluate_probs, find_optimal_weights, print_ensemble_report,
    )

    device = torch.device(args.device)
    t_start = time.time()

    print(f"\nStage 9 — Ensemble")
    print(f"  device   : {device}")
    print(f"  TTA n    : {args.tta_n}  (jitter={args.jitter_std}, scale=[{args.scale_lo},{args.scale_hi}])")
    print(f"  batch    : {args.batch_size}")
    print(f"  test run : {args.predict_test}\n")

    # ------------------------------------------------------------------
    # 1. Load val raw windows for PyTorch models
    # ------------------------------------------------------------------
    # Validation is a flat HDF5 stream — use SHLWindowDataset(preload=True)
    # which does one sequential read and keeps (N, 500, 9) in RAM.
    from featureflyers_shl.data.dataset import SHLWindowDataset
    print("Loading val windows via SHLWindowDataset (preload=True) …", flush=True)
    t0 = time.time()
    ds_val = SHLWindowDataset(HDF5_PATH, split="validation", position="Bag",
                              sample_limit=None, seed=42, preload=True)
    # ds_val._X : (N_val, 500, 9) float32 — transpose to channels-first
    raw_val = ds_val._X.transpose(0, 2, 1).astype(np.float32)  # (N_val, 9, 500)
    y_val   = ds_val._labels.astype(np.int64)                  # 0-based
    N_val   = len(raw_val)
    print(f"  {N_val:,} windows  labels 0–{y_val.max()}  ({time.time()-t0:.1f}s)\n")

    # ------------------------------------------------------------------
    # 2. Generate val logits / probs for each model
    # ------------------------------------------------------------------
    model_names:  list[str]        = []
    val_logits:   list[np.ndarray] = []   # (N_val, 8) raw logits — for calibration
    val_probs_xgb: np.ndarray | None = None  # XGB already gives probs directly

    # --- InceptionTime ---
    if not args.no_inception:
        print("Generating val logits — InceptionTime …", flush=True)
        t0 = time.time()
        model, norm_fn, name = load_inception(args.inception_dir.resolve(), device)
        logits = _infer_pytorch_model(
            model, raw_val, device, norm_fn,
            batch_size=args.batch_size, tta_n=args.tta_n,
            jitter_std=args.jitter_std, scale_lo=args.scale_lo, scale_hi=args.scale_hi,
        )
        val_logits.append(logits)
        model_names.append(name)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        f1_raw, _ = evaluate_probs(
            __import__("scipy.special", fromlist=["softmax"]).softmax(logits, axis=1), y_val)
        print(f"  done in {time.time()-t0:.1f}s  raw F1={f1_raw:.4f}\n", flush=True)

    # --- IMUFormer ---
    if not args.no_imuformer:
        print("Generating val logits — IMUFormer …", flush=True)
        t0 = time.time()
        model, norm_fn, name = load_imuformer(args.imuformer_dir.resolve(), device)
        logits = _infer_pytorch_model(
            model, raw_val, device, norm_fn,
            batch_size=args.batch_size, tta_n=args.tta_n,
            jitter_std=args.jitter_std, scale_lo=args.scale_lo, scale_hi=args.scale_hi,
        )
        val_logits.append(logits)
        model_names.append(name)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        f1_raw, _ = evaluate_probs(
            __import__("scipy.special", fromlist=["softmax"]).softmax(logits, axis=1), y_val)
        print(f"  done in {time.time()-t0:.1f}s  raw F1={f1_raw:.4f}\n", flush=True)

    # --- MOMENT+stat XGB ---
    if not args.no_moment_xgb:
        print("Generating val probs — MOMENT+stat XGB …", flush=True)
        t0 = time.time()
        xgb_probs = load_moment_xgb_probs(args.moment_xgb_dir.resolve(), split="validation")
        if xgb_probs is not None:
            val_probs_xgb = xgb_probs
            model_names.append("MOMENT-XGB")
            f1_xgb, _ = evaluate_probs(xgb_probs, y_val)
            print(f"  done in {time.time()-t0:.1f}s  raw F1={f1_xgb:.4f}\n", flush=True)

    if not model_names:
        print("ERROR: no models loaded — use --no-* flags to disable specific models only.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Temperature calibration (per model)
    # ------------------------------------------------------------------
    print("Temperature calibration …")
    calibrated_probs: list[np.ndarray] = []

    for i, (name, logits) in enumerate(zip(model_names[:len(val_logits)], val_logits)):
        T = find_temperature(logits, y_val)
        probs = apply_temperature(logits, T)
        f1_cal, acc_cal = evaluate_probs(probs, y_val)
        print(f"  {name:<14}  T={T:.4f}  F1={f1_cal:.4f}  Acc={acc_cal:.4f}")
        calibrated_probs.append(probs)

    if val_probs_xgb is not None:
        # XGB predict_proba is already probability (no logits); use T=1 by default
        f1_xgb, acc_xgb = evaluate_probs(val_probs_xgb, y_val)
        print(f"  {'MOMENT-XGB':<14}  T=1.000 (probs)  F1={f1_xgb:.4f}  Acc={acc_xgb:.4f}")
        calibrated_probs.append(val_probs_xgb)

    # ------------------------------------------------------------------
    # 4. Ensemble — uniform then weight-optimised
    # ------------------------------------------------------------------
    print("\nEnsemble (uniform weights) …")
    probs_uniform = weighted_average(calibrated_probs)
    f1_uni, acc_uni = evaluate_probs(probs_uniform, y_val)
    print(f"  Macro-F1={f1_uni:.4f}  Accuracy={acc_uni:.4f}")

    print("\nEnsemble (weight optimisation) …")
    opt_weights = find_optimal_weights(calibrated_probs, y_val)
    probs_opt   = weighted_average(calibrated_probs, weights=opt_weights.tolist())
    f1_opt, acc_opt = evaluate_probs(probs_opt, y_val)
    for name, w in zip(model_names, opt_weights):
        print(f"  {name:<14}  w={w:.4f}")
    print(f"  Macro-F1={f1_opt:.4f}  Accuracy={acc_opt:.4f}")

    # Use the better of uniform / optimised
    best_probs = probs_opt if f1_opt >= f1_uni else probs_uniform
    best_f1    = max(f1_opt, f1_uni)
    best_label = "weight-optimised" if f1_opt >= f1_uni else "uniform"
    print(f"\nBest ensemble: {best_label}  Macro-F1={best_f1:.4f}")

    # Detailed report for best ensemble
    print_ensemble_report(best_probs, y_val, LABEL_MAP,
                          title=f"Stage 9 Ensemble ({best_label})")

    # Save val predictions for temporal-smoothing sweep (Stage 7b)
    _val_preds_path = args.output.parent / "val_preds_ensemble.txt"
    _val_preds_1based = best_probs.argmax(axis=1).astype(np.int64) + 1
    write_submission(_val_preds_1based, _val_preds_path)
    print(f"\nVal preds saved → {_val_preds_path}  (use with smooth_predictions.py --sweep)")

    # Summary table
    print(f"\n{'Model':<16}  {'Val F1':>8}  {'Val Acc':>8}")
    print("-" * 38)
    for i, name in enumerate(model_names):
        p = calibrated_probs[i]
        f1, acc = evaluate_probs(p, y_val)
        print(f"{name:<16}  {f1:>8.4f}  {acc:>8.4f}")
    print("-" * 38)
    print(f"{'Uniform avg':<16}  {f1_uni:>8.4f}  {acc_uni:>8.4f}")
    print(f"{'Opt weighted':<16}  {f1_opt:>8.4f}  {acc_opt:>8.4f}")

    total_val_time = time.time() - t_start
    print(f"\nVal phase done in {total_val_time:.1f}s")

    # ------------------------------------------------------------------
    # 5. Optionally generate test submission
    # ------------------------------------------------------------------
    if not args.predict_test:
        print("\n[--predict-test not set] Skipping test inference.")
        print(f"To generate submission:\n"
              f"  CUDA_VISIBLE_DEVICES=2 python -u scripts/run_stage9_ensemble.py "
              f"--tta-n {args.tta_n} --device cuda:0 --predict-test \\\n"
              f"    --output {args.output}")
        return

    print("\n" + "="*60)
    print("TEST INFERENCE")
    print("="*60)
    t_test = time.time()

    # Load test windows: test/data is pre-windowed (92726, 500, 9) → (N, 9, 500)
    import h5py
    print("\nLoading test windows from HDF5 …", flush=True)
    with h5py.File(HDF5_PATH, "r") as hf:
        raw_test = hf["test"]["data"][:].astype(np.float32)
    raw_test = raw_test.transpose(0, 2, 1)
    N_test   = len(raw_test)
    print(f"  {N_test:,} test windows loaded\n", flush=True)

    # Generate test logits / probs for each model using same calibration T
    test_probs_all: list[np.ndarray] = []

    pt_model_dirs = []
    if not args.no_inception:  pt_model_dirs.append(("inception", args.inception_dir.resolve()))
    if not args.no_imuformer:  pt_model_dirs.append(("imuformer", args.imuformer_dir.resolve()))

    for (mtype, run_dir), logits_val, name in zip(
            pt_model_dirs, val_logits, model_names[:len(val_logits)]):
        print(f"Test inference — {name} …", flush=True)
        t0 = time.time()
        if mtype == "inception":
            model, norm_fn, _ = load_inception(run_dir, device)
        else:
            model, norm_fn, _ = load_imuformer(run_dir, device)

        T_model = find_temperature(logits_val, y_val)
        test_logits = _infer_pytorch_model(
            model, raw_test, device, norm_fn,
            batch_size=args.batch_size, tta_n=args.tta_n,
            jitter_std=args.jitter_std, scale_lo=args.scale_lo, scale_hi=args.scale_hi,
        )
        test_probs = apply_temperature(test_logits, T_model)
        test_probs_all.append(test_probs)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  done in {time.time()-t0:.1f}s\n", flush=True)

    # MOMENT-XGB test probs
    if not args.no_moment_xgb and val_probs_xgb is not None:
        print("Test inference — MOMENT-XGB (from test embeddings) …", flush=True)
        xgb_test_probs = load_moment_xgb_probs(args.moment_xgb_dir.resolve(), split="test")
        if xgb_test_probs is not None:
            test_probs_all.append(xgb_test_probs)
        else:
            print("  MOMENT-XGB test embeddings not cached — skipping for test.")

    # Combine with optimal weights (reuse from val, pad if XGB test missing)
    if len(test_probs_all) < len(calibrated_probs):
        effective_weights = opt_weights[:len(test_probs_all)]
        effective_weights = effective_weights / effective_weights.sum()
    else:
        effective_weights = opt_weights

    test_ensemble = weighted_average(test_probs_all, weights=effective_weights.tolist())
    preds_0based  = test_ensemble.argmax(axis=1)
    preds_1based  = preds_0based.astype(np.int64) + 1  # 1-indexed for submission

    from collections import Counter
    dist = {LABEL_MAP[k-1]: int(v)
            for k, v in sorted(Counter(preds_1based.tolist()).items())}
    print(f"Prediction distribution: {dist}\n")

    write_submission(preds_1based, args.output.resolve())
    verify_submission(args.output.resolve(), n=N_test)

    print(f"\nTotal elapsed: {time.time()-t_start:.1f}s")
    print(f"Best val Macro-F1 : {best_f1:.4f}  ({best_label})")
    print(f"Submission        : {args.output}")


if __name__ == "__main__":
    main()
