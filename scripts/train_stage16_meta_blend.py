#!/usr/bin/env python3
"""
Stage 16 — LightGBM meta-blend ensemble for SHL 2026.

Blending (learned ensemble): for each registered model, run inference on the
validation set to get per-class probabilities, stack them as meta-features,
and train a LightGBM classifier.  Optionally generate a test submission.

Phase A (run immediately — 3 Stage-12 models):
  Features: 3 models × 8 class probs = 24 features
  Meta-learner: LightGBM multiclass (8 classes)

Phase B (5 models, MiniRocket excluded — F1=0.9438):
  Re-run this script with --spectrogram-dir / --resnet1d-dir.

Phase C (6 models, + MVPF — run after Stage 17 finishes):
  Add --mvpf-dir to include MVPF. Note: test inference for MVPF repeats
  the single available test position across all 4 slots (cross-position
  transformer degenerates to averaging; encoder backbone still classifies).

Blending vs. stacking
  Proper stacking uses OOF (out-of-fold) train predictions.  Here we use the
  validation set (57 576 windows) as the blend set — this is a valid "holdout
  blending" approach for a single held-out split and gives good results in
  practice.  The meta-learner uses an internal 80/20 train/test split of the
  validation set to avoid overfitting to the blend set.

Usage
-----
# Phase A — 3 Stage-12 models (runs immediately on GPU 3)
CUDA_VISIBLE_DEVICES=3 nohup python -u scripts/train_stage16_meta_blend.py \\
    --device cuda:0 \\
    --inception-dir outputs/execution-output/inception_posPool_nb32_d6_sfull_focal_g2.0_balsampler_ep100_bs512 \\
    --imuformer-dir outputs/execution-output/imuformer_d128tf2_posPool_sfull_strat40000_focal_g2.0_balsampler_nocw_ep60 \\
    --predict-test \\
    --output outputs/execution-output/submissions/FeatureFlyers_blend_s16a_lgbm.txt \\
    > outputs/execution-output/logs/stage16_gpu3_meta_blend_phaseA.log 2>&1 &

# Phase B — all 6 models (run after Stages 13-15 finish)
CUDA_VISIBLE_DEVICES=3 nohup python -u scripts/train_stage16_meta_blend.py \\
    --device cuda:0 \\
    --inception-dir   outputs/execution-output/inception_posPool_nb32_d6_sfull_focal_g2.0_balsampler_ep100_bs512 \\
    --imuformer-dir   outputs/execution-output/imuformer_d128tf2_posPool_sfull_strat40000_focal_g2.0_balsampler_nocw_ep60 \\
    --minirocket-dir  outputs/execution-output/minirocket_posPool_... \\
    --spectrogram-dir outputs/execution-output/spectrogramcnn_posPool_... \\
    --resnet1d-dir    outputs/execution-output/resnet1d_posPool_... \\
    --predict-test \\
    --output outputs/execution-output/submissions/FeatureFlyers_blend_s16b_lgbm.txt \\
    > outputs/execution-output/logs/stage16_gpu3_meta_blend_phaseB.log 2>&1 &

# Phase C — 6 models (run after Stage 17 MVPF finishes)
CUDA_VISIBLE_DEVICES=3 nohup python -u scripts/train_stage16_meta_blend.py \\
    --device cuda:0 \\
    --inception-dir   outputs/execution-output/inception_posPool_nb32_d6_sfull_focal_g2.0_balsampler_ep100_bs512 \\
    --imuformer-dir   outputs/execution-output/imuformer_d128tf2_posPool_sfull_strat40000_focal_g2.0_balsampler_nocw_ep60 \\
    --spectrogram-dir outputs/execution-output/spectrogramcnn_posPool_nfft64_hop16_sfull_focal_g2.0_balsampler_ep80_bs256 \\
    --resnet1d-dir    outputs/execution-output/resnet1d_posPool_f64_sfull_focal_g2.0_balsampler_ep100_bs512 \\
    --mvpf-dir        outputs/execution-output/mvpf_4pos_fd256_bf64_h4tf2_sfull_focal_g2.0_balsampler_aug_ep80_bs256 \\
    --predict-test \\
    --output outputs/execution-output/submissions/FeatureFlyers_blend_s16c_lgbm.txt \\
    > outputs/execution-output/logs/stage16_gpu3_meta_blend_phaseC.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HDF5_PATH    = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
EMB_DIR      = REPO_ROOT / "dataset" / "processed" / "embeddings"
FEAT_DIR     = REPO_ROOT / "dataset" / "processed" / "features"
OUT_DIR      = REPO_ROOT / "outputs" / "execution-output"
N_CLASSES    = 8
WIN_SIZE     = 500
LABEL_MAP    = {0:"Still",1:"Walking",2:"Run",3:"Bike",4:"Car",5:"Bus",6:"Train",7:"Metro"}


# ---------------------------------------------------------------------------
# Model loaders (mirrors run_stage9_ensemble.py)
# ---------------------------------------------------------------------------

def _norm_perwindow(x):
    import torch
    x = x.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    mu  = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (x - mu) / std


def _infer_pytorch(model, raw_windows, device, batch_size=512, tta_n=3,
                   jitter_std=0.02, scale_lo=0.9, scale_hi=1.1):
    import torch
    from featureflyers_shl.ensemble.tta import augment_batch
    model.eval()
    N = len(raw_windows)
    all_logits = np.zeros((N, N_CLASSES), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end  = min(start + batch_size, N)
            x    = torch.from_numpy(raw_windows[start:end]).to(device)
            x    = _norm_perwindow(x)
            bl   = np.zeros((end - start, N_CLASSES), dtype=np.float64)
            for _ in range(tta_n):
                xi = augment_batch(x, jitter_std, scale_lo, scale_hi) if tta_n > 1 else x
                bl += model(xi).float().cpu().numpy()
            all_logits[start:end] = bl / tta_n
    return all_logits.astype(np.float32)


def _softmax(logits):
    from scipy.special import softmax
    return softmax(logits, axis=1).astype(np.float32)


def load_inception_probs(run_dir, raw_windows, device, tta_n=3):
    import torch
    from featureflyers_shl.models.inception import InceptionTime
    cfg = json.loads((run_dir / "config.json").read_text())
    model = InceptionTime(n_channels=9, n_classes=N_CLASSES,
                          nb_filters=cfg["nb_filters"], depth=cfg["depth"],
                          bottleneck=cfg["bottleneck"], dropout=0.0)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval().to(device)
    print(f"  [InceptionTime] {run_dir.name}  F1={cfg.get('best_f1','?')}")
    logits = _infer_pytorch(model, raw_windows, device, tta_n=tta_n)
    del model; torch.cuda.empty_cache() if device.type == "cuda" else None
    return _softmax(logits)


def load_imuformer_probs(run_dir, raw_windows, device, tta_n=3):
    import torch
    from featureflyers_shl.models.imu_former import IMUFormer
    ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu")
    cfg  = ckpt["config"]
    model = IMUFormer(n_classes=N_CLASSES, d=cfg.get("d",128),
                      n_heads=cfg.get("n_heads",4),
                      n_tf_layers=cfg.get("n_tf_layers",2), dropout=0.0)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    print(f"  [IMUFormer    ] {run_dir.name}")
    logits = _infer_pytorch(model, raw_windows, device, tta_n=tta_n)
    del model; torch.cuda.empty_cache() if device.type == "cuda" else None
    return _softmax(logits)


def load_minirocket_probs(run_dir, raw_windows, device, tta_n=1):
    """MiniRocket: deterministic (no TTA needed — no dropout/BN in inference mode)."""
    import torch
    from scripts.train_stage13_minirocket import MiniRocketModel
    cfg = json.loads((run_dir / "config.json").read_text())
    model = MiniRocketModel(n_kernels=cfg["n_kernels"], dilations=cfg["dilations"],
                            n_channels=9, n_classes=N_CLASSES, seed=cfg["seed"])
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval().to(device)
    print(f"  [MiniRocket   ] {run_dir.name}  F1={cfg.get('best_f1','?')}")
    logits = _infer_pytorch(model, raw_windows, device, tta_n=tta_n)
    del model; torch.cuda.empty_cache() if device.type == "cuda" else None
    return _softmax(logits)


def load_spectrogram_probs(run_dir, raw_windows, device, tta_n=3):
    import torch
    from featureflyers_shl.models.spectrogram_cnn import SpectrogramCNN
    cfg = json.loads((run_dir / "config.json").read_text())
    model = SpectrogramCNN(n_channels=9, n_classes=N_CLASSES,
                           n_fft=cfg["n_fft"], hop_length=cfg["hop_length"],
                           win_length=cfg.get("win_length", cfg["n_fft"]),
                           n_frames=cfg.get("n_frames", 32), dropout=0.0)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval().to(device)
    print(f"  [SpectrogramCNN] {run_dir.name}  F1={cfg.get('best_f1','?')}")
    logits = _infer_pytorch(model, raw_windows, device, tta_n=tta_n)
    del model; torch.cuda.empty_cache() if device.type == "cuda" else None
    return _softmax(logits)


def load_resnet1d_probs(run_dir, raw_windows, device, tta_n=3):
    import torch
    from featureflyers_shl.models.resnet1d import ResNet1D
    cfg = json.loads((run_dir / "config.json").read_text())
    model = ResNet1D(n_channels=9, n_classes=N_CLASSES,
                     base_filters=cfg.get("base_filters", 64), dropout=0.0)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval().to(device)
    print(f"  [ResNet1D     ] {run_dir.name}  F1={cfg.get('best_f1','?')}")
    logits = _infer_pytorch(model, raw_windows, device, tta_n=tta_n)
    del model; torch.cuda.empty_cache() if device.type == "cuda" else None
    return _softmax(logits)


def load_mvpf_probs(run_dir, split, hdf5_path, device, tta_n=3, jitter_std=0.02):
    """MVPF: all 4 positions → (B, 4, 9, 500).

    train/validation splits: load each position via SHLWindowDataset.
    test split: only one unlabeled array in HDF5 — repeat it 4× as a fallback
    (cross-position transformer degenerates to averaging identical encoder
    outputs; the shared encoder backbone still discriminates locomotion classes).
    """
    import torch
    from featureflyers_shl.data.dataset import SHLWindowDataset
    import h5py

    cfg = json.loads((run_dir / "config.json").read_text())
    model_class = cfg.get("model", "MVPF")
    if model_class == "MVPFv2":
        from featureflyers_shl.models.mvpf_v2 import MVPFv2
        model = MVPFv2(
            n_positions=cfg.get("n_positions", 4),
            n_channels=cfg.get("n_channels", 9),
            n_classes=N_CLASSES,
            feat_dim=cfg.get("feat_dim", 256),
            base_filters=cfg.get("base_filters", 64),
            n_heads=cfg.get("n_heads", 8),
            n_tf_layers=cfg.get("n_tf_layers", 3),
            dropout=0.0,
            encoder_dropout=0.0,
        )
    else:
        from featureflyers_shl.models.mvpf import MVPF
        model = MVPF(
            n_positions=cfg.get("n_positions", 4),
            n_channels=cfg.get("n_channels", 9),
            n_classes=N_CLASSES,
            feat_dim=cfg.get("feat_dim", 256),
            base_filters=cfg.get("base_filters", 64),
            n_heads=cfg.get("n_heads", 4),
            n_tf_layers=cfg.get("n_tf_layers", 2),
            dropout=0.0,
            encoder_dropout=0.0,
        )
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval().to(device)
    print(f"  [MVPF         ] {run_dir.name}  F1={cfg.get('best_f1','?')}")

    positions = cfg.get("positions", ["Bag", "Hand", "Hips", "Torso"])
    if split == "test":
        # Only a single position available at test time — repeat it for all 4 slots
        with h5py.File(hdf5_path, "r") as hf:
            single = hf["test"]["data"][:].astype(np.float32).transpose(0, 2, 1)  # (N, 9, 500)
        raw_multi = np.stack([single] * len(positions), axis=1)  # (N, 4, 9, 500)
        print(f"    test fallback: repeating single position ×{len(positions)}")
    else:
        pos_arrays = []
        for pos in positions:
            ds = SHLWindowDataset(hdf5_path, split=split, position=pos,
                                  sample_limit=None, seed=42, preload=True)
            pos_arrays.append(ds._X.transpose(0, 2, 1).astype(np.float32))  # (N, 9, 500)
        raw_multi = np.stack(pos_arrays, axis=1)  # (N, 4, 9, 500)

    N          = raw_multi.shape[0]
    batch_size = 256
    all_logits = np.zeros((N, N_CLASSES), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x = torch.from_numpy(raw_multi[start:end]).to(device)  # (B, 4, 9, 500)
            x = _norm_perwindow(x)
            bl = np.zeros((end - start, N_CLASSES), dtype=np.float64)
            for i in range(tta_n):
                xi = x + torch.randn_like(x) * jitter_std if (tta_n > 1 and i > 0) else x
                bl += model(xi).float().cpu().numpy()
            all_logits[start:end] = bl / tta_n

    del model, raw_multi
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return _softmax(all_logits.astype(np.float32))


def load_moment_mlp_probs(run_dir: Path, split: str):
    """Stage 19 MOMENT-MLP: load pre-saved val/test probability arrays. No GPU needed."""
    fname = "val_probs.npy" if split == "validation" else "test_probs.npy"
    p = run_dir / fname
    if not p.exists():
        print(f"  [MOMENT-MLP   ] SKIP — missing {p}")
        return None
    probs = np.load(p).astype(np.float32)
    print(f"  [MOMENT-MLP   ] {run_dir.name}  N={probs.shape[0]:,}")
    return probs


def load_precomputed_probs(run_dir: Path, split: str, tag: str):
    """Generic loader for any stage that saves val_probs.npy / test_probs.npy."""
    fname = "val_probs.npy" if split == "validation" else "test_probs.npy"
    p = run_dir / fname
    if not p.exists():
        print(f"  [{tag:<13}] SKIP — missing {p}")
        return None
    probs = np.load(p).astype(np.float32)
    print(f"  [{tag:<13}] {run_dir.name}  N={probs.shape[0]:,}")
    return probs


def load_moment_xgb_probs(run_dir, split):
    """XGB: predict from cached embeddings + stat features. No GPU needed."""
    import joblib
    emb_path  = EMB_DIR  / f"{split}_Bag_moment_normperwindow_mean_pool.npz"
    feat_path = FEAT_DIR / f"{split}_Bag.npz"
    for p in (emb_path, feat_path, run_dir / "model.joblib"):
        if not p.exists():
            print(f"  [MOMENT-XGB   ] SKIP — missing {p.name}")
            return None
    bundle = joblib.load(run_dir / "model.joblib")
    clf    = bundle["sklearn_model"]
    X = np.hstack([np.load(emb_path)["X"], np.load(feat_path)["X"]]).astype(np.float32)
    print(f"  [MOMENT-XGB   ] {run_dir.name}  N={X.shape[0]:,}")
    return clf.predict_proba(X).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device",           default="cuda:0")
    parser.add_argument("--batch-size",       type=int,  default=512)
    parser.add_argument("--tta-n",            type=int,  default=3)
    parser.add_argument("--inception-dir",    type=Path, default=None)
    parser.add_argument("--imuformer-dir",    type=Path, default=None)
    parser.add_argument("--moment-xgb-dir",   type=Path, default=None)
    parser.add_argument("--minirocket-dir",   type=Path, default=None)
    parser.add_argument("--spectrogram-dir",  type=Path, default=None)
    parser.add_argument("--resnet1d-dir",     type=Path, default=None)
    parser.add_argument("--mvpf-dir",         type=Path, default=None)
    parser.add_argument("--moment-mlp-dir",   type=Path, default=None,
                        help="Stage 19 MOMENT-MLP run dir (contains val_probs.npy / test_probs.npy)")
    parser.add_argument("--hmm-dir",          type=Path, default=None,
                        help="Stage 20 HMM output dir (contains val_probs.npy / test_probs.npy)")
    parser.add_argument("--bilstm-dir",       type=Path, default=None,
                        help="Stage 20 BiLSTM output dir (contains val_probs.npy / test_probs.npy)")
    parser.add_argument("--lgbm-n-estimators",type=int,  default=500)
    parser.add_argument("--lgbm-lr",          type=float,default=0.05)
    parser.add_argument("--lgbm-num-leaves",  type=int,  default=63)
    parser.add_argument("--blend-test-frac",  type=float,default=0.2,
                        help="Fraction of val set held out for meta-learner eval")
    parser.add_argument("--predict-test",     action="store_true")
    parser.add_argument("--output",           type=Path,
                        default=OUT_DIR / "submissions" / "FeatureFlyers_blend_s16_lgbm.txt")
    parser.add_argument("--seed",             type=int,  default=42)
    args = parser.parse_args()

    import torch
    import h5py
    import lightgbm as lgb
    from sklearn.metrics import f1_score, classification_report
    from sklearn.model_selection import train_test_split
    from featureflyers_shl.data.dataset import SHLWindowDataset

    device = torch.device(args.device)
    t_start = time.time()

    print(f"\nStage 16 — LightGBM meta-blend")
    print(f"  device  : {device}  TTA={args.tta_n}")

    # ----- 1. Load validation windows -----
    print("\nLoading val windows …", flush=True)
    ds_val  = SHLWindowDataset(HDF5_PATH, split="validation", position="Bag",
                               sample_limit=None, seed=args.seed, preload=True)
    raw_val = ds_val._X.transpose(0, 2, 1).astype(np.float32)  # (N, 9, 500)
    y_val   = ds_val._labels.astype(np.int64)
    N_val   = len(raw_val)
    print(f"  {N_val:,} windows  labels 0–{y_val.max()}\n")

    # ----- 2. Collect val probabilities from each model -----
    print("Collecting val probabilities …")
    val_probs_list = []   # each: (N_val, 8)
    model_names    = []

    if args.inception_dir and args.inception_dir.exists():
        val_probs_list.append(load_inception_probs(args.inception_dir, raw_val, device, args.tta_n))
        model_names.append("InceptionTime")

    if args.imuformer_dir and args.imuformer_dir.exists():
        val_probs_list.append(load_imuformer_probs(args.imuformer_dir, raw_val, device, args.tta_n))
        model_names.append("IMUFormer")

    if args.moment_xgb_dir and args.moment_xgb_dir.exists():
        p = load_moment_xgb_probs(args.moment_xgb_dir, "validation")
        if p is not None:
            val_probs_list.append(p)
            model_names.append("MOMENT-XGB")
    else:
        # Try the default MOMENT-XGB dir
        default_mxgb = OUT_DIR / "foundation_moment_posPool_mean_pool_normperwindow_xgb_hybrid_sfull_strat40000"
        if default_mxgb.exists():
            p = load_moment_xgb_probs(default_mxgb, "validation")
            if p is not None:
                val_probs_list.append(p)
                model_names.append("MOMENT-XGB")

    if args.minirocket_dir and args.minirocket_dir.exists():
        val_probs_list.append(load_minirocket_probs(args.minirocket_dir, raw_val, device, tta_n=1))
        model_names.append("MiniRocket")

    if args.spectrogram_dir and args.spectrogram_dir.exists():
        val_probs_list.append(load_spectrogram_probs(args.spectrogram_dir, raw_val, device, args.tta_n))
        model_names.append("SpectrogramCNN")

    if args.resnet1d_dir and args.resnet1d_dir.exists():
        val_probs_list.append(load_resnet1d_probs(args.resnet1d_dir, raw_val, device, args.tta_n))
        model_names.append("ResNet1D")

    if args.mvpf_dir and args.mvpf_dir.exists():
        val_probs_list.append(load_mvpf_probs(args.mvpf_dir, "validation", HDF5_PATH, device, args.tta_n))
        model_names.append("MVPF")

    if args.moment_mlp_dir and args.moment_mlp_dir.exists():
        p = load_moment_mlp_probs(args.moment_mlp_dir, "validation")
        if p is not None:
            val_probs_list.append(p)
            model_names.append("MOMENT-MLP")

    if args.hmm_dir and args.hmm_dir.exists():
        p = load_precomputed_probs(args.hmm_dir, "validation", "S20-HMM")
        if p is not None:
            val_probs_list.append(p)
            model_names.append("S20-HMM")

    if args.bilstm_dir and args.bilstm_dir.exists():
        p = load_precomputed_probs(args.bilstm_dir, "validation", "S20-BiLSTM")
        if p is not None:
            val_probs_list.append(p)
            model_names.append("S20-BiLSTM")

    if not val_probs_list:
        print("ERROR: no model directories provided or found.")
        sys.exit(1)

    n_models = len(model_names)
    print(f"\n  Models loaded: {model_names}")

    # Print individual val F1s
    print("\nPer-model val Macro-F1 (TTA n={}):".format(args.tta_n))
    for name, probs in zip(model_names, val_probs_list):
        f1 = f1_score(y_val, probs.argmax(1), average="macro", zero_division=0)
        print(f"  {name:<16}  {f1:.4f}")

    # ----- 3. Stack meta-features -----
    X_meta = np.hstack(val_probs_list)   # (N_val, n_models × 8)
    print(f"\nMeta-feature matrix: {X_meta.shape}  ({n_models} models × 8 classes)")

    # Internal 80/20 split for meta-learner evaluation
    idx_tr, idx_te = train_test_split(
        np.arange(N_val), test_size=args.blend_test_frac,
        stratify=y_val, random_state=args.seed
    )
    X_tr, X_te = X_meta[idx_tr], X_meta[idx_te]
    y_tr, y_te = y_val[idx_tr],  y_val[idx_te]

    # ----- 4. Train LightGBM meta-classifier -----
    print(f"\nTraining LightGBM meta-classifier …")
    print(f"  train: {len(X_tr):,}  |  eval: {len(X_te):,}")
    print(f"  n_estimators={args.lgbm_n_estimators}  lr={args.lgbm_lr}  num_leaves={args.lgbm_num_leaves}")

    lgb_params = dict(
        objective="multiclass",
        num_class=N_CLASSES,
        n_estimators=args.lgbm_n_estimators,
        learning_rate=args.lgbm_lr,
        num_leaves=args.lgbm_num_leaves,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=args.seed,
        n_jobs=8,          # cap threads to avoid contention with CUDA context
        verbose=-1,
    )

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)],
    )

    # Evaluate on hold-out
    y_te_pred = clf.predict(X_te)
    f1_meta = f1_score(y_te, y_te_pred, average="macro", zero_division=0)
    acc_meta = float((y_te_pred == y_te).mean())
    print(f"\nMeta-learner eval (hold-out {args.blend_test_frac:.0%} of val):")
    print(f"  Macro-F1={f1_meta:.4f}   Accuracy={acc_meta:.4f}")

    # Also evaluate on full val set (train+test split combined) — informative but biased
    y_val_pred_full = clf.predict(X_meta)
    f1_full = f1_score(y_val, y_val_pred_full, average="macro", zero_division=0)
    print(f"  Full-val Macro-F1={f1_full:.4f}  (biased — blending set overlaps train)")

    print("\n" + classification_report(y_te, y_te_pred,
                                        target_names=list(LABEL_MAP.values()), zero_division=0))

    # Save meta model + val probs
    save_dir = OUT_DIR / f"meta_blend_s16_lgbm_{n_models}models"
    save_dir.mkdir(parents=True, exist_ok=True)
    import joblib as jl
    jl.dump(clf, save_dir / "lgbm_meta.joblib")
    val_probs_ensemble = clf.predict_proba(X_meta).astype(np.float32)
    np.save(save_dir / "val_probs.npy", val_probs_ensemble)
    np.save(save_dir / "val_labels.npy", y_val.astype(np.int32))

    meta_cfg = dict(
        model_names=model_names, n_models=n_models,
        meta_f1_holdout=round(f1_meta, 4), meta_f1_full=round(f1_full, 4),
        lgbm_params=lgb_params, best_iteration=int(clf.best_iteration_),
    )
    (save_dir / "config.json").write_text(json.dumps(meta_cfg, indent=2))
    print(f"Meta-learner saved → {save_dir.relative_to(REPO_ROOT)}/")

    # ----- 5. Optional test submission -----
    if not args.predict_test:
        print("\n[--predict-test not set] Skipping test inference.")
        return

    print("\n" + "=" * 60)
    print("TEST INFERENCE")
    print("=" * 60)
    t_test = time.time()

    print("\nLoading test windows …", flush=True)
    with __import__("h5py").File(HDF5_PATH, "r") as hf:
        raw_test = hf["test"]["data"][:].astype(np.float32).transpose(0, 2, 1)
    N_test = len(raw_test)
    print(f"  {N_test:,} test windows\n")

    test_probs_list = []
    print("Collecting test probabilities …")

    if args.inception_dir and args.inception_dir.exists():
        test_probs_list.append(load_inception_probs(args.inception_dir, raw_test, device, args.tta_n))

    if args.imuformer_dir and args.imuformer_dir.exists():
        test_probs_list.append(load_imuformer_probs(args.imuformer_dir, raw_test, device, args.tta_n))

    # MOMENT-XGB test
    mxgb_dir = args.moment_xgb_dir or (
        OUT_DIR / "foundation_moment_posPool_mean_pool_normperwindow_xgb_hybrid_sfull_strat40000")
    if mxgb_dir.exists() and "MOMENT-XGB" in model_names:
        p = load_moment_xgb_probs(mxgb_dir, "test")
        if p is not None:
            test_probs_list.append(p)

    if args.minirocket_dir and args.minirocket_dir.exists():
        test_probs_list.append(load_minirocket_probs(args.minirocket_dir, raw_test, device, tta_n=1))

    if args.spectrogram_dir and args.spectrogram_dir.exists():
        test_probs_list.append(load_spectrogram_probs(args.spectrogram_dir, raw_test, device, args.tta_n))

    if args.resnet1d_dir and args.resnet1d_dir.exists():
        test_probs_list.append(load_resnet1d_probs(args.resnet1d_dir, raw_test, device, args.tta_n))

    if args.mvpf_dir and args.mvpf_dir.exists() and "MVPF" in model_names:
        test_probs_list.append(load_mvpf_probs(args.mvpf_dir, "test", HDF5_PATH, device, args.tta_n))

    if args.moment_mlp_dir and args.moment_mlp_dir.exists() and "MOMENT-MLP" in model_names:
        p = load_moment_mlp_probs(args.moment_mlp_dir, "test")
        if p is not None:
            test_probs_list.append(p)

    if args.hmm_dir and args.hmm_dir.exists() and "S20-HMM" in model_names:
        p = load_precomputed_probs(args.hmm_dir, "test", "S20-HMM")
        if p is not None:
            test_probs_list.append(p)

    if args.bilstm_dir and args.bilstm_dir.exists() and "S20-BiLSTM" in model_names:
        p = load_precomputed_probs(args.bilstm_dir, "test", "S20-BiLSTM")
        if p is not None:
            test_probs_list.append(p)

    # If some models missing on test (e.g. XGB test embeddings absent), use remaining
    X_test_meta = np.hstack(test_probs_list[:len(test_probs_list)])
    if X_test_meta.shape[1] != X_meta.shape[1]:
        print(f"WARNING: test features {X_test_meta.shape[1]} != val features {X_meta.shape[1]}.")
        print("  Using average ensemble fallback for missing models.")
        avg_probs = np.mean(test_probs_list, axis=0)
        preds_1based = avg_probs.argmax(1).astype(np.int64) + 1
    else:
        test_probs_ensemble = clf.predict_proba(X_test_meta).astype(np.float32)
        np.save(save_dir / "test_probs.npy", test_probs_ensemble)
        preds_0based  = test_probs_ensemble.argmax(1)
        preds_1based  = preds_0based.astype(np.int64) + 1

    from collections import Counter
    dist = {LABEL_MAP[k-1]: int(v) for k, v in sorted(Counter(preds_1based.tolist()).items())}
    print(f"\nPrediction distribution: {dist}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {N_test:,} lines → {args.output} …", flush=True)
    with open(args.output, "w") as f:
        for lbl in preds_1based:
            f.write(",".join([str(int(lbl))] * WIN_SIZE) + "\n")

    # Verify
    lines = args.output.read_text().splitlines()
    ok = (len(lines) == N_test
          and len(lines[0].split(",")) == WIN_SIZE
          and all(1 <= int(v) <= 8 for v in lines[0].split(",")))
    mb = args.output.stat().st_size / 1024 / 1024
    print(f"  [{('PASS' if ok else 'FAIL')}]  {N_test:,} lines × {WIN_SIZE} predictions  {mb:.1f} MB")

    print(f"\nTotal elapsed : {time.time()-t_start:.1f}s")
    print(f"Val F1 (holdout) : {f1_meta:.4f}")
    print(f"Submission       : {args.output}")


if __name__ == "__main__":
    main()
