#!/usr/bin/env python3
"""
Stage 18 — Multi-View Position Fusion v2 (MVPFv2) for SHL 2026.

Architecture improvements over Stage 17 (MVPF v1):
  - 4-stage ResNet1D encoder (vs 3) → smaller bottleneck, larger receptive field
  - Temporal attention pooling (vs global average pool)
  - 3-layer cross-position transformer with 8 heads (vs 2/4)
  - Gated position fusion (learned sigmoid gate per position token)
  - LayerNorm before classification head

Training improvements:
  - IMU rotation augmentation: random 3D rotation applied to acc (ch 0-2) and
    gyro (ch 3-5) channels. Teaches orientation-invariant locomotion features —
    the most impactful domain-specific augmentation for IMU/HAR tasks.
  - Magnitude warping: smooth random time-varying amplitude scaling via linear
    interpolation of random knots. Simulates intensity variation in movement.
  - Position dropout (p=0.25) + jitter (σ=0.05) — carried over from v1.
  - LR warmup (epochs 1-5: 5e-5 → 2e-4) eliminates early oscillation.
  - ReduceLROnPlateau (patience=7, factor=0.5) after warmup.
  - Stochastic Weight Averaging (SWA) over last 20 epochs for smoother,
    better-calibrated model weights.

Usage
-----
# Full run (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_stage18_mvpf_v2.py \\
    --loss focal --focal-gamma 2.0 --sampler balanced \\
    --seed 42 --device cuda:0 \\
    > outputs/execution-output/logs/stage18_gpu0_mvpf_v2.log 2>&1 &

# Quick smoke test
python scripts/train_stage18_mvpf_v2.py --sample-limit 5000 --epochs 3 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

REPO_ROOT    = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HDF5_PATH    = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
DEFAULT_OUTD = REPO_ROOT / "outputs" / "execution-output"
POSITIONS    = ["Bag", "Hand", "Hips", "Torso"]
N_CLASSES    = 8
LABEL_MAP    = {0: "Still", 1: "Walking", 2: "Run", 3: "Bike",
                4: "Car",   5: "Bus",     6: "Train",  7: "Metro"}

# IMU channel layout (SHL 2026):  0-2 = acc(x,y,z), 3-5 = gyro(x,y,z), 6-8 = other
_ACC_SLICE  = slice(0, 3)
_GYRO_SLICE = slice(3, 6)


# ---------------------------------------------------------------------------
# Multi-position dataset (identical to Stage 17)
# ---------------------------------------------------------------------------

class MultiPositionWindowDataset(Dataset):
    def __init__(self, hdf5_path, split, positions=POSITIONS,
                 sample_limit=None, seed=42, preload=True):
        from featureflyers_shl.data.dataset import SHLWindowDataset
        self._datasets = []
        for pos in positions:
            ds = SHLWindowDataset(hdf5_path, split=split, position=pos,
                                  sample_limit=sample_limit, seed=seed, preload=preload)
            self._datasets.append(ds)
        lengths = [len(ds) for ds in self._datasets]
        assert len(set(lengths)) == 1, f"Mismatched position lengths: {dict(zip(positions, lengths))}"
        self._n = lengths[0]
        ref_lbl = self._datasets[0]._labels
        for ds, pos in zip(self._datasets[1:], positions[1:]):
            n_mm = (ds._labels != ref_lbl).sum()
            if n_mm > 0:
                print(f"  WARNING: {n_mm} label mismatches for {pos}")
        self._labels = ref_lbl

    def __len__(self): return self._n

    def __getitem__(self, idx):
        xs = [torch.from_numpy(ds._X[idx].astype(np.float32)).T for ds in self._datasets]
        return torch.stack(xs, dim=0), int(self._labels[idx])  # (4, 9, 500), label


# ---------------------------------------------------------------------------
# Normalization and augmentation
# ---------------------------------------------------------------------------

def _norm_perwindow(x: torch.Tensor) -> torch.Tensor:
    """Per-window, per-channel z-score.  x: (B, P, C, T)"""
    x   = x.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    mu  = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (x - mu) / std


def _augment_v2(
    x: torch.Tensor,
    rot_p: float        = 0.7,
    mag_warp_p: float   = 0.5,
    mag_sigma: float    = 0.15,
    jitter_sigma: float = 0.05,
    pos_drop_p: float   = 0.25,
) -> torch.Tensor:
    """
    Train-time augmentation pipeline for MVPF v2.  x: (B, P, C, T), C=9.

    1. IMU rotation  — same random 3-D rotation applied to acc (ch 0-2) and
       gyro (ch 3-5) of every position in the same sample.  Simulates arbitrary
       sensor orientation and teaches orientation-invariant locomotion features.

    2. Magnitude warping — smooth random time-varying amplitude scaling via
       linear interpolation of 4 random knots ~ N(1, mag_sigma).  Simulates
       variation in movement intensity within the 5-second window.

    3. Jitter — additive Gaussian noise in normalized space.  Prevents fitting
       to exact per-window amplitude patterns.

    4. Position dropout — zero out one random position per sample with
       probability pos_drop_p.  Forces the cross-position transformer to learn
       redundant, position-agnostic locomotion features.
    """
    B, P, C, T = x.shape

    # ---- 1. IMU rotation ------------------------------------------------
    if rot_p > 0:
        rand_mat = torch.randn(B, 3, 3, device=x.device)
        Q, _     = torch.linalg.qr(rand_mat)               # random orthogonal (B, 3, 3)
        # Ensure proper rotation (det = +1): if det=-1, negate Q  (det(-Q)=-det(Q)=+1)
        dets = torch.linalg.det(Q)                          # (B,)
        Q    = Q * dets.view(B, 1, 1)                       # -Q when det=-1
        rot_mask = torch.rand(B, device=x.device) < rot_p  # (B,)
        I    = torch.eye(3, device=x.device).unsqueeze(0).expand(B, -1, -1)
        Q_eff = torch.where(rot_mask.view(B, 1, 1), Q, I)  # (B, 3, 3)
        # Apply Q to acc and gyro channel triplets
        x = x.clone()
        for sl in (_ACC_SLICE, _GYRO_SLICE):
            ch   = x[:, :, sl, :]                           # (B, P, 3, T)
            ch_t = ch.permute(0, 1, 3, 2)                   # (B, P, T, 3)
            # v_rot = v @ Q^T  (row-vector convention)
            ch_r = torch.matmul(ch_t, Q_eff.unsqueeze(1).transpose(-1, -2))
            x[:, :, sl, :] = ch_r.permute(0, 1, 3, 2)

    # ---- 2. Magnitude warping ------------------------------------------
    if mag_warp_p > 0:
        warp_mask = torch.rand(B, device=x.device) < mag_warp_p
        if warp_mask.any():
            n_knots = 4
            knots   = 1.0 + torch.randn(B, 1, n_knots, device=x.device) * mag_sigma
            warp    = F.interpolate(knots, size=T, mode="linear", align_corners=True)  # (B, 1, T)
            warp    = warp.view(B, 1, 1, T)
            ones    = torch.ones_like(warp)
            warp_eff = torch.where(warp_mask.view(B, 1, 1, 1), warp, ones)
            x = x * warp_eff

    # ---- 3. Jitter -------------------------------------------------------
    x = x + torch.randn_like(x) * jitter_sigma

    # ---- 4. Position dropout --------------------------------------------
    if pos_drop_p > 0:
        drop_mask = torch.rand(B, device=x.device) < pos_drop_p
        drop_pos  = torch.randint(0, P, (B,), device=x.device)
        pos_idx   = torch.arange(P, device=x.device).unsqueeze(0)
        kill_mask = (pos_idx == drop_pos.unsqueeze(1)) & drop_mask.unsqueeze(1)  # (B, P)
        x = x * (~kill_mask).float().unsqueeze(-1).unsqueeze(-1)

    return x


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = total_correct = total_n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x    = _norm_perwindow(x)
        x    = _augment_v2(x)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        l = loss.item()
        if l == l:
            total_loss += l * len(y)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n       += len(y)
    return total_loss / total_n, total_correct / total_n


def eval_epoch(model, loader, device):
    from sklearn.metrics import f1_score
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = _norm_perwindow(x.to(device))
            all_preds.append(model(x).argmax(1).cpu().numpy())
            all_labels.append(y.numpy())
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = float((preds == labels).mean())
    f1  = float(f1_score(labels, preds, average="macro", zero_division=0))
    return acc, f1, preds, labels


# ---------------------------------------------------------------------------
# LR warmup helper
# ---------------------------------------------------------------------------

def _warmup_factor(epoch: int, warmup_epochs: int) -> float:
    """Linear warmup: epoch 1 → small factor, epoch warmup_epochs → 1.0."""
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return 1.0
    return epoch / warmup_epochs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hdf5-path",       type=Path,  default=HDF5_PATH)
    parser.add_argument("--sample-limit",    type=int,   default=None)
    parser.add_argument("--epochs",          type=int,   default=80)
    parser.add_argument("--batch-size",      type=int,   default=256)
    parser.add_argument("--lr",              type=float, default=2e-4)
    parser.add_argument("--warmup-epochs",   type=int,   default=5)
    parser.add_argument("--feat-dim",        type=int,   default=256)
    parser.add_argument("--base-filters",    type=int,   default=64)
    parser.add_argument("--n-heads",         type=int,   default=8)
    parser.add_argument("--n-tf-layers",     type=int,   default=3)
    parser.add_argument("--dropout",         type=float, default=0.4)
    parser.add_argument("--encoder-dropout", type=float, default=0.2)
    parser.add_argument("--patience",        type=int,   default=15)
    parser.add_argument("--swa-start",       type=int,   default=50,
                        help="Epoch from which to start SWA weight averaging (0=disabled)")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--device",          default="cuda:0")
    parser.add_argument("--output-dir",      type=Path,  default=DEFAULT_OUTD)
    parser.add_argument("--loss",            default="focal", choices=["ce", "focal"])
    parser.add_argument("--focal-gamma",     type=float, default=2.0)
    parser.add_argument("--class-weights",   default="none", choices=["none", "balanced"])
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--sampler",         default="balanced", choices=["random", "balanced"])
    args = parser.parse_args()

    from sklearn.metrics import classification_report
    from featureflyers_shl.models.mvpf_v2      import MVPFv2
    from featureflyers_shl.training.losses      import build_criterion

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device  = torch.device(args.device)
    use_amp = device.type == "cuda"
    lim_str = str(args.sample_limit) if args.sample_limit else "full"

    loss_tag    = f"focal_g{args.focal_gamma}" if args.loss == "focal" else "ce"
    sampler_tag = "_balsampler" if args.sampler == "balanced" else ""
    swa_tag     = f"_swa{args.swa_start}" if args.swa_start > 0 else ""
    run_name    = (f"mvpf_v2_4pos_fd{args.feat_dim}_bf{args.base_filters}"
                   f"_h{args.n_heads}tf{args.n_tf_layers}"
                   f"_s{lim_str}_{loss_tag}{sampler_tag}{swa_tag}_rotaug"
                   f"_ep{args.epochs}_bs{args.batch_size}")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStage 18 — MVPF v2 (rotation aug + magnitude warp + SWA)")
    print(f"  device      : {device}  (AMP={'on' if use_amp else 'off'})")
    print(f"  feat_dim    : {args.feat_dim}  base_filters={args.base_filters}")
    print(f"  transformer : {args.n_tf_layers} layers  {args.n_heads} heads")
    print(f"  epochs      : {args.epochs}  warmup={args.warmup_epochs}  patience={args.patience}")
    print(f"  SWA start   : ep{args.swa_start}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  loss        : {args.loss}" + (f"  gamma={args.focal_gamma}" if args.loss == "focal" else ""))
    print(f"  sampler     : {args.sampler}")
    print(f"  output      : {run_dir.relative_to(REPO_ROOT)}\n")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print("Preloading train windows (all 4 positions) …")
    t0 = time.time()
    ds_train = MultiPositionWindowDataset(
        args.hdf5_path, split="train", positions=POSITIONS,
        sample_limit=args.sample_limit, seed=args.seed, preload=True,
    )
    for ds, pos in zip(ds_train._datasets, POSITIONS):
        print(f"  {pos}: {len(ds):,} windows")

    print("Preloading val windows …")
    ds_val = MultiPositionWindowDataset(
        args.hdf5_path, split="validation", positions=POSITIONS,
        sample_limit=None, seed=args.seed + 1, preload=True,
    )
    print(f"  val: {len(ds_val):,} windows  ({time.time()-t0:.1f}s)\n")

    counts_np = np.bincount(ds_train._labels, minlength=N_CLASSES).astype(np.float64)
    counts_t  = torch.from_numpy(counts_np).float()

    print("Training class distribution:")
    total_n = int(counts_np.sum())
    for i, (name, cnt) in enumerate(zip(LABEL_MAP.values(), counts_np)):
        pct = 100.0 * cnt / total_n
        print(f"  [{i}] {name:<10} : {int(cnt):>9,}  ({pct:5.1f}%)")
    print()

    criterion = build_criterion(
        loss_type=args.loss, class_weights=args.class_weights,
        counts=counts_t, n_classes=N_CLASSES,
        focal_gamma=args.focal_gamma, label_smoothing=args.label_smoothing,
        device=device,
    )

    loader_kw = dict(batch_size=args.batch_size, num_workers=0,
                     pin_memory=(device.type == "cuda"))

    if args.sampler == "balanced":
        inv_freq  = 1.0 / np.where(counts_np > 0, counts_np, 1.0)
        sample_w  = torch.from_numpy(inv_freq[ds_train._labels].astype(np.float32))
        sampler   = WeightedRandomSampler(sample_w, num_samples=len(ds_train), replacement=True)
        train_loader = DataLoader(ds_train, sampler=sampler, **loader_kw)
    else:
        train_loader = DataLoader(ds_train, shuffle=True, **loader_kw)

    val_loader = DataLoader(ds_val, shuffle=False, **loader_kw)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = MVPFv2(
        n_positions=len(POSITIONS), n_channels=9, n_classes=N_CLASSES,
        feat_dim=args.feat_dim, base_filters=args.base_filters,
        n_heads=args.n_heads, n_tf_layers=args.n_tf_layers,
        dropout=args.dropout, encoder_dropout=args.encoder_dropout,
    ).to(device)

    print(f"Model   : MVPFv2  n_params={model.n_params:,}")
    print(f"  PositionEncoderV2    : 4-stage ResNet1D (9→{args.base_filters}→...→{args.base_filters*4}) + AttnPool")
    print(f"  CrossPosFusionV2     : {args.n_tf_layers}×TransformerBlock(d={args.feat_dim}, h={args.n_heads}) + gate")
    print(f"  Head                 : LN → Dropout({args.dropout}) → Linear({args.feat_dim}, {N_CLASSES})\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * (1.0 / args.warmup_epochs),
                                  weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7,
        threshold=1e-4, min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler() if use_amp else None

    # SWA setup
    use_swa = args.swa_start > 0 and args.swa_start < args.epochs
    if use_swa:
        from torch.optim.swa_utils import AveragedModel, update_bn
        swa_model = AveragedModel(model)
        print(f"SWA     : enabled from epoch {args.swa_start}")

    print(f"{'Ep':>4}  {'TrainLoss':>9}  {'TrainAcc':>8}  {'ValAcc':>7}  "
          f"{'MacroF1':>8}  {'LR':>8}  {'SWA':>4}  {'Time':>6}")
    print("-" * 74)

    best_f1    = 0.0
    best_state = None
    no_improve = 0
    history    = []
    t_train    = time.time()

    for epoch in range(1, args.epochs + 1):
        # LR warmup: linearly ramp from lr/warmup_epochs to lr
        if epoch <= args.warmup_epochs and args.warmup_epochs > 0:
            lr_scale = epoch / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * lr_scale

        t_ep = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)

        # SWA: accumulate weights and evaluate SWA model
        swa_active = use_swa and epoch >= args.swa_start
        if swa_active:
            swa_model.update_parameters(model)
        eval_model = swa_model if swa_active else model
        val_acc, val_f1, val_preds, val_labels = eval_epoch(eval_model, val_loader, device)

        # Only step ReduceLROnPlateau before SWA phase; keep LR stable during SWA
        if not swa_active and epoch > args.warmup_epochs:
            scheduler.step(val_f1)
        lr_now   = optimizer.param_groups[0]["lr"]
        ep_time  = time.time() - t_ep

        print(f"{epoch:>4}  {tr_loss:>9.4f}  {tr_acc:>7.3%}  {val_acc:>7.3%}  "
              f"{val_f1:>8.4f}  {lr_now:>8.2e}  {'SWA' if swa_active else '   ':>4}  {ep_time:>5.1f}s",
              flush=True)

        history.append(dict(epoch=epoch, train_loss=round(tr_loss, 5),
                            train_acc=round(tr_acc, 4), val_acc=round(val_acc, 4),
                            val_macro_f1=round(val_f1, 4), lr=round(lr_now, 8),
                            swa_active=swa_active, epoch_time_s=round(ep_time, 1)))

        if val_f1 >= best_f1:
            best_f1    = val_f1
            eval_m     = eval_model.module if hasattr(eval_model, "module") else eval_model
            best_state = {k: v.cpu().clone() for k, v in eval_m.state_dict().items()}
            best_preds, best_labels = val_preds, val_labels
            no_improve = 0
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break

    # Update BN stats for SWA model (re-run one pass over training data)
    if use_swa and len(history) >= args.swa_start:
        print("\nUpdating SWA BatchNorm statistics …")
        update_bn(train_loader, swa_model, device=device)
        print("Evaluating SWA model (final) …")
        swa_acc, swa_f1, swa_preds, swa_labels = eval_epoch(swa_model, val_loader, device)
        print(f"  SWA final val Macro-F1 = {swa_f1:.4f}  (prev best = {best_f1:.4f})")
        if swa_f1 >= best_f1:
            best_f1    = swa_f1
            swa_m      = swa_model.module if hasattr(swa_model, "module") else swa_model
            best_state = {k: v.cpu().clone() for k, v in swa_m.state_dict().items()}
            best_preds, best_labels = swa_preds, swa_labels
            print("  → SWA model is best; saving SWA weights.")

    total_time = time.time() - t_train
    print(f"\nTraining done in {total_time:.1f}s   best val macro-F1 = {best_f1:.4f}")

    # -----------------------------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------------------------
    base_m = model.module if hasattr(model, "module") else model
    base_m.load_state_dict(best_state)
    base_m.eval()

    torch.save(best_state, run_dir / "model.pt")

    cfg = dict(
        model="MVPFv2", n_positions=len(POSITIONS), n_channels=9,
        feat_dim=args.feat_dim, base_filters=args.base_filters,
        n_heads=args.n_heads, n_tf_layers=args.n_tf_layers,
        dropout=args.dropout, encoder_dropout=args.encoder_dropout,
        n_params=model.n_params, positions=POSITIONS,
        loss=args.loss, focal_gamma=args.focal_gamma,
        class_weights=args.class_weights, sampler=args.sampler,
        swa_start=args.swa_start, best_f1=round(best_f1, 4),
        epochs_trained=len(history), total_train_time_s=round(total_time, 1),
        seed=args.seed,
    )
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(dict(**cfg, history=history), indent=2))

    report = classification_report(best_labels, best_preds,
                                   target_names=list(LABEL_MAP.values()), zero_division=0)
    (run_dir / "classification_report.txt").write_text(report)

    print(f"\nArtifacts saved to {run_dir.relative_to(REPO_ROOT)}/")
    print(f"  Macro-F1 : {best_f1:.4f}")
    print(report)


if __name__ == "__main__":
    main()
