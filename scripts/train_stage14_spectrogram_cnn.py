#!/usr/bin/env python3
"""
Stage 14 — Spectrogram CNN on all 4 positions (pool fusion) for SHL 2026.

The key insight: Bus (~20 Hz engine rumble), Train (rail-periodicity ~50–100 Hz)
and Metro (distinct tunnel resonance) look nearly identical in the raw time domain
but are clearly separable in frequency space.

Pipeline
--------
  Raw window (9, 500) → per-channel STFT → log-magnitude (9, 33, 32)
  → Modified ResNet-18 [first conv: 9 ch in] → Linear(512, 8) → logits

STFT (torch.stft, center=True):
  n_fft=64, hop_length=16, win_length=64
  → n_freq = 33, n_frames = ceil(500/16) = 32

Uses focal γ=2 + balanced sampler (winning config from Stage 12).

Usage
-----
# Smoke test
python scripts/train_stage14_spectrogram_cnn.py --sample-limit 5000 --epochs 3 --device cuda:0

# Full run (GPU 1)
CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/train_stage14_spectrogram_cnn.py \\
    --loss focal --focal-gamma 2.0 --sampler balanced --class-weights none \\
    --seed 42 --device cuda:0 \\
    > outputs/execution-output/logs/stage14_gpu1_spectrogram_cnn.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT    = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HDF5_PATH    = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
DEFAULT_OUTD = REPO_ROOT / "outputs" / "execution-output"
POSITIONS    = ["Bag", "Hand", "Hips", "Torso"]
N_CLASSES    = 8
LABEL_MAP    = {0:"Still",1:"Walking",2:"Run",3:"Bike",4:"Car",5:"Bus",6:"Train",7:"Metro"}


def _norm_perwindow(x):
    x = x.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    mu  = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (x - mu) / std


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    import torch
    import torch.nn as nn
    model.train()
    total_loss = total_correct = total_n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = _norm_perwindow(x)
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
    import torch
    from sklearn.metrics import f1_score
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = _norm_perwindow(x.to(device))
            preds = model(x).argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = float((preds == labels).mean())
    f1  = float(f1_score(labels, preds, average="macro", zero_division=0))
    return acc, f1, preds, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hdf5-path",       type=Path,  default=HDF5_PATH)
    parser.add_argument("--sample-limit",    type=int,   default=None)
    parser.add_argument("--epochs",          type=int,   default=80)
    parser.add_argument("--batch-size",      type=int,   default=256,
                        help="Smaller default: spectrograms are larger tensors than raw windows")
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--dropout",         type=float, default=0.3)
    parser.add_argument("--n-fft",           type=int,   default=64)
    parser.add_argument("--hop-length",      type=int,   default=16)
    parser.add_argument("--patience",        type=int,   default=15)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--device",          default="cuda:0")
    parser.add_argument("--output-dir",      type=Path,  default=DEFAULT_OUTD)
    parser.add_argument("--loss",            default="focal", choices=["ce", "focal"])
    parser.add_argument("--focal-gamma",     type=float, default=2.0)
    parser.add_argument("--class-weights",   default="none", choices=["none", "balanced"])
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--sampler",         default="balanced", choices=["random", "balanced"])
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
    from sklearn.metrics import classification_report

    from featureflyers_shl.data.dataset           import SHLWindowDataset
    from featureflyers_shl.models.spectrogram_cnn import SpectrogramCNN
    from featureflyers_shl.training.losses        import build_criterion

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device  = torch.device(args.device)
    use_amp = device.type == "cuda"
    lim_str = str(args.sample_limit) if args.sample_limit else "full"

    loss_tag    = f"focal_g{args.focal_gamma}" if args.loss == "focal" else "ce"
    sampler_tag = "_balsampler" if args.sampler == "balanced" else ""
    run_name = (f"spectrogramcnn_posPool_nfft{args.n_fft}_hop{args.hop_length}"
                f"_s{lim_str}_{loss_tag}{sampler_tag}"
                f"_ep{args.epochs}_bs{args.batch_size}")
    run_dir  = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStage 14 — Spectrogram CNN pool fusion")
    print(f"  device     : {device}  (AMP={'on' if use_amp else 'off'})")
    print(f"  STFT       : n_fft={args.n_fft}, hop={args.hop_length} → (9, {args.n_fft//2+1}, 32) spectrograms")
    print(f"  epochs     : {args.epochs}  patience={args.patience}")
    print(f"  loss       : {args.loss}" + (f"  gamma={args.focal_gamma}" if args.loss == "focal" else ""))
    print(f"  sampler    : {args.sampler}")
    print(f"  output     : {run_dir.relative_to(REPO_ROOT)}\n")

    print("Preloading train windows into RAM …")
    t0 = time.time()
    train_datasets = [
        SHLWindowDataset(args.hdf5_path, split="train", position=pos,
                         sample_limit=args.sample_limit, seed=args.seed, preload=True)
        for pos in POSITIONS
    ]
    for ds, pos in zip(train_datasets, POSITIONS):
        print(f"  {pos}: {len(ds):,} windows", flush=True)

    print("Preloading val windows into RAM …")
    ds_val   = SHLWindowDataset(args.hdf5_path, split="validation", position="Bag",
                                sample_limit=None, seed=args.seed + 1, preload=True)
    ds_train = ConcatDataset(train_datasets)
    print(f"  Bag val: {len(ds_val):,} windows  ({time.time()-t0:.1f}s)\n")

    all_labels = np.concatenate([ds._labels for ds in train_datasets])
    counts_np  = np.bincount(all_labels, minlength=N_CLASSES).astype(np.float64)
    counts_t   = torch.from_numpy(counts_np).float()

    print("Training class distribution:")
    total_n = int(counts_np.sum())
    for i, (name, cnt) in enumerate(zip(LABEL_MAP.values(), counts_np)):
        pct = 100.0 * cnt / total_n
        print(f"  [{i}] {name:<10} : {int(cnt):>9,}  ({pct:5.1f}%)  {'#' * max(1,int(pct/2))}")
    print()

    criterion = build_criterion(
        loss_type=args.loss, class_weights=args.class_weights,
        counts=counts_t, n_classes=N_CLASSES,
        focal_gamma=args.focal_gamma, label_smoothing=args.label_smoothing,
        device=device,
    )
    print(f"Criterion : {criterion}\n")

    loader_kw = dict(batch_size=args.batch_size, num_workers=0,
                     pin_memory=(device.type == "cuda"))

    if args.sampler == "balanced":
        inv_freq = 1.0 / np.where(counts_np > 0, counts_np, 1.0)
        sample_w = torch.from_numpy(inv_freq[all_labels].astype(np.float32))
        sampler  = WeightedRandomSampler(sample_w, num_samples=len(ds_train), replacement=True)
        train_loader = DataLoader(ds_train, sampler=sampler, **loader_kw)
        print("Sampler   : WeightedRandomSampler (balanced)")
    else:
        train_loader = DataLoader(ds_train, shuffle=True, **loader_kw)
        print("Sampler   : random shuffle")

    val_loader = DataLoader(ds_val, shuffle=False, **loader_kw)
    print()

    model = SpectrogramCNN(
        n_channels=9, n_classes=N_CLASSES,
        n_fft=args.n_fft, hop_length=args.hop_length,
        win_length=args.n_fft, n_frames=32,
        dropout=args.dropout,
    ).to(device)
    print(f"Model   : SpectrogramCNN (ResNet-18 backbone)  params={model.n_params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler() if use_amp else None

    print(f"{'Ep':>4}  {'TrainLoss':>9}  {'TrainAcc':>8}  {'ValAcc':>7}  "
          f"{'MacroF1':>8}  {'LR':>8}  {'Time':>6}")
    print("-" * 68)

    best_f1    = 0.0
    best_state = None
    no_improve = 0
    history    = []
    t_train    = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_acc, val_f1, val_preds, val_labels = eval_epoch(model, val_loader, device)
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()
        ep_time = time.time() - t_ep

        print(f"{epoch:>4}  {tr_loss:>9.4f}  {tr_acc:>7.3%}  {val_acc:>7.3%}  "
              f"{val_f1:>8.4f}  {lr_now:>8.2e}  {ep_time:>5.1f}s", flush=True)

        history.append(dict(epoch=epoch, train_loss=round(tr_loss, 5),
                            train_acc=round(tr_acc, 4), val_acc=round(val_acc, 4),
                            val_macro_f1=round(val_f1, 4), lr=round(lr_now, 8),
                            epoch_time_s=round(ep_time, 1)))

        if val_f1 >= best_f1:
            best_f1    = val_f1
            _m = model.module if hasattr(model, "module") else model
            best_state = {k: v.cpu().clone() for k, v in _m.state_dict().items()}
            best_preds, best_labels = val_preds, val_labels
            no_improve = 0
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break

    total_time = time.time() - t_train
    print(f"\nTraining done in {total_time:.1f}s   best val macro-F1 = {best_f1:.4f}")

    # --- Save artifacts ---
    _m = model.module if hasattr(model, "module") else model
    _m.load_state_dict(best_state)
    _m.eval()

    torch.save(best_state, run_dir / "model.pt")

    # Save config compatible with ensemble loading (load_inception pattern)
    cfg = dict(
        model="SpectrogramCNN", n_fft=args.n_fft, hop_length=args.hop_length,
        win_length=args.n_fft, n_frames=32, dropout=args.dropout, n_params=model.n_params,
        fusion="pool", positions=POSITIONS, loss=args.loss,
        focal_gamma=args.focal_gamma, class_weights=args.class_weights,
        sampler=args.sampler, best_f1=round(best_f1, 4),
        epochs_trained=len(history), total_train_time_s=round(total_time, 1), seed=args.seed,
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
