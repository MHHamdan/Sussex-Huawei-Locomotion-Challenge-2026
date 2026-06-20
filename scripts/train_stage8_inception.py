#!/usr/bin/env python3
"""
Stage 8 — InceptionTime on all 4 positions (pool fusion) for SHL 2026.

Pool fusion: Bag + Hand + Hips + Torso stacked as independent training samples.
Validation: Bag-only (57 576 windows) for fair comparison with prior stages.
Test: single mixed-position array — compatible with pool models.

Usage
-----
# Smoke test
python scripts/train_stage8_inception.py --sample-limit 5000 --epochs 3 --device cuda:0

# Full run (Stage 8)
python scripts/train_stage8_inception.py --epochs 100 --patience 15 --device cuda:0 \\
    2>&1 | tee outputs/execution-output/stage8_inception_pool_full.log
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


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def _norm_perwindow(x: "torch.Tensor") -> "torch.Tensor":
    """Z-score each window per channel. Replaces NaN/Inf before normalizing."""
    x = x.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    mu  = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (x - mu) / std


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    import torch
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        l = loss.item()
        if not (l != l):  # skip NaN batches in loss accumulation
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hdf5-path",    type=Path, default=HDF5_PATH)
    parser.add_argument("--sample-limit", type=int,   default=None,
                        help="Stratified window limit per position per split (None=full)")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=512)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--nb-filters",   type=int,   default=32,
                        help="Filters per branch in each inception module")
    parser.add_argument("--depth",        type=int,   default=6,
                        help="Number of inception modules (must be multiple of 3)")
    parser.add_argument("--bottleneck",   type=int,   default=32)
    parser.add_argument("--patience",     type=int,   default=15,
                        help="Early stopping patience (0=disabled)")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--num-workers",  type=int,   default=4)
    parser.add_argument("--output-dir",   type=Path,  default=DEFAULT_OUTD)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, ConcatDataset
    from sklearn.metrics import classification_report, confusion_matrix

    from featureflyers_shl.data.dataset    import SHLWindowDataset
    from featureflyers_shl.models.inception import InceptionTime

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device  = torch.device(args.device)
    use_amp = device.type == "cuda"
    lim_str = str(args.sample_limit) if args.sample_limit else "full"

    run_name = (f"inception_posPool_nb{args.nb_filters}_d{args.depth}"
                f"_s{lim_str}_ep{args.epochs}_bs{args.batch_size}")
    run_dir  = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStage 8 — InceptionTime pool fusion")
    print(f"  device     : {device}  (AMP={'on' if use_amp else 'off'})")
    print(f"  positions  : {POSITIONS} (pool — stacked as independent samples)")
    print(f"  nb_filters : {args.nb_filters}  depth={args.depth}  bottleneck={args.bottleneck}")
    print(f"  epochs     : {args.epochs}  patience={args.patience}")
    print(f"  output     : {run_dir.relative_to(REPO_ROOT)}\n")

    # ---- preload all windows into RAM via SHLWindowDataset(preload=True) ----
    # Default (preload=None, sample_limit=None) does lazy HDF5 reads — slow for
    # random DataLoader access. preload=True does one sequential HDF5 read per
    # position at init and keeps everything in RAM.
    # 4 positions × ~392K × 9 × 500 × float32 ≈ 28 GB — fits in 64 GB RAM.
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
    ds_val  = SHLWindowDataset(args.hdf5_path, split="validation", position="Bag",
                               sample_limit=None, seed=args.seed + 1, preload=True)
    ds_train = ConcatDataset(train_datasets)
    n_train  = len(ds_train)
    print(f"  Bag val: {len(ds_val):,} windows  ({time.time()-t0:.1f}s total)\n")

    # Class weights from training labels
    counts = sum(ds.class_counts().astype(np.float64) for ds in train_datasets)
    weights = torch.tensor(
        (counts.sum() / (N_CLASSES * np.where(counts > 0, counts, 1.0))).astype(np.float32),
        device=device,
    )

    # num_workers=0: TensorDataset is already in RAM, workers only add IPC overhead
    loader_kw = dict(batch_size=args.batch_size, num_workers=0,
                     pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(ds_train, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(ds_val,   shuffle=False, **loader_kw)

    # ---- model ----
    model = InceptionTime(
        n_channels=9, n_classes=N_CLASSES,
        nb_filters=args.nb_filters, depth=args.depth,
        bottleneck=args.bottleneck, dropout=args.dropout,
    ).to(device)
    n_params = InceptionTime.n_params(model)
    print(f"Model   : InceptionTime  params={n_params:,}\n")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler() if use_amp else None

    print(f"{'Ep':>4}  {'TrainLoss':>9}  {'TrainAcc':>8}  "
          f"{'ValAcc':>7}  {'MacroF1':>8}  {'LR':>8}  {'Time':>6}")
    print("-" * 68)

    best_f1    = 0.0
    best_state = None
    no_improve = 0
    history    = []
    t_train    = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion,
                                      device, scaler)
        val_acc, val_f1, _, _ = eval_epoch(model, val_loader, device)
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()
        ep_time = time.time() - t_ep

        print(f"{epoch:>4}  {tr_loss:>9.4f}  {tr_acc:>7.3%}  "
              f"{val_acc:>7.3%}  {val_f1:>8.4f}  {lr_now:>8.2e}  {ep_time:>5.1f}s",
              flush=True)

        history.append(dict(epoch=epoch, train_loss=round(tr_loss,5),
                            train_acc=round(tr_acc,4), val_acc=round(val_acc,4),
                            val_macro_f1=round(val_f1,4), lr=round(lr_now,8),
                            epoch_time_s=round(ep_time,1)))

        if val_f1 >= best_f1:
            best_f1    = val_f1
            _m = model.module if hasattr(model, "module") else model
            best_state = {k: v.cpu().clone() for k, v in _m.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break

    total_time = time.time() - t_train
    print(f"\nTraining done in {total_time:.1f}s   best val macro-F1 = {best_f1:.4f}")

    # ---- final eval ----
    _m = model.module if hasattr(model, "module") else model
    _m.load_state_dict(best_state)
    final_acc, final_f1, final_preds, final_labels = eval_epoch(model, val_loader, device)
    report = classification_report(
        final_labels, final_preds,
        labels=list(range(N_CLASSES)),
        target_names=[LABEL_MAP[i] for i in range(N_CLASSES)],
        zero_division=0,
    )
    print(f"\n{'='*60}")
    print(f"Best model — Accuracy: {final_acc:.4f}   Macro-F1: {final_f1:.4f}")
    print(f"{'='*60}")
    print(report)

    cm = confusion_matrix(final_labels, final_preds, labels=list(range(N_CLASSES)))
    header   = "Pred ->  " + "  ".join(f"{LABEL_MAP[i]:>8}" for i in range(N_CLASSES))
    cm_lines = [header]
    for i, row in enumerate(cm):
        cm_lines.append(f"{LABEL_MAP[i]:>8}  " + "  ".join(f"{v:>8}" for v in row))
    cm_str = "\n".join(cm_lines)
    print("\nConfusion matrix:\n" + cm_str)

    # ---- save ----
    torch.save(best_state, run_dir / "model.pt")
    (run_dir / "classification_report.txt").write_text(report)
    (run_dir / "confusion_matrix.txt").write_text(cm_str)

    config = dict(
        model="InceptionTime", positions=POSITIONS, fusion="pool",
        nb_filters=args.nb_filters, depth=args.depth, bottleneck=args.bottleneck,
        n_params=n_params, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, dropout=args.dropout, patience=args.patience,
        sample_limit=args.sample_limit, seed=args.seed, device=str(device),
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    metrics = dict(
        best_val_macro_f1=round(final_f1, 4),
        best_val_accuracy=round(final_acc, 4),
        total_time_s=round(total_time, 1),
        epochs_trained=len(history),
        history=history,
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved to {run_dir.relative_to(REPO_ROOT)}/")
    print(f"  Macro-F1 : {final_f1:.4f}")
    print(f"  Accuracy : {final_acc:.4f}")


if __name__ == "__main__":
    main()
