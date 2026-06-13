#!/usr/bin/env python3
"""
Train a 1D CNN deep baseline on raw SHL 2026 sensor windows.

Usage
-----
# Smoke test (fast: 5000 windows, 2 epochs, auto device)
python scripts/train_deep.py --position Bag --sample-limit 5000 --epochs 2 --batch-size 128 --device auto

# Full Bag-position training
python scripts/train_deep.py --position Bag --epochs 50 --batch-size 512 --device auto

# Four-position early-fusion (concatenate windows along channel dim) — planned Stage 4
python scripts/train_deep.py --position Bag Hand Hips Torso --epochs 50 --device auto

Outputs saved to:  outputs/execution-output/<run_name>/
  model.pt     — model state dict
  config.json  — all hyperparameters
  metrics.json — per-epoch and final validation metrics
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HDF5_PATH    = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
DEFAULT_OUTD = REPO_ROOT / "outputs" / "execution-output"

LABEL_MAP = {0:"Still",1:"Walking",2:"Run",3:"Bike",
             4:"Car",5:"Bus",6:"Train",7:"Metro"}
N_CLASSES = 8


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = total_correct = total_n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with __import__("torch").amp.autocast("cuda"):
                logits = model(x)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
        total_loss    += loss.item() * len(y)
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
            x = x.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc    = float((preds == labels).mean())
    f1     = float(f1_score(labels, preds, average="macro", zero_division=0))
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
    parser.add_argument("--position",     default="Bag",
                        choices=["Bag", "Hand", "Hips", "Torso"])
    parser.add_argument("--sample-limit", type=int, default=None,
                        help="Stratified window limit per split (None = full dataset)")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=512)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--device",       default="cuda",
                        help="cpu | cuda | cuda:0 … (default: cuda)")
    parser.add_argument("--num-workers",  type=int,   default=0,
                        help="DataLoader workers (0 = main process, safest for HDF5)")
    parser.add_argument("--output-dir",   type=Path,  default=DEFAULT_OUTD)
    parser.add_argument("--patience",     type=int,   default=0,
                        help="Early-stopping patience (epochs without val macro-F1 improvement). "
                             "0 = disabled.")
    args = parser.parse_args()

    # ---- imports (fail early with helpful messages) ----
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        print("ERROR: PyTorch not installed — run: pip install torch")
        sys.exit(1)
    try:
        from sklearn.metrics import classification_report
    except ImportError:
        print("ERROR: scikit-learn not installed — run: pip install scikit-learn")
        sys.exit(1)

    from featureflyers_shl.data.dataset   import SHLWindowDataset
    from featureflyers_shl.models.cnn1d  import CNN1D

    # ---- reproducibility ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- device ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = device.type == "cuda"
    print(f"\nDevice  : {device}  (AMP={'on' if use_amp else 'off'})")

    if not args.hdf5_path.exists():
        print(f"ERROR: HDF5 not found: {args.hdf5_path}")
        sys.exit(1)

    # ---- run name & output dir ----
    lim_str  = str(args.sample_limit) if args.sample_limit else "full"
    run_name = f"cnn1d_pos{args.position}_s{lim_str}_ep{args.epochs}_bs{args.batch_size}"
    run_dir  = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run     : {run_name}")
    print(f"Output  : {run_dir.relative_to(REPO_ROOT)}\n")

    # ---- datasets ----
    print("Building datasets …")
    t0 = time.time()
    ds_train = SHLWindowDataset(
        args.hdf5_path, split="train",
        position=args.position, sample_limit=args.sample_limit, seed=args.seed,
    )
    ds_val = SHLWindowDataset(
        args.hdf5_path, split="validation",
        position=args.position, sample_limit=args.sample_limit, seed=args.seed + 1,
    )
    print(f"  Train : {len(ds_train):,} windows")
    print(f"  Val   : {len(ds_val):,} windows  ({time.time()-t0:.1f}s)\n")

    # ---- class weights (from training set) ----
    counts = ds_train.class_counts()   # (8,) int64
    print(f"  Class counts (train): { {LABEL_MAP[i]: int(counts[i]) for i in range(N_CLASSES)} }")
    weights = torch.tensor(
        1.0 / np.where(counts > 0, counts.astype(np.float32), 1.0),
        dtype=torch.float32,
        device=device,
    )
    weights = weights / weights.sum() * N_CLASSES   # normalise

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(ds_train, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(ds_val,   shuffle=False, **loader_kw)

    # ---- model ----
    model = CNN1D(n_channels=9, n_classes=N_CLASSES, dropout=args.dropout).to(device)
    n_params = CNN1D.n_params(model)
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"\nModel   : CNN1D  params={n_params:,}  (DataParallel x{n_gpus} GPUs)")
    else:
        print(f"\nModel   : CNN1D  params={n_params:,}")

    criterion  = nn.CrossEntropyLoss(weight=weights)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.amp.GradScaler() if use_amp else None

    # ---- training loop ----
    print(f"\n{'Ep':>4}  {'TrainLoss':>9}  {'TrainAcc':>8}  "
          f"{'ValAcc':>7}  {'MacroF1':>8}  {'LR':>8}  {'Time':>6}")
    print("-" * 68)

    history: list[dict] = []
    best_f1 = 0.0
    best_state = None
    no_improve = 0
    t_train = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion,
                                      device, scaler)
        val_acc, val_f1, _, _ = eval_epoch(model, val_loader, device)
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()
        ep_time = time.time() - t_ep

        print(f"{epoch:>4}  {tr_loss:>9.4f}  {tr_acc:>7.3%}  "
              f"{val_acc:>7.3%}  {val_f1:>8.4f}  {lr_now:>8.2e}  {ep_time:>5.1f}s")

        rec = dict(
            epoch=epoch, train_loss=round(tr_loss, 5), train_acc=round(tr_acc, 4),
            val_acc=round(val_acc, 4), val_macro_f1=round(val_f1, 4),
            lr=round(lr_now, 8), epoch_time_s=round(ep_time, 1),
        )
        history.append(rec)

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

    # ---- final eval on best model ----
    _m = model.module if hasattr(model, "module") else model
    _m.load_state_dict(best_state)
    final_acc, final_f1, final_preds, final_labels = eval_epoch(model, val_loader, device)
    present = sorted(set(final_labels.tolist()))
    report  = classification_report(
        final_labels, final_preds,
        labels=present, target_names=[LABEL_MAP[l] for l in present],
        zero_division=0,
    )
    print(f"\n{'='*60}")
    print(f"Best model — Accuracy: {final_acc:.4f}   Macro-F1: {final_f1:.4f}")
    print(f"{'='*60}")
    print(report)

    # ---- confusion matrix (text) ----
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(final_labels, final_preds,
                          labels=list(range(N_CLASSES)))
    present_names = [LABEL_MAP[i] for i in range(N_CLASSES)]
    header = "Pred ->  " + "  ".join(f"{n:>8}" for n in present_names)
    cm_lines = [header]
    for i, row in enumerate(cm):
        row_str = f"{present_names[i]:>8} | " + "  ".join(f"{v:>8d}" for v in row)
        cm_lines.append(row_str)
    cm_text = "\n".join(cm_lines)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm_text)
    (run_dir / "confusion_matrix.txt").write_text(cm_text)

    # ---- save artefacts ----
    torch.save(best_state, run_dir / "model.pt")

    config = dict(
        model="CNN1D", position=args.position, sample_limit=args.sample_limit,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        dropout=args.dropout, seed=args.seed, patience=args.patience,
        device=str(device), amp=use_amp, n_params=n_params,
        n_train=len(ds_train), n_val=len(ds_val),
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    metrics = dict(
        best_val_macro_f1=round(final_f1, 4),
        best_val_accuracy=round(final_acc, 4),
        total_time_s=round(total_time, 1),
        epochs_trained=len(history),
        early_stopped=(args.patience > 0 and no_improve >= args.patience),
        history=history,
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "classification_report.txt").write_text(report)

    print(f"\nSaved → {run_dir.relative_to(REPO_ROOT)}/")
    print(f"  model.pt  config.json  metrics.json  classification_report.txt  confusion_matrix.txt")


if __name__ == "__main__":
    main()
