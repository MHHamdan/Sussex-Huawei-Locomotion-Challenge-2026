#!/usr/bin/env python3
"""
Stage 17 — Multi-View Position Fusion Network (MVPF) for SHL 2026.

All four sensor positions (Bag, Hand, Hips, Torso) are processed jointly.
The shared ResNet1D encoder extracts per-position features; a 2-layer
cross-position transformer learns which combination of positions matters for
each locomotion class.

Key advantage over pooled-ensemble models: the model sees cross-position
correlations within each sample (e.g. "Bag vibrates but Torso is smooth")
instead of averaging independent predictions after the fact.

Architecture: PositionEncoder (shared, 3-stage ResNet1D) + CrossPositionTransformer
n_params ≈ 1.45 M

Usage
-----
# Smoke test
python scripts/train_stage17_mvpf.py --sample-limit 5000 --epochs 3 --device cuda:0

# Full run (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_stage17_mvpf.py \\
    --loss focal --focal-gamma 2.0 --sampler balanced --class-weights none \\
    --seed 42 --device cuda:0 \\
    > outputs/execution-output/logs/stage17_gpu0_mvpf.log 2>&1 &
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

REPO_ROOT    = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HDF5_PATH    = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
DEFAULT_OUTD = REPO_ROOT / "outputs" / "execution-output"
POSITIONS    = ["Bag", "Hand", "Hips", "Torso"]
N_CLASSES    = 8
LABEL_MAP    = {0:"Still", 1:"Walking", 2:"Run", 3:"Bike",
                4:"Car",   5:"Bus",     6:"Train", 7:"Metro"}


# ---------------------------------------------------------------------------
# Multi-position dataset
# ---------------------------------------------------------------------------

class MultiPositionWindowDataset(Dataset):
    """
    Loads all 4 sensor positions from HDF5 and returns stacked (4, 9, 500) tensors.

    Assumes windows are temporally aligned across positions (same index = same
    time slice), which holds for the SHL 2026 HDF5 since all positions share the
    same sliding-window grid over the same recording session.
    """

    def __init__(
        self,
        hdf5_path: Path,
        split: str,
        positions: list[str] = POSITIONS,
        sample_limit: int | None = None,
        seed: int = 42,
        preload: bool = True,
    ) -> None:
        from featureflyers_shl.data.dataset import SHLWindowDataset

        self._datasets = []
        for pos in positions:
            ds = SHLWindowDataset(
                hdf5_path, split=split, position=pos,
                sample_limit=sample_limit, seed=seed, preload=preload,
            )
            self._datasets.append(ds)

        lengths = [len(ds) for ds in self._datasets]
        assert len(set(lengths)) == 1, f"Position window counts differ: {dict(zip(positions, lengths))}"
        self._n = lengths[0]

        ref_labels = self._datasets[0]._labels
        for ds, pos in zip(self._datasets[1:], positions[1:]):
            n_mm = (ds._labels != ref_labels).sum()
            if n_mm > 0:
                print(f"  WARNING: {n_mm} label mismatches for {pos} — positions may not be aligned")

        self._labels = ref_labels   # (N,) int64 numpy

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        xs = []
        for ds in self._datasets:
            window = ds._X[idx]                                     # (500, 9) numpy
            x = torch.from_numpy(window.astype(np.float32)).T      # (9, 500)
            xs.append(x)
        x_stack = torch.stack(xs, dim=0)   # (4, 9, 500)
        y = int(self._labels[idx])
        return x_stack, y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_perwindow(x: torch.Tensor) -> torch.Tensor:
    """z-score each channel of each window independently.  x: (B, P, C, T)"""
    x   = x.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    mu  = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (x - mu) / std


def _augment(x: torch.Tensor, jitter_sigma: float = 0.05, pos_drop_p: float = 0.25) -> torch.Tensor:
    """Train-time augmentation on normalized x: (B, P, C, T).

    Jitter: Gaussian noise with sigma=5% of unit std prevents memorisation
    of exact per-window patterns without changing relative channel structure.

    Position dropout: zero one random position per sample with p=pos_drop_p
    forces the cross-position transformer to learn redundant multi-view
    representations rather than collapsing onto a single dominant position.
    """
    x = x + torch.randn_like(x) * jitter_sigma
    B, P = x.shape[:2]
    drop_mask = torch.rand(B, device=x.device) < pos_drop_p             # (B,)
    drop_pos  = torch.randint(0, P, (B,), device=x.device)              # (B,)
    pos_idx   = torch.arange(P, device=x.device).unsqueeze(0)           # (1, P)
    kill_mask = (pos_idx == drop_pos.unsqueeze(1)) & drop_mask.unsqueeze(1)  # (B, P)
    x = x * (~kill_mask).float().unsqueeze(-1).unsqueeze(-1)
    return x


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = total_correct = total_n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x    = _norm_perwindow(x)
        x    = _augment(x)
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
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--feat-dim",        type=int,   default=256)
    parser.add_argument("--base-filters",    type=int,   default=64)
    parser.add_argument("--n-heads",         type=int,   default=4)
    parser.add_argument("--n-tf-layers",     type=int,   default=2)
    parser.add_argument("--dropout",         type=float, default=0.4)
    parser.add_argument("--encoder-dropout", type=float, default=0.2)
    parser.add_argument("--patience",        type=int,   default=15)
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
    from featureflyers_shl.models.mvpf        import MVPF
    from featureflyers_shl.training.losses    import build_criterion

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device  = torch.device(args.device)
    use_amp = device.type == "cuda"
    lim_str = str(args.sample_limit) if args.sample_limit else "full"

    loss_tag    = f"focal_g{args.focal_gamma}" if args.loss == "focal" else "ce"
    sampler_tag = "_balsampler" if args.sampler == "balanced" else ""
    run_name    = (f"mvpf_4pos_fd{args.feat_dim}_bf{args.base_filters}"
                   f"_h{args.n_heads}tf{args.n_tf_layers}"
                   f"_s{lim_str}_{loss_tag}{sampler_tag}_aug"
                   f"_ep{args.epochs}_bs{args.batch_size}")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStage 17 — Multi-View Position Fusion (MVPF)")
    print(f"  device     : {device}  (AMP={'on' if use_amp else 'off'})")
    print(f"  feat_dim   : {args.feat_dim}  base_filters={args.base_filters}")
    print(f"  transformer: {args.n_tf_layers} layers  {args.n_heads} heads")
    print(f"  epochs     : {args.epochs}  patience={args.patience}")
    print(f"  batch_size : {args.batch_size}  (×4 positions = {args.batch_size*4} encoder calls/batch)")
    print(f"  loss       : {args.loss}" + (f"  gamma={args.focal_gamma}" if args.loss == "focal" else ""))
    print(f"  sampler    : {args.sampler}")
    print(f"  output     : {run_dir.relative_to(REPO_ROOT)}\n")

    # -----------------------------------------------------------------------
    # Data loading — all 4 positions for both train and val
    # -----------------------------------------------------------------------
    print("Preloading train windows (all 4 positions) into RAM …")
    t0 = time.time()
    ds_train = MultiPositionWindowDataset(
        args.hdf5_path, split="train", positions=POSITIONS,
        sample_limit=args.sample_limit, seed=args.seed, preload=True,
    )
    for ds, pos in zip(ds_train._datasets, POSITIONS):
        print(f"  {pos}: {len(ds):,} windows")

    print("Preloading val windows (all 4 positions) into RAM …")
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
        inv_freq  = 1.0 / np.where(counts_np > 0, counts_np, 1.0)
        sample_w  = torch.from_numpy(inv_freq[ds_train._labels].astype(np.float32))
        sampler   = WeightedRandomSampler(sample_w, num_samples=len(ds_train), replacement=True)
        train_loader = DataLoader(ds_train, sampler=sampler, **loader_kw)
        print("Sampler   : WeightedRandomSampler (balanced)")
    else:
        train_loader = DataLoader(ds_train, shuffle=True, **loader_kw)
        print("Sampler   : random shuffle")

    val_loader = DataLoader(ds_val, shuffle=False, **loader_kw)
    print()

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = MVPF(
        n_positions=len(POSITIONS), n_channels=9, n_classes=N_CLASSES,
        feat_dim=args.feat_dim, base_filters=args.base_filters,
        n_heads=args.n_heads, n_tf_layers=args.n_tf_layers,
        dropout=args.dropout, encoder_dropout=args.encoder_dropout,
    ).to(device)
    print(f"Model   : MVPF  n_params={model.n_params:,}")
    print(f"  PositionEncoder : shared 3-stage ResNet1D (9→64→128→256→{args.feat_dim})")
    print(f"  CrossPosFusion  : {args.n_tf_layers}×TransformerBlock(d={args.feat_dim}, h={args.n_heads})")
    print(f"  Head            : Dropout({args.dropout}) → Linear({args.feat_dim}, {N_CLASSES})\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    # ReduceLROnPlateau: halves LR after 5 epochs without val-F1 improvement.
    # Responds to the actual learning signal rather than a fixed cosine curve,
    # which barely moves in the first 20 epochs with T_max=80.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
        threshold=1e-4, min_lr=1e-6,
    )
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
        scheduler.step(val_f1)
        lr_now = optimizer.param_groups[0]["lr"]
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

    # -----------------------------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------------------------
    _m = model.module if hasattr(model, "module") else model
    _m.load_state_dict(best_state)
    _m.eval()

    torch.save(best_state, run_dir / "model.pt")

    cfg = dict(
        model="MVPF", n_positions=len(POSITIONS), n_channels=9,
        feat_dim=args.feat_dim, base_filters=args.base_filters,
        n_heads=args.n_heads, n_tf_layers=args.n_tf_layers,
        dropout=args.dropout, n_params=model.n_params,
        positions=POSITIONS, fusion="cross_position_attention",
        loss=args.loss, focal_gamma=args.focal_gamma,
        class_weights=args.class_weights, sampler=args.sampler,
        best_f1=round(best_f1, 4), epochs_trained=len(history),
        total_train_time_s=round(total_time, 1), seed=args.seed,
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
