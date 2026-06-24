#!/usr/bin/env python3
"""
Stage 13 — GPU-MiniRocket (offline feature precomputation) for SHL 2026.

Two-phase approach that reduces total runtime from ~15 hours to ~25 minutes:

  Phase 1 — Feature extraction (one GPU pass, ~15 min):
    Raw window (B, 9, 500) → z-score → frozen MiniRocket kernels → PPV features
    Stored as float16 CPU tensors (≈85 GB for 1.57 M training windows).

  Phase 2 — Linear head training (CPU→GPU per batch, ~10 min):
    Pre-extracted features (B, 27000) → BN → Dropout → Linear(27000, 8) → logits
    Per-epoch time drops from ~900 s to ~15 s (no Conv1d per epoch).

The saved model.pt stores the full MiniRocketModel state dict (extractor buffers +
head parameters) so ensemble inference works with model(x) unchanged.

Usage
-----
# Smoke test
python scripts/train_stage13_minirocket.py --sample-limit 5000 --epochs 3 --device cuda:0

# Full run (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_stage13_minirocket.py \\
    --loss focal --focal-gamma 2.0 --sampler balanced --class-weights none \\
    --seed 42 --device cuda:0 \\
    > outputs/execution-output/logs/stage13_gpu0_minirocket.log 2>&1 &
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

REPO_ROOT    = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HDF5_PATH    = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
DEFAULT_OUTD = REPO_ROOT / "outputs" / "execution-output"
POSITIONS    = ["Bag", "Hand", "Hips", "Torso"]
N_CLASSES    = 8
LABEL_MAP    = {0:"Still",1:"Walking",2:"Run",3:"Bike",4:"Car",5:"Bus",6:"Train",7:"Metro"}


# ---------------------------------------------------------------------------
# Model (identical interface to original — model(x) works for inference)
# ---------------------------------------------------------------------------

class MiniRocketExtractor(nn.Module):
    """Frozen random convolutional kernel bank (PPV features)."""

    def __init__(
        self,
        n_kernels: int = 1000,
        kernel_size: int = 9,
        dilations: list[int] | None = None,
        n_channels: int = 9,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if dilations is None:
            dilations = [1, 4, 16]
        self.dilations   = dilations
        self.n_channels  = n_channels
        self.n_kernels   = n_kernels
        self.kernel_size = kernel_size

        rng = np.random.default_rng(seed)
        W   = np.zeros((n_kernels, 1, kernel_size), dtype=np.float32)
        for i in range(n_kernels):
            pos = rng.choice(kernel_size, size=3, replace=False)
            W[i, 0, pos] = rng.choice([-1.0, 1.0], size=3)
        self.register_buffer("kernels", torch.from_numpy(W))
        self.feature_dim = n_kernels * len(dilations) * n_channels

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        feats = []
        for d in self.dilations:
            pad = d * (self.kernel_size - 1) // 2
            for c in range(C):
                xc  = x[:, c:c+1, :]
                out = F.conv1d(xc, self.kernels, padding=pad, dilation=d)[:, :, :T]
                ppv = (out > 0).float().mean(dim=2)
                feats.append(ppv)
        return torch.cat(feats, dim=1)


class MiniRocketModel(nn.Module):
    """Frozen extractor + trainable BN + Linear head (same interface as original)."""

    def __init__(
        self,
        n_kernels: int = 1000,
        kernel_size: int = 9,
        dilations: list[int] | None = None,
        n_channels: int = 9,
        n_classes: int = 8,
        dropout: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.extractor = MiniRocketExtractor(n_kernels, kernel_size, dilations, n_channels, seed)
        feat_dim = self.extractor.feature_dim
        self.bn      = nn.BatchNorm1d(feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(feat_dim, n_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        for p in self.extractor.parameters():
            p.requires_grad_(False)

        self.feat_dim        = feat_dim
        self.n_params_head   = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extractor(x)
        return self.head(self.dropout(self.bn(feats)))

    def forward_feats(self, feats: torch.Tensor) -> torch.Tensor:
        """Forward pass when precomputed float32 features are already available."""
        return self.head(self.dropout(self.bn(feats)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_perwindow(x: torch.Tensor) -> torch.Tensor:
    x   = x.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    mu  = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (x - mu) / std


@torch.no_grad()
def extract_features_from_array(
    extractor: MiniRocketExtractor,
    raw_np: np.ndarray,        # (N, 500, 9) float32
    device: torch.device,
    batch_size: int = 512,
    desc: str = "",
    out: torch.Tensor | None = None,   # pre-allocated slice to write into
) -> torch.Tensor:
    """Batch-extract float16 PPV features into `out` (or a new tensor if None)."""
    extractor.eval()
    N        = len(raw_np)
    feat_dim = extractor.feature_dim
    if out is None:
        out = torch.empty((N, feat_dim), dtype=torch.float16)

    t0 = time.time()
    n_batches = (N + batch_size - 1) // batch_size
    for i, start in enumerate(range(0, N, batch_size)):
        end     = min(start + batch_size, N)
        batch   = raw_np[start:end]                              # (B, 500, 9)
        x       = torch.from_numpy(batch.astype(np.float32)).permute(0, 2, 1).to(device)  # (B, 9, 500)
        x       = _norm_perwindow(x)
        out[start:end] = extractor(x).cpu().half()

        if (i + 1) % 500 == 0 or (i + 1) == n_batches:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (n_batches - i - 1)
            print(f"    {desc}  [{i+1}/{n_batches}]  {elapsed:.0f}s elapsed  ETA {eta:.0f}s", flush=True)

    return out   # view into the pre-allocated slice


def train_epoch_feats(
    model: MiniRocketModel,
    feat_f16: torch.Tensor,    # (N, feat_dim) float16 CPU
    labels: torch.Tensor,      # (N,) int64 CPU
    idx: np.ndarray,           # epoch sample order (may be balanced oversample)
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = total_correct = total_n = 0
    for start in range(0, len(idx), batch_size):
        bidx    = idx[start : start + batch_size]
        x       = feat_f16[bidx].float().to(device)
        y       = labels[bidx].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits  = model.forward_feats(x)
        loss    = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        l = loss.item()
        if l == l:
            total_loss += l * len(y)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n       += len(y)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def eval_feats(
    model: MiniRocketModel,
    feat_f16: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    from sklearn.metrics import f1_score
    model.eval()
    all_preds = []
    for start in range(0, len(feat_f16), batch_size):
        x = feat_f16[start : start + batch_size].float().to(device)
        all_preds.append(model.forward_feats(x).argmax(1).cpu().numpy())
    preds     = np.concatenate(all_preds)
    labels_np = labels.numpy()
    acc = float((preds == labels_np).mean())
    f1  = float(f1_score(labels_np, preds, average="macro", zero_division=0))
    return acc, f1, preds, labels_np


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
    parser.add_argument("--epochs",          type=int,   default=60)
    parser.add_argument("--batch-size",      type=int,   default=4096,
                        help="Large: no GPU Conv1d per batch in Phase 2")
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--dropout",         type=float, default=0.1)
    parser.add_argument("--n-kernels",       type=int,   default=1000)
    parser.add_argument("--dilations",       type=int,   nargs="+", default=[1, 4, 16])
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

    from featureflyers_shl.data.dataset    import SHLWindowDataset
    from featureflyers_shl.training.losses import build_criterion
    from sklearn.metrics import classification_report

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device  = torch.device(args.device)
    lim_str = str(args.sample_limit) if args.sample_limit else "full"
    loss_tag    = f"focal_g{args.focal_gamma}" if args.loss == "focal" else "ce"
    sampler_tag = "_balsampler" if args.sampler == "balanced" else ""
    dil_str     = "-".join(str(d) for d in args.dilations)
    run_name    = (f"minirocket_posPool_k{args.n_kernels}_dil{dil_str}"
                   f"_s{lim_str}_{loss_tag}{sampler_tag}"
                   f"_ep{args.epochs}_bs{args.batch_size}_precomp")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    feat_dim = args.n_kernels * len(args.dilations) * 9

    print(f"\nStage 13 — GPU-MiniRocket (offline precomputation)")
    print(f"  device       : {device}")
    print(f"  n_kernels    : {args.n_kernels}  kernel_size=9  dilations={args.dilations}")
    print(f"  feature_dim  : {feat_dim:,}  (n_kernels × dilations × 9 channels)")
    print(f"  epochs       : {args.epochs}  patience={args.patience}  batch_size={args.batch_size}")
    print(f"  loss         : {args.loss}" + (f"  gamma={args.focal_gamma}" if args.loss == "focal" else ""))
    print(f"  sampler      : {args.sampler}")
    print(f"  output       : {run_dir.relative_to(REPO_ROOT)}\n")

    # -----------------------------------------------------------------------
    # Phase 1: Offline feature extraction (one GPU pass per position)
    # -----------------------------------------------------------------------
    model = MiniRocketModel(
        n_kernels=args.n_kernels, dilations=args.dilations,
        n_channels=9, n_classes=N_CLASSES, dropout=args.dropout, seed=args.seed,
    ).to(device)
    print(f"Model         : MiniRocketModel")
    print(f"  feat_dim    : {feat_dim:,}")
    print(f"  head params : {model.n_params_head:,}  (BN + Dropout + Linear)\n")

    print("=" * 60)
    print("Phase 1 — Offline feature extraction (runs once)")
    print(f"  RAM needed: {feat_dim * 1_568_568 * 2 / 1e9:.1f} GB float16 for all 4 positions")
    print("=" * 60)
    t_extract = time.time()

    # Probe counts without loading data to size the pre-allocated tensor
    print("  Probing window counts …", flush=True)
    pos_counts = []
    for pos in POSITIONS:
        ds_probe = SHLWindowDataset(args.hdf5_path, split="train", position=pos,
                                    sample_limit=args.sample_limit, seed=args.seed, preload=False)
        pos_counts.append(len(ds_probe))
        del ds_probe
    N_train_total = sum(pos_counts)
    print(f"  Pre-allocating {N_train_total:,} × {feat_dim:,} float16 "
          f"({N_train_total * feat_dim * 2 / 1e9:.1f} GB) …", flush=True)

    # Single allocation — each position writes directly into its slice (no torch.cat needed)
    train_feats  = torch.empty((N_train_total, feat_dim), dtype=torch.float16)
    train_labels = torch.empty(N_train_total, dtype=torch.int64)
    offset = 0

    for pos, N_pos in zip(POSITIONS, pos_counts):
        print(f"\n  Loading {pos} windows …", flush=True)
        ds = SHLWindowDataset(args.hdf5_path, split="train", position=pos,
                              sample_limit=args.sample_limit, seed=args.seed, preload=True)
        print(f"    {len(ds):,} windows loaded  →  extracting features …", flush=True)
        extract_features_from_array(
            model.extractor, ds._X, device, batch_size=512, desc=pos,
            out=train_feats[offset : offset + N_pos])
        train_labels[offset : offset + N_pos] = torch.from_numpy(ds._labels.astype(np.int64))
        offset += N_pos
        del ds   # free raw windows immediately; features already in train_feats

    print(f"\n  Loading val windows (Bag) …", flush=True)
    ds_val     = SHLWindowDataset(args.hdf5_path, split="validation", position="Bag",
                                  sample_limit=None, seed=args.seed + 1, preload=True)
    val_feats  = extract_features_from_array(
        model.extractor, ds_val._X, device, batch_size=512, desc="Val")
    val_labels = torch.from_numpy(ds_val._labels.astype(np.int64))
    del ds_val

    N_train = len(train_feats)
    extract_time = time.time() - t_extract
    feat_gb = train_feats.element_size() * train_feats.nelement() / 1e9
    print(f"\nPhase 1 done in {extract_time:.1f}s")
    print(f"  train feats : {N_train:,} × {feat_dim:,}  ({feat_gb:.1f} GB float16)")
    print(f"  val feats   : {len(val_feats):,} × {feat_dim:,}")

    counts_np = np.bincount(train_labels.numpy(), minlength=N_CLASSES).astype(np.float64)
    counts_t  = torch.from_numpy(counts_np).float()

    print("\nTraining class distribution:")
    total_n = int(counts_np.sum())
    for i, (name, cnt) in enumerate(zip(LABEL_MAP.values(), counts_np)):
        pct = 100.0 * cnt / total_n
        print(f"  [{i}] {name:<10} : {int(cnt):>9,}  ({pct:5.1f}%)  {'#' * max(1,int(pct/2))}")

    # -----------------------------------------------------------------------
    # Phase 2: Train linear head on precomputed features
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 2 — Linear head training on precomputed features")
    print("=" * 60)

    criterion = build_criterion(
        loss_type=args.loss, class_weights=args.class_weights,
        counts=counts_t, n_classes=N_CLASSES,
        focal_gamma=args.focal_gamma, label_smoothing=args.label_smoothing,
        device=device,
    )
    print(f"\nCriterion : {criterion}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Balanced oversampling: pre-generate epoch index arrays
    rng = np.random.default_rng(args.seed + 1)
    if args.sampler == "balanced":
        inv_freq = 1.0 / np.where(counts_np > 0, counts_np, 1.0)
        sample_w = inv_freq[train_labels.numpy()]
        sample_w = (sample_w / sample_w.sum()).astype(np.float64)
        print(f"Sampler   : balanced oversampling (p ∝ 1/class_count)")
    else:
        sample_w = None
        print(f"Sampler   : random shuffle")

    eval_batch = args.batch_size * 4   # no grad → larger batches for eval

    print(f"\n{'Ep':>4}  {'TrainLoss':>9}  {'TrainAcc':>8}  {'ValAcc':>7}  "
          f"{'MacroF1':>8}  {'LR':>8}  {'Time':>6}")
    print("-" * 68)

    best_f1    = 0.0
    best_state = None
    no_improve = 0
    history    = []
    t_train    = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()

        if sample_w is not None:
            epoch_idx = rng.choice(N_train, size=N_train, replace=True, p=sample_w).astype(np.int64)
        else:
            epoch_idx = rng.permutation(N_train).astype(np.int64)

        tr_loss, tr_acc = train_epoch_feats(
            model, train_feats, train_labels, epoch_idx,
            optimizer, criterion, args.batch_size, device)
        val_acc, val_f1, val_preds, val_labels_np = eval_feats(
            model, val_feats, val_labels, eval_batch, device)

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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds, best_labels_arr = val_preds, val_labels_np
            no_improve = 0
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break

    total_time = time.time() - t_train
    print(f"\nTraining done in {total_time:.1f}s   best val macro-F1 = {best_f1:.4f}")

    # --- Save artifacts (full model state for inference compatibility) ---
    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "model.pt")

    cfg = dict(
        model="MiniRocket", n_kernels=args.n_kernels, dilations=args.dilations,
        kernel_size=9, n_channels=9, feature_dim=feat_dim, dropout=args.dropout,
        n_params_head=model.n_params_head, fusion="pool", positions=POSITIONS,
        loss=args.loss, focal_gamma=args.focal_gamma,
        class_weights=args.class_weights, sampler=args.sampler,
        best_f1=round(best_f1, 4), epochs_trained=len(history),
        extract_time_s=round(extract_time, 1),
        train_time_s=round(total_time, 1),
        total_time_s=round(extract_time + total_time, 1),
        seed=args.seed,
    )
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(dict(**cfg, history=history), indent=2))

    report = classification_report(best_labels_arr, best_preds,
                                   target_names=list(LABEL_MAP.values()), zero_division=0)
    (run_dir / "classification_report.txt").write_text(report)

    print(f"\nArtifacts saved to {run_dir.relative_to(REPO_ROOT)}/")
    print(f"  Macro-F1    : {best_f1:.4f}")
    print(f"  Extract time: {extract_time:.1f}s")
    print(f"  Train time  : {total_time:.1f}s")
    print(report)


if __name__ == "__main__":
    main()
