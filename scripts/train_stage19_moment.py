#!/usr/bin/env python3
"""
Stage 19 — MOMENT-1-large Feature Extraction + MLP Classifier for SHL 2026.

Two-phase workflow:
  Phase A (--extract):  Run frozen MOMENT-1-large over all 4 sensor positions,
                        save per-position mean-pooled 1024-d embeddings to disk.
                        One-time cost: ~20 min on a single 2080 Ti.

  Phase B (--train):    Train a lightweight MLP on cached 4096-d concatenated
                        embeddings.  No GPU memory pressure — pure MLP training.

The 4096-d feature (4 positions × 1024 MOMENT-large d_model) is richer than
our single-position ResNet1D features.  This script adds MOMENT predictions to
the Stage 16 Phase-C meta-blend.

Usage
-----
# Phase A — extract embeddings (GPU 0)
CUDA_VISIBLE_DEVICES=0 python scripts/train_stage19_moment.py --extract \
    --device cuda:0 --batch-size 128

# Phase B — train MLP (CPU or any GPU)
python scripts/train_stage19_moment.py --train --epochs 60 --lr 1e-3

# Both in one go
CUDA_VISIBLE_DEVICES=0 python scripts/train_stage19_moment.py \
    --extract --train --device cuda:0 --batch-size 128 --epochs 60 --lr 1e-3
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

REPO_ROOT    = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HDF5_PATH    = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
DEFAULT_OUTD = REPO_ROOT / "outputs" / "execution-output"
POSITIONS    = ["Bag", "Hand", "Hips", "Torso"]
N_CLASSES    = 8
FEAT_DIM     = 1024   # MOMENT-1-large d_model
SEQ_LEN      = 512    # pad 500 → 512 (64 patches × patch_len=8)
LABEL_MAP    = {0:"Still", 1:"Walking", 2:"Run", 3:"Bike",
                4:"Car",   5:"Bus",     6:"Train", 7:"Metro"}


# ---------------------------------------------------------------------------
# MOMENT wrapper
# ---------------------------------------------------------------------------

class MOMENTEncoder(nn.Module):
    """
    Frozen MOMENT-1-large encoder that returns per-sample mean embeddings.

    For each input (B, C, T):
      1. Pad T → 512 (must be multiple of patch_len=8)
      2. Run MOMENT classify with reduction='mean' → T5 processes (B*C, n_patches, d_model)
         then means over C → (B, n_patches, d_model)
      3. ClassificationHead.forward means over patches → (B, d_model=1024)

    Returns (B, 1024) embeddings.
    """

    def __init__(self, model_id: str = "AutonLab/MOMENT-1-large") -> None:
        super().__init__()
        from momentfm import MOMENTPipeline
        from momentfm.models.moment import ClassificationHead

        self.moment = MOMENTPipeline.from_pretrained(
            model_id,
            model_kwargs={"task_name": "classification", "n_channels": 9, "num_class": N_CLASSES},
        )
        self.moment = self.moment.half()   # fp16 for all submodules → 0.68 GB vs 1.36 GB
        self.moment.task_name = "classification"
        # Override head with mean-reduction head → (B, 1024)
        # Must match the reduction= kwarg passed in forward()
        self.moment.head = ClassificationHead(
            n_channels=9, d_model=FEAT_DIM, n_classes=N_CLASSES,
            head_dropout=0.0, reduction="mean",
        )
        self.d_model = FEAT_DIM

        # Freeze everything
        for p in self.moment.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) with T≤512 → (B, d_model=1024)"""
        B, C, T = x.shape
        if T < SEQ_LEN:
            x = torch.nn.functional.pad(x, (0, SEQ_LEN - T))
        with torch.no_grad(), torch.autocast("cuda"):
            # fp16 model + autocast: pass fp32 input, autocast handles mixed precision.
            # Explicitly pass reduction="mean" to match the "mean" head (linear(1024, 8)).
            out = self.moment(x_enc=x.float(), reduction="mean")
        # out.embeddings: (B, n_patches, d_model) — already mean over channels
        return out.embeddings.mean(dim=1).float()   # mean over patches → (B, 1024)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class MultiPositionWindowDataset(Dataset):
    """Returns (x, y) where x: (4, 9, 500), y: int label."""

    def __init__(self, hdf5_path, split, sample_limit=None, seed=42):
        from featureflyers_shl.data.dataset import SHLWindowDataset
        self._datasets = []
        for pos in POSITIONS:
            ds = SHLWindowDataset(
                hdf5_path, split=split, position=pos,
                sample_limit=sample_limit, seed=seed, preload=True,
            )
            self._datasets.append(ds)
        self._n = len(self._datasets[0])
        self._labels = self._datasets[0]._labels

    def __len__(self): return self._n

    def __getitem__(self, idx):
        windows = [ds[idx][0] for ds in self._datasets]   # each: tensor(9, 500)
        x = torch.stack(windows, dim=0)                    # (4, 9, 500)
        y = int(self._labels[idx])
        return x, y


class CachedEmbeddingDataset(Dataset):
    """Loads pre-extracted (N, 4, 1024) embeddings and labels from .npz."""

    def __init__(self, emb_path: Path) -> None:
        data = np.load(emb_path)
        self.emb    = torch.tensor(data["embeddings"], dtype=torch.float32)  # (N, 4, 1024)
        self.labels = torch.tensor(data["labels"],     dtype=torch.long)     # (N,)
        self.feats  = self.emb.view(len(self.emb), -1)                       # (N, 4096)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# MLP head
# ---------------------------------------------------------------------------

class MOMENTClassifier(nn.Module):
    """
    Lightweight MLP on top of 4096-d MOMENT features.
    4 positions × 1024 MOMENT d_model → (512 → 128 → n_classes).
    """

    def __init__(
        self,
        in_feat: int = 4 * FEAT_DIM,
        hidden1: int = 512,
        hidden2: int = 128,
        n_classes: int = N_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_feat),
            nn.Linear(in_feat, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden1),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma  = gamma
        self.smooth = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none",
                                         label_smoothing=self.smooth)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# Sklearn-style macro F1
# ---------------------------------------------------------------------------

def macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return f1_score(labels, preds, average="macro", zero_division=0)


# ---------------------------------------------------------------------------
# Phase A — Extract MOMENT embeddings
# ---------------------------------------------------------------------------

def extract_embeddings(args) -> None:
    device = torch.device(args.device)

    print("Loading MOMENT-1-large encoder …")
    encoder = MOMENTEncoder(args.moment_model).to(device)
    encoder.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "moment_embeddings"
    cache_dir.mkdir(exist_ok=True)

    extract_splits = set(args.extract_splits.split(",")) if args.extract_splits else {"train", "validation", "test"}

    for split in ("train", "validation"):
        if split not in extract_splits:
            print(f"  {split}: skipped (not in --extract-splits)")
            continue
        emb_path = cache_dir / f"{split}_embeddings.npz"
        if emb_path.exists() and not args.force_extract:
            print(f"  {split}: cache exists — skip (use --force-extract to redo)")
            continue

        print(f"\nExtracting {split} embeddings …")
        ds = MultiPositionWindowDataset(args.hdf5_path, split=split,
                                        sample_limit=args.sample_limit)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

        all_embs   = []
        all_labels = []
        t0 = time.time()

        for step, (x, y) in enumerate(loader, 1):
            # x: (B, 4, 9, 500)
            B, P, C, T = x.shape
            x_flat = x.view(B * P, C, T).to(device)   # (B*4, 9, 500)

            emb = encoder(x_flat)                       # (B*4, 1024)
            emb = emb.view(B, P, FEAT_DIM)              # (B, 4, 1024)

            all_embs.append(emb.cpu().numpy().astype(np.float16))
            all_labels.append(y.numpy())

            if step % 50 == 0 or step == len(loader):
                elapsed = time.time() - t0
                pct = 100 * step / len(loader)
                print(f"  [{step:4d}/{len(loader)}] {pct:.0f}%  {elapsed:.0f}s elapsed")

        embeddings = np.concatenate(all_embs,   axis=0).astype(np.float16)  # (N, 4, 1024)
        labels     = np.concatenate(all_labels, axis=0)
        tmp_emb_path = emb_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp_emb_path, embeddings=embeddings, labels=labels)
        tmp_emb_path.rename(emb_path)
        print(f"  Saved: {emb_path}  shape={embeddings.shape}  "
              f"size={emb_path.stat().st_size / 1e9:.2f} GB")

    # Test set — only 1 position available; repeat ×4 as fallback
    test_emb_path = cache_dir / "test_embeddings.npz"
    if "test" not in extract_splits:
        print("\n  test: skipped (not in --extract-splits)")
    elif test_emb_path.exists() and not args.force_extract:
        print(f"\n  test: cache exists — skip")
    else:
        print("\nExtracting test embeddings (single-position ×4 fallback) …")
        from featureflyers_shl.data.dataset import SHLWindowDataset

        class _SinglePosWrapper(Dataset):
            def __init__(self, ds):
                self.ds = ds
                self._labels = ds._labels
            def __len__(self): return len(self.ds)
            def __getitem__(self, idx):
                x, y = self.ds[idx]            # (9, 500), int
                x4 = x.unsqueeze(0).repeat(4, 1, 1)   # (4, 9, 500)
                return x4, int(y)

        ds_test  = SHLWindowDataset(args.hdf5_path, split="test", position="Hips",
                                    preload=True)
        ds_wrap  = _SinglePosWrapper(ds_test)
        t_loader = DataLoader(ds_wrap, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

        embs       = []
        all_labels = []
        t0 = time.time()
        for step, (x, y) in enumerate(t_loader, 1):
            B, P, C, T = x.shape
            x_flat = x.view(B * P, C, T).to(device)
            emb = encoder(x_flat).view(B, P, FEAT_DIM)
            embs.append(emb.cpu().numpy().astype(np.float16))
            all_labels.append(y.numpy())
            if step % 50 == 0 or step == len(t_loader):
                print(f"  test [{step:4d}/{len(t_loader)}]  {time.time()-t0:.0f}s")

        test_embs  = np.concatenate(embs,       axis=0).astype(np.float16)
        test_labels = np.concatenate(all_labels, axis=0)
        tmp_test_path = test_emb_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp_test_path, embeddings=test_embs, labels=test_labels)
        tmp_test_path.rename(test_emb_path)
        print(f"  Saved: {test_emb_path}  shape={test_embs.shape}")

    print("\nExtraction complete.")


# ---------------------------------------------------------------------------
# Phase B — Train MLP on cached embeddings
# ---------------------------------------------------------------------------

def train_mlp(args) -> None:
    cache_dir = Path(args.output_dir) / "moment_embeddings"
    train_path = cache_dir / "train_embeddings.npz"
    val_path   = cache_dir / "validation_embeddings.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"Run --extract first.  Missing: {train_path}")

    print("Loading cached MOMENT embeddings …")
    ds_train = CachedEmbeddingDataset(train_path)
    ds_val   = CachedEmbeddingDataset(val_path)
    print(f"  Train: {len(ds_train):,}  Val: {len(ds_val):,}")
    print(f"  Feature dim: {ds_train.feats.shape[1]}")

    # Balanced sampler
    labels = ds_train.labels.numpy()
    counts = np.bincount(labels, minlength=N_CLASSES)
    weights = 1.0 / counts[labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

    train_loader = DataLoader(ds_train, batch_size=args.mlp_batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=args.mlp_batch_size * 4, shuffle=False,
                              num_workers=4, pin_memory=True)

    device = torch.device(args.mlp_device)
    model  = MOMENTClassifier(dropout=0.3).to(device)
    total  = sum(p.numel() for p in model.parameters())
    print(f"  MLP params: {total:,}")

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    run_name = f"moment_large_mlp_ep{args.epochs}_lr{args.lr:.0e}_bs{args.mlp_batch_size}"
    run_dir  = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_f1  = 0.0
    best_ep  = 0

    print(f"\nTraining MLP  ({args.epochs} epochs, device={args.mlp_device})")
    print(f"  output: {run_dir}")
    print(f"{'Ep':>4}  {'TrLoss':>8}  {'TrAcc':>8}  {'ValAcc':>8}  {'F1':>8}  "
          f"{'LR':>10}  {'Time':>7}")
    print("-" * 68)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = tr_correct = tr_total = 0

        for feats, labels_b in train_loader:
            feats    = feats.to(device)
            labels_b = labels_b.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss   = criterion(logits, labels_b)
            loss.backward()
            optimizer.step()

            tr_loss    += loss.item() * len(labels_b)
            tr_correct += (logits.argmax(1) == labels_b).sum().item()
            tr_total   += len(labels_b)

        scheduler.step()

        # Validation
        model.eval()
        val_preds  = []
        val_labels = []
        with torch.no_grad():
            for feats, labels_b in val_loader:
                logits = model(feats.to(device))
                val_preds.append(logits.argmax(1).cpu().numpy())
                val_labels.append(labels_b.numpy())

        val_preds  = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_acc    = (val_preds == val_labels).mean() * 100
        val_f1     = macro_f1(val_preds, val_labels)
        tr_acc     = tr_correct / tr_total * 100
        tr_loss_avg = tr_loss / tr_total
        lr_now      = optimizer.param_groups[0]["lr"]
        ep_time     = time.time() - t0

        flag = " *" if val_f1 > best_f1 else ""
        print(f"{epoch:4d}  {tr_loss_avg:>8.4f}  {tr_acc:>7.3f}%  {val_acc:>7.3f}%  "
              f"{val_f1:>8.4f}  {lr_now:>10.2e}  {ep_time:>5.1f}s{flag}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_ep = epoch
            torch.save(model.state_dict(), run_dir / "model.pt")
            config = dict(model="MOMENTClassifier", moment_model=args.moment_model,
                          feat_dim=FEAT_DIM, n_positions=4,
                          in_feat=4*FEAT_DIM, best_val_macro_f1=round(best_f1, 4),
                          best_epoch=best_ep, epochs=args.epochs, lr=args.lr)
            (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\nBest val macro-F1 = {best_f1:.4f} @ ep{best_ep}")
    print(f"Checkpoint: {run_dir / 'model.pt'}")

    # Save val predictions for meta-blend inspection
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()
    all_probs  = []
    all_labels = []
    with torch.no_grad():
        for feats, labels_b in val_loader:
            logits = model(feats.to(device))
            probs  = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels_b.numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    np.save(run_dir / "val_probs.npy",  probs)
    np.save(run_dir / "val_labels.npy", labels)
    print(f"Val probs saved → {run_dir}/val_probs.npy  shape={probs.shape}")

    # Save test predictions
    test_emb_path = cache_dir / "test_embeddings.npz"
    if test_emb_path.exists():
        ds_test  = CachedEmbeddingDataset(test_emb_path)
        t_loader = DataLoader(ds_test, batch_size=args.mlp_batch_size * 4, shuffle=False,
                              num_workers=2)
        test_probs = []
        with torch.no_grad():
            for feats, _ in t_loader:
                logits = model(feats.to(device))
                test_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        test_probs = np.concatenate(test_probs)
        np.save(run_dir / "test_probs.npy", test_probs)
        print(f"Test probs saved → {run_dir}/test_probs.npy  shape={test_probs.shape}")

    # Classification report
    from sklearn.metrics import classification_report
    report = classification_report(labels, probs.argmax(1),
                                   target_names=list(LABEL_MAP.values()))
    print("\nClassification Report (best val epoch):")
    print(report)
    (run_dir / "classification_report.txt").write_text(report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Stage 19 MOMENT feature extraction + MLP")
    p.add_argument("--extract",      action="store_true", help="Run Phase A: extract embeddings")
    p.add_argument("--train",        action="store_true", help="Run Phase B: train MLP")
    p.add_argument("--force-extract",  action="store_true", help="Overwrite existing embedding cache")
    p.add_argument("--extract-splits", type=str, default=None,
                   help="Comma-separated splits to extract, e.g. 'train,validation,test'. Default: all.")

    # Shared
    p.add_argument("--hdf5-path",    type=Path,  default=HDF5_PATH)
    p.add_argument("--output-dir",   type=Path,  default=DEFAULT_OUTD)
    p.add_argument("--sample-limit", type=int,   default=None)
    p.add_argument("--moment-model", type=str,   default="AutonLab/MOMENT-1-large")

    # Phase A
    p.add_argument("--device",       type=str,   default="cuda:0")
    p.add_argument("--batch-size",   type=int,   default=128,
                   help="Batch size for MOMENT embedding extraction")

    # Phase B
    p.add_argument("--mlp-device",   type=str,   default="cpu")
    p.add_argument("--mlp-batch-size", type=int, default=512)
    p.add_argument("--epochs",       type=int,   default=60)
    p.add_argument("--lr",           type=float, default=1e-3)

    args = p.parse_args()

    if not args.extract and not args.train:
        p.error("Specify at least one of --extract / --train")

    if args.extract:
        extract_embeddings(args)

    if args.train:
        train_mlp(args)


if __name__ == "__main__":
    main()
