#!/usr/bin/env python3
"""
Stage 23 — Frozen MOMENT embeddings + hand-crafted statistical/spectral features
            → MLP classifier (strongest pure foundation-model submission path).

Feature vector per window (5,512-d total):
  - MOMENT-1-large frozen embeddings: 4 positions × 1,024-d  = 4,096-d
  - Statistical/spectral hand-crafted: 4 positions × 354-d   = 1,416-d
  Combined: 5,512-d  → 3-layer MLP (same architecture as Stage 19 but richer input)

Why MLP over LightGBM for this input:
  MOMENT embeddings encode meaning across ALL 1024 dimensions jointly.
  Tree models (LightGBM/XGB) split one dimension at a time and cannot form
  linear combinations — they see the embedding as noise.  MLP learns linear
  combinations in the first layer and naturally exploits the full embedding
  geometry.  Stage 22 (LightGBM on 5512-d) reached F1=0.6539; MLP is expected
  to exceed both Stage 22 and Stage 19 (MOMENT-only MLP, F1=0.7681).

SHL 2026 compliance:
  - Foundation weights: NEVER updated (frozen MOMENT extraction only)
  - Trained component: MLP head (lightweight, ~2M params on 5512-d input)
  - No scratch-trained deep models used

Usage
-----
CUDA_VISIBLE_DEVICES=0 python scripts/train_stage23_moment_stat_mlp.py
CUDA_VISIBLE_DEVICES=0 python scripts/train_stage23_moment_stat_mlp.py --epochs 60 --lr 5e-4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HDF5_PATH = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
FEAT_DIR  = REPO_ROOT / "dataset" / "processed" / "features"
EMB_DIR   = REPO_ROOT / "outputs" / "execution-output" / "moment_embeddings"
OUT_DIR   = REPO_ROOT / "outputs" / "execution-output" / "moment_stat_mlp_stage23"
SUB_DIR   = REPO_ROOT / "outputs" / "execution-output" / "submissions"

POSITIONS = ["Bag", "Hand", "Hips", "Torso"]
WIN_SIZE  = 500
FFT_TOP_K = 20
N_CLASSES = 8


# ── MLP architecture ──────────────────────────────────────────────────────────

class MomentStatMLP(nn.Module):
    """3-layer MLP for combined MOMENT + stat feature input."""

    def __init__(self, in_dim: int = 5512, hidden: int = 1024,
                 n_classes: int = 8, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden // 2, hidden // 4),
            nn.BatchNorm1d(hidden // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(hidden // 4, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_moment_embeddings(split: str) -> np.ndarray:
    path = EMB_DIR / f"{split}_embeddings.npz"
    print(f"  MOMENT {split}: {path.name}", flush=True)
    d = np.load(path)
    emb = d[list(d.keys())[0]].astype(np.float32)   # (N, 4, 1024)
    return emb.reshape(emb.shape[0], -1)              # (N, 4096)


def load_stat_features(split: str) -> np.ndarray:
    parts = []
    for pos in POSITIONS:
        d = np.load(FEAT_DIR / f"{split}_{pos}.npz")
        parts.append(d["X"].astype(np.float32))       # (N, 354)
    return np.concatenate(parts, axis=1)              # (N, 1416)


def load_labels(split: str) -> np.ndarray:
    return np.load(FEAT_DIR / f"{split}_Bag.npz")["y"].astype(np.int64) - 1  # 0-based


def extract_test_stat_features() -> np.ndarray:
    import h5py
    from featureflyers_shl.features.statistical import extract_batch
    print("  Extracting test stat features from HDF5 …", flush=True)
    t0 = time.time()
    with h5py.File(HDF5_PATH, "r") as hf:
        raw = hf["test"]["data"][:].astype(np.float32)
    n_win = raw.shape[0] // WIN_SIZE
    windows = raw[: n_win * WIN_SIZE].reshape(n_win, WIN_SIZE, 9)
    del raw
    batch = 4096
    feats = []
    for i in range(0, n_win, batch):
        feats.append(extract_batch(windows[i: i + batch], FFT_TOP_K))
        if (i // batch) % 4 == 0:
            print(f"    stat {i:,}/{n_win:,}", flush=True)
    single = np.vstack(feats).astype(np.float32)      # (92726, 354)
    del windows
    combined = np.tile(single, 4)                     # (92726, 1416) — repeat ×4
    print(f"  Test stat: {combined.shape}  ({time.time()-t0:.0f}s)", flush=True)
    return combined


def build_features(split: str):
    emb = load_moment_embeddings(split)
    if split == "test":
        stat = extract_test_stat_features()
        y = None
    else:
        stat = load_stat_features(split)
        y = load_labels(split)
    X = np.concatenate([emb, stat], axis=1)
    print(f"  {split}: X={X.shape}  labels={'none' if y is None else y.shape}", flush=True)
    return X, y


# ── Normalisation (fit on train, apply to val/test) ───────────────────────────

def fit_normaliser(X_tr: np.ndarray):
    mean = X_tr.mean(axis=0, keepdims=True).astype(np.float32)
    std  = X_tr.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
    return mean, std


def normalise(X: np.ndarray, mean, std) -> np.ndarray:
    return (X - mean) / std


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, loader, optimiser, criterion, device):
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimiser.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, X_t: torch.Tensor, y_np: np.ndarray, device, batch=4096):
    model.eval()
    preds = []
    for i in range(0, len(X_t), batch):
        preds.append(model(X_t[i: i + batch].to(device)).argmax(1).cpu())
    y_pred = torch.cat(preds).numpy()
    return f1_score(y_np, y_pred, average="macro"), y_pred


@torch.no_grad()
def predict_probs(model, X_t: torch.Tensor, device, batch=4096) -> np.ndarray:
    model.eval()
    probs = []
    sm = nn.Softmax(dim=1)
    for i in range(0, len(X_t), batch):
        probs.append(sm(model(X_t[i: i + batch].to(device))).cpu().numpy())
    return np.vstack(probs)


# ── Submission writer ─────────────────────────────────────────────────────────

def write_submission(labels_1based: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting submission → {path.name} …", flush=True)
    with open(path, "w") as fh:
        for lbl in labels_1based:
            fh.write(",".join([str(lbl)] * WIN_SIZE) + "\n")
    print(f"  {path}  ({path.stat().st_size/1e6:.1f} MB)", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs",   type=int,   default=60)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden",   type=int,   default=1024)
    parser.add_argument("--dropout",  type=float, default=0.3)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--device",   type=str,   default="cuda:0")
    parser.add_argument("--no-test",  action="store_true")
    parser.add_argument("--out-dir",  type=Path,  default=OUT_DIR)
    parser.add_argument("--output",   type=Path,
                        default=SUB_DIR / "FeatureFlyers_moment_stat_mlp_s23.txt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage 23 — MOMENT (frozen) + Stat features → MLP")
    print(f"  Input dim: 4096 + 1416 = 5512")
    print(f"  Hidden: {args.hidden}  Dropout: {args.dropout}")
    print(f"  Epochs: {args.epochs}  LR: {args.lr}  BS: {args.batch_size}")
    print(f"  Device: {device}")
    print("=" * 60)

    # ── Load features ────────────────────────────────────────────────────────
    print("\n[1/4] Loading train features …")
    X_tr, y_tr = build_features("train")

    print("\n[2/4] Loading validation features …")
    X_val, y_val = build_features("validation")

    # Normalise (z-score, fit on train only)
    print("\nFitting normaliser on train …", flush=True)
    mean, std = fit_normaliser(X_tr)
    np.save(args.out_dir / "norm_mean.npy", mean)
    np.save(args.out_dir / "norm_std.npy",  std)
    X_tr_n  = normalise(X_tr,  mean, std)
    X_val_n = normalise(X_val, mean, std)

    # Tensors
    T_tr  = torch.from_numpy(X_tr_n)
    T_val = torch.from_numpy(X_val_n)
    y_tr_t = torch.from_numpy(y_tr)

    train_ds = TensorDataset(T_tr, y_tr_t)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=4, pin_memory=True)

    # ── Build model ──────────────────────────────────────────────────────────
    in_dim = X_tr_n.shape[1]
    model = MomentStatMLP(in_dim=in_dim, hidden=args.hidden,
                          n_classes=N_CLASSES, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params: {n_params:,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\n[3/4] Training for {args.epochs} epochs …\n")
    best_f1, best_ep = 0.0, 0
    t0_total = time.time()

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train(model, train_dl, optimiser, criterion, device)
        val_f1, _ = evaluate(model, T_val, y_val, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"  ep {ep:03d}/{args.epochs}  loss={tr_loss:.4f}  "
              f"val_F1={val_f1:.4f}  ({elapsed:.0f}s)", flush=True)

        if val_f1 > best_f1:
            best_f1, best_ep = val_f1, ep
            torch.save(model.state_dict(), args.out_dir / "best_model.pt")

    total_time = time.time() - t0_total
    print(f"\nTraining complete: {total_time:.0f}s total")
    print(f"Best val Macro-F1: {best_f1:.4f}  (epoch {best_ep})")

    # Load best checkpoint and report per-class
    model.load_state_dict(torch.load(args.out_dir / "best_model.pt", map_location=device))
    _, y_pred = evaluate(model, T_val, y_val, device)
    labels = ["Still", "Walking", "Run", "Bike", "Car", "Bus", "Train", "Metro"]
    print("\n" + classification_report(y_val, y_pred, target_names=labels, digits=4))

    # Save metrics
    metrics = {"best_val_macro_f1": round(best_f1, 6), "best_epoch": best_ep,
               "epochs": args.epochs, "lr": args.lr, "hidden": args.hidden,
               "in_dim": in_dim, "seed": args.seed}
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ── Test prediction ───────────────────────────────────────────────────────
    if not args.no_test:
        print("\n[4/4] Loading test features and predicting …")
        X_test, _ = build_features("test")
        X_test_n  = normalise(X_test, mean, std)
        T_test    = torch.from_numpy(X_test_n)

        probs = predict_probs(model, T_test, device)
        np.save(args.out_dir / "test_probs.npy", probs)

        y_test = probs.argmax(axis=1).astype(np.int64) + 1   # 1-based
        write_submission(y_test, args.output)

    print("\n=== Stage 23 complete ===")
    print(f"  Best val Macro-F1: {best_f1:.4f}  (epoch {best_ep})")
    if not args.no_test:
        print(f"  Submission: {args.output}")


if __name__ == "__main__":
    main()
