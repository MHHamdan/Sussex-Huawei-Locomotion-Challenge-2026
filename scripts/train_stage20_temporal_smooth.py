#!/usr/bin/env python3
"""
Stage 20 — Temporal Sequence Smoothing for SHL 2026.

Phase A (--mode hmm):
    HMM/Viterbi post-processing on Stage 16 ensemble softmax probs.
    Builds an 8×8 transition matrix from training label sequences and
    applies log-space Viterbi decoding. Zero extra GPU memory needed.

Phase B (--mode bilstm):
    Bidirectional LSTM trained on PCA-compressed MOMENT-1-large embeddings
    (extracted in Stage 19). Learns temporal context over ~500s windows.
    Saves val_probs.npy + test_probs.npy as an 8th model for Stage 16 blend.

Phase C (--mode both):
    Runs both phases, saves combined (averaged) probs.

Prerequisite: re-run Stage 16 once with --predict-test to populate
  outputs/execution-output/meta_blend_s16_lgbm_7models/val_probs.npy
  outputs/execution-output/meta_blend_s16_lgbm_7models/test_probs.npy

Usage
-----
# Phase A — HMM only
python scripts/train_stage20_temporal_smooth.py \\
  --mode hmm \\
  --val-probs  outputs/execution-output/meta_blend_s16_lgbm_7models/val_probs.npy \\
  --test-probs outputs/execution-output/meta_blend_s16_lgbm_7models/test_probs.npy \\
  --output-dir outputs/execution-output/stage20_hmm

# Phase B — BiLSTM only
python scripts/train_stage20_temporal_smooth.py \\
  --mode bilstm \\
  --embeddings-dir outputs/execution-output/moment_embeddings \\
  --output-dir outputs/execution-output/stage20_bilstm \\
  --device cpu --epochs 30

# Phase C — combined (HMM + BiLSTM averaged)
python scripts/train_stage20_temporal_smooth.py \\
  --mode both \\
  --val-probs  outputs/execution-output/meta_blend_s16_lgbm_7models/val_probs.npy \\
  --test-probs outputs/execution-output/meta_blend_s16_lgbm_7models/test_probs.npy \\
  --embeddings-dir outputs/execution-output/moment_embeddings \\
  --output-dir outputs/execution-output/stage20_combined \\
  --device cpu --epochs 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).parent.parent
HDF5_PATH   = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
OUT_DIR     = REPO_ROOT / "outputs" / "execution-output"
N_CLASSES   = 8
LABEL_NAMES = ["Still", "Walking", "Run", "Bike", "Car", "Bus", "Train", "Metro"]
WINDOW_HOP  = 250  # samples; must match Stage 19 extraction


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 20 — Temporal Sequence Smoothing")
    p.add_argument("--mode", choices=["hmm", "bilstm", "both"], default="both")

    # HMM inputs
    p.add_argument("--val-probs",  type=Path, default=None,
                   help="(N_val, 8) ensemble val probs from Stage 16")
    p.add_argument("--test-probs", type=Path, default=None,
                   help="(N_test, 8) ensemble test probs from Stage 16")
    p.add_argument("--hmm-laplace", type=float, default=0.5,
                   help="Laplace smoothing for transition matrix")

    # BiLSTM inputs
    p.add_argument("--embeddings-dir", type=Path,
                   default=OUT_DIR / "moment_embeddings",
                   help="Dir with train/val/test_embeddings.npz from Stage 19")
    p.add_argument("--pca-dim",   type=int, default=128)
    p.add_argument("--seq-len",   type=int, default=200,
                   help="Sequence length in windows (~500s context)")
    p.add_argument("--seq-stride", type=int, default=100,
                   help="Stride for training sequences")
    p.add_argument("--hidden",    type=int, default=128)
    p.add_argument("--n-layers",  type=int, default=2)
    p.add_argument("--dropout",   type=float, default=0.3)
    p.add_argument("--epochs",    type=int, default=30)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device",    type=str, default="cpu")
    p.add_argument("--label-smoothing", type=float, default=0.05)

    # General
    p.add_argument("--output-dir", type=Path,
                   default=OUT_DIR / "stage20_temporal_smooth")
    p.add_argument("--hdf5-path", type=Path, default=HDF5_PATH)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def get_window_labels(hdf5_path: Path, split: str) -> np.ndarray:
    """Extract per-window labels matching SHLWindowDataset's centre-sample rule exactly."""
    win_size, hop = 500, WINDOW_HOP
    with h5py.File(hdf5_path, "r") as f:
        flat = f[f"{split}/Bag/labels"][:]       # (N_samples,) int8, labels 1-8
    N = len(flat)
    # Mirror: np.arange(WIN_SIZE//2, N - WIN_SIZE//2, HOP_SIZE) in SHLWindowDataset
    centres = np.arange(win_size // 2, N - win_size // 2, hop, dtype=np.int64)
    labels = flat[centres].astype(np.int64) - 1  # → 0-based
    return labels


def load_moment_features(emb_dir: Path, split: str,
                          pca: PCA | None = None,
                          pca_dim: int = 128) -> tuple[np.ndarray, PCA]:
    """
    Load (N, 4, 1024) float16 MOMENT embeddings, mean-pool 4 positions → (N, 1024),
    then PCA → (N, pca_dim).  If pca is None, fits on this split (use for train).
    """
    path = emb_dir / f"{split}_embeddings.npz"
    print(f"  Loading {split} embeddings from {path.name} …", flush=True)
    d = np.load(path)
    embs = d["embeddings"].astype(np.float32)      # (N, 4, 1024)
    embs_mean = embs.mean(axis=1)                  # (N, 1024)
    del embs

    if pca is None:
        print(f"  Fitting PCA({pca_dim}) on {embs_mean.shape} …", flush=True)
        pca = PCA(n_components=pca_dim, random_state=42)
        pca.fit(embs_mean)
        var = pca.explained_variance_ratio_.sum()
        print(f"  PCA explained variance: {var:.3f}")

    features = pca.transform(embs_mean).astype(np.float32)  # (N, pca_dim)
    print(f"  {split}: features {features.shape}", flush=True)
    return features, pca


# ---------------------------------------------------------------------------
# Phase A — HMM / Viterbi
# ---------------------------------------------------------------------------

def build_transition_matrix(hdf5_path: Path, laplace: float = 0.5) -> np.ndarray:
    """
    Build 8×8 row-stochastic transition matrix from training label sequences
    sampled at window-hop stride.
    """
    print("Building transition matrix from train labels …", flush=True)
    labels = get_window_labels(hdf5_path, "train")
    A = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    for t in range(len(labels) - 1):
        A[labels[t], labels[t + 1]] += 1.0
    A += laplace
    A /= A.sum(axis=1, keepdims=True)
    print(f"  Transition matrix built from {len(labels):,} train windows.")
    return A.astype(np.float32)


def viterbi_decode(log_emission: np.ndarray, log_trans: np.ndarray,
                   log_prior: np.ndarray) -> np.ndarray:
    """
    Log-space Viterbi decoding.

    Args:
        log_emission : (T, K) — log softmax probs (emission model)
        log_trans    : (K, K) — log transition matrix
        log_prior    : (K,)   — log initial state distribution
    Returns:
        path : (T,) — MAP label sequence
    """
    T, K = log_emission.shape
    viterbi = np.full((T, K), -np.inf, dtype=np.float64)
    backptr = np.zeros((T, K), dtype=np.int32)

    viterbi[0] = log_prior + log_emission[0]
    for t in range(1, T):
        trans_scores = viterbi[t - 1, :, None] + log_trans   # (K, K)
        backptr[t]   = trans_scores.argmax(axis=0)
        viterbi[t]   = trans_scores.max(axis=0) + log_emission[t]

    # Backtrack
    path = np.empty(T, dtype=np.int32)
    path[-1] = viterbi[-1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
    return path


def hmm_smooth(probs: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """
    Apply Viterbi smoothing to (N, 8) softmax probs.
    Returns (N, 8) one-hot probability matrix (hard assignment).
    """
    eps = 1e-9
    log_emission = np.log(probs + eps)
    log_trans    = np.log(trans + eps)
    prior        = probs.mean(axis=0)
    log_prior    = np.log(prior + eps)

    path = viterbi_decode(log_emission, log_trans, log_prior)

    # Return one-hot — downstream blend can use these as hard probs
    smoothed = np.zeros_like(probs)
    smoothed[np.arange(len(path)), path] = 1.0
    return smoothed


def run_hmm(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.val_probs is None or args.test_probs is None:
        raise ValueError("--val-probs and --test-probs required for HMM mode. "
                         "Re-run Stage 16 with --predict-test to generate them.")

    print("\n" + "=" * 60)
    print("PHASE A — HMM / Viterbi Smoothing")
    print("=" * 60)

    trans = build_transition_matrix(args.hdf5_path, laplace=args.hmm_laplace)

    print(f"\nLoading val probs  : {args.val_probs}")
    val_probs  = np.load(args.val_probs).astype(np.float32)
    print(f"Loading test probs : {args.test_probs}")
    test_probs = np.load(args.test_probs).astype(np.float32)

    # Evaluate ensemble before smoothing
    val_labels = get_window_labels(args.hdf5_path, "validation")
    pre_f1 = f1_score(val_labels, val_probs.argmax(1), average="macro")
    print(f"\nEnsemble val macro-F1 (pre-HMM)  : {pre_f1:.4f}")

    t0 = time.time()
    val_smooth  = hmm_smooth(val_probs,  trans)
    test_smooth = hmm_smooth(test_probs, trans)
    elapsed = time.time() - t0

    post_f1 = f1_score(val_labels, val_smooth.argmax(1), average="macro")
    print(f"HMM val macro-F1  (post-Viterbi) : {post_f1:.4f}  ({elapsed:.1f}s)")
    print(f"Delta: {post_f1 - pre_f1:+.4f}")

    _print_per_class(val_labels, val_smooth.argmax(1))

    return {"val": val_smooth, "test": test_smooth, "val_labels": val_labels,
            "val_f1": post_f1}


# ---------------------------------------------------------------------------
# Phase B — BiLSTM Sequence Classifier
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """Sliding-window sequences of (T, feat_dim) features with (T,) labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 seq_len: int, stride: int) -> None:
        self.features = torch.from_numpy(features)   # (N, D)
        self.labels   = torch.from_numpy(labels.astype(np.int64))
        self.seq_len  = seq_len
        self.stride   = stride
        N = len(features)
        self.starts = list(range(0, max(1, N - seq_len + 1), stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = self.starts[idx]
        e = min(s + self.seq_len, len(self.features))
        x = self.features[s:e]
        y = self.labels[s:e]
        # Pad if shorter than seq_len
        pad = self.seq_len - len(x)
        if pad > 0:
            x = torch.cat([x, torch.zeros(pad, x.shape[1])], dim=0)
            y = torch.cat([y, torch.full((pad,), -1, dtype=torch.long)], dim=0)
        return x, y, torch.tensor(e - s, dtype=torch.long)   # valid length


class TemporalLSTM(nn.Module):
    """
    Bidirectional LSTM sequence classifier.
    Input : (B, T, feat_dim)
    Output: (B, T, N_CLASSES) logits
    """

    def __init__(self, feat_dim: int, hidden: int = 128,
                 n_layers: int = 2, dropout: float = 0.3,
                 n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                    # (B, T, hidden)
        x, _ = self.lstm(x)                 # (B, T, hidden*2)
        return self.head(x)                 # (B, T, n_classes)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_bilstm(model: TemporalLSTM, train_ds: SequenceDataset,
                 val_features: np.ndarray, val_labels: np.ndarray,
                 args: argparse.Namespace, device: torch.device) -> TemporalLSTM:
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-1,
                                    label_smoothing=args.label_smoothing)

    best_f1, best_state = 0.0, None
    print(f"\n{'Ep':>4}  {'TrLoss':>8}  {'TrAcc':>8}  {'ValF1':>7}  {'LR':>9}  {'Time':>6}")
    print("-" * 55)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_correct, total_valid = 0.0, 0, 0

        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                           # (B, T, K)
            B, T, K = logits.shape
            loss = criterion(logits.reshape(B * T, K), y.reshape(B * T))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            mask = y.reshape(-1) != -1
            preds = logits.reshape(B * T, K).argmax(1)
            total_correct += (preds[mask] == y.reshape(-1)[mask]).sum().item()
            total_valid   += mask.sum().item()
            total_loss    += loss.item() * B

        scheduler.step()
        tr_acc  = total_correct / max(total_valid, 1)
        tr_loss = total_loss / len(train_ds)
        val_f1  = _eval_bilstm(model, val_features, val_labels, args, device)

        if val_f1 > best_f1:
            best_f1   = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        lr = scheduler.get_last_lr()[0]
        print(f"{ep:>4}  {tr_loss:>8.4f}  {tr_acc:>7.3%}  {val_f1:>7.4f}  "
              f"{lr:>9.2e}  {time.time()-t0:>5.1f}s")

    print(f"\nBest val macro-F1 = {best_f1:.4f}")
    model.load_state_dict(best_state)
    return model


def _eval_bilstm(model: TemporalLSTM, features: np.ndarray, labels: np.ndarray,
                 args: argparse.Namespace, device: torch.device) -> float:
    probs = predict_bilstm(model, features, args.seq_len, device)
    return f1_score(labels, probs.argmax(1), average="macro")


def predict_bilstm(model: TemporalLSTM, features: np.ndarray,
                   seq_len: int, device: torch.device) -> np.ndarray:
    """
    Non-overlapping inference: tile features into chunks of seq_len,
    flatten predictions back to (N, 8) probs.
    """
    model.eval()
    N, D = features.shape
    all_logits = np.zeros((N, N_CLASSES), dtype=np.float32)
    feat_t = torch.from_numpy(features)

    with torch.no_grad():
        for start in range(0, N, seq_len):
            end   = min(start + seq_len, N)
            chunk = feat_t[start:end].unsqueeze(0).to(device)  # (1, T, D)
            pad   = seq_len - chunk.shape[1]
            if pad > 0:
                chunk = torch.cat([chunk,
                                   torch.zeros(1, pad, D, device=device)], dim=1)
            logits = model(chunk)[0, :end - start].cpu().numpy()  # (T, K)
            all_logits[start:end] = logits

    probs = _softmax(all_logits)
    return probs


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def run_bilstm(args: argparse.Namespace,
               val_labels: np.ndarray | None = None) -> dict[str, np.ndarray]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("\n" + "=" * 60)
    print("PHASE B — BiLSTM Sequence Classifier")
    print("=" * 60)

    # Load + compress MOMENT embeddings
    print("\nLoading MOMENT embeddings …")
    train_feat, pca = load_moment_features(args.embeddings_dir, "train",
                                           pca_dim=args.pca_dim)
    val_feat,   _   = load_moment_features(args.embeddings_dir, "validation", pca)
    test_feat,  _   = load_moment_features(args.embeddings_dir, "test", pca)

    # Window labels
    train_labels = get_window_labels(args.hdf5_path, "train")
    if val_labels is None:
        val_labels = get_window_labels(args.hdf5_path, "validation")

    # Datasets
    train_ds = SequenceDataset(train_feat, train_labels, args.seq_len, args.seq_stride)
    print(f"\nTrain sequences : {len(train_ds):,}  "
          f"(len={args.seq_len}, stride={args.seq_stride})")
    print(f"Val windows     : {len(val_feat):,}")
    print(f"Test windows    : {len(test_feat):,}")

    # Model
    model = TemporalLSTM(feat_dim=args.pca_dim, hidden=args.hidden,
                         n_layers=args.n_layers, dropout=args.dropout).to(device)
    print(f"\nTemporalLSTM  n_params={model.n_params:,}  device={device}")

    # Train
    model = train_bilstm(model, train_ds, val_feat, val_labels, args, device)

    # Inference
    print("\nRunning val inference …")
    val_probs  = predict_bilstm(model, val_feat,  args.seq_len, device)
    print("Running test inference …")
    test_probs = predict_bilstm(model, test_feat, args.seq_len, device)

    val_f1 = f1_score(val_labels, val_probs.argmax(1), average="macro")
    print(f"\nBiLSTM val macro-F1 : {val_f1:.4f}")
    _print_per_class(val_labels, val_probs.argmax(1))

    return {"val": val_probs, "test": test_probs, "val_labels": val_labels,
            "val_f1": val_f1, "model": model, "pca": pca}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_per_class(labels: np.ndarray, preds: np.ndarray) -> None:
    f1s = f1_score(labels, preds, average=None, labels=list(range(N_CLASSES)))
    print("\nPer-class F1:")
    for i, (name, f) in enumerate(zip(LABEL_NAMES, f1s)):
        print(f"  {name:<10}: {f:.4f}")


def _save_outputs(out_dir: Path, val_probs: np.ndarray, test_probs: np.ndarray,
                  val_labels: np.ndarray, val_f1: float, tag: str,
                  model: TemporalLSTM | None = None,
                  pca: PCA | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "val_probs.npy",  val_probs)
    np.save(out_dir / "test_probs.npy", test_probs)
    np.save(out_dir / "val_labels.npy", val_labels)

    cfg = {"tag": tag, "val_macro_f1": round(float(val_f1), 4),
           "val_shape": list(val_probs.shape),
           "test_shape": list(test_probs.shape)}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    if model is not None:
        torch.save(model.state_dict(), out_dir / "model.pt")
    if pca is not None:
        import joblib
        joblib.dump(pca, out_dir / "pca.joblib")

    display = out_dir.resolve().relative_to(REPO_ROOT) if out_dir.resolve().is_relative_to(REPO_ROOT) else out_dir
    print(f"\nArtifacts saved → {display}/")
    print(f"  val_probs.npy  : {val_probs.shape}")
    print(f"  test_probs.npy : {test_probs.shape}")
    print(f"  val macro-F1   : {val_f1:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t_start = time.time()

    print(f"Stage 20 — Temporal Sequence Smoothing  [mode={args.mode}]")
    print(f"  output-dir : {args.output_dir}")

    if args.mode == "hmm":
        result = run_hmm(args)
        _save_outputs(
            args.output_dir,
            result["val"], result["test"], result["val_labels"],
            result["val_f1"], tag="hmm",
        )

    elif args.mode == "bilstm":
        result = run_bilstm(args)
        _save_outputs(
            args.output_dir,
            result["val"], result["test"], result["val_labels"],
            result["val_f1"], tag="bilstm",
            model=result["model"], pca=result["pca"],
        )

    elif args.mode == "both":
        hmm_result    = run_hmm(args)
        bilstm_result = run_bilstm(args, val_labels=hmm_result["val_labels"])

        # Average ensemble probs (soft combination)
        val_combined  = 0.5 * hmm_result["val"]  + 0.5 * bilstm_result["val"]
        test_combined = 0.5 * hmm_result["test"] + 0.5 * bilstm_result["test"]
        combined_f1   = f1_score(hmm_result["val_labels"],
                                 val_combined.argmax(1), average="macro")
        print(f"\n{'='*60}")
        print(f"COMBINED val macro-F1: {combined_f1:.4f}  "
              f"(HMM={hmm_result['val_f1']:.4f}  "
              f"BiLSTM={bilstm_result['val_f1']:.4f})")
        _print_per_class(hmm_result["val_labels"], val_combined.argmax(1))

        # Save all three outputs
        _save_outputs(args.output_dir / "hmm", hmm_result["val"],
                      hmm_result["test"], hmm_result["val_labels"],
                      hmm_result["val_f1"], tag="hmm")
        _save_outputs(args.output_dir / "bilstm", bilstm_result["val"],
                      bilstm_result["test"], bilstm_result["val_labels"],
                      bilstm_result["val_f1"], tag="bilstm",
                      model=bilstm_result["model"], pca=bilstm_result["pca"])
        _save_outputs(args.output_dir / "combined", val_combined,
                      test_combined, hmm_result["val_labels"],
                      combined_f1, tag="combined")

    print(f"\nTotal elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
