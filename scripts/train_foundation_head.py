#!/usr/bin/env python3
"""
Train a lightweight MLP head on frozen foundation-model embeddings.

Encoder options:
  moment   -- MOMENT-1-large (341M frozen params, 1024-dim embeddings).
               Downloads ~1.4 GB on first run from HuggingFace.
  fallback -- Frozen random projection (NOT a real foundation model).
               Fast, no download, reproducible placeholder.

Embeddings are cached under dataset/processed/embeddings/ after the first
extraction so subsequent runs skip the slow foundation-model forward pass.

Head architecture: Linear(embed_dim, 256) -> ReLU -> Dropout -> Linear(256, 8)

Baseline to beat: XGBoost pool, macro-F1=0.6389, accuracy=68.4% (Bag-only val)

Usage
-----
# Smoke test -- fallback encoder, 5000 windows, 2 epochs
python scripts/train_foundation_head.py \\
    --position Bag --sample-limit 5000 --encoder fallback \\
    --epochs 2 --batch-size 128 --device cuda

# Smoke test -- MOMENT encoder (downloads model on first run)
python scripts/train_foundation_head.py \\
    --position Bag --sample-limit 5000 --encoder moment \\
    --epochs 2 --batch-size 128 --device cuda

# Full Bag-position run with MOMENT
python scripts/train_foundation_head.py \\
    --position Bag --encoder moment \\
    --epochs 30 --batch-size 512 --device cuda
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
EMBED_DIR    = REPO_ROOT / "dataset" / "processed" / "embeddings"
DEFAULT_OUTD = REPO_ROOT / "outputs" / "execution-output"

LABEL_MAP = {0: "Still", 1: "Walking", 2: "Run", 3: "Bike",
             4: "Car",   5: "Bus",     6: "Train", 7: "Metro"}
N_CLASSES = 8
XGB_POOL_BASELINE_F1 = 0.6389


def _extract_embeddings(encoder, dataset, batch_size: int, device) -> tuple:
    """
    Forward all dataset windows through the frozen encoder.

    Returns (X_emb, y):
        X_emb -- (N, embed_dim) float32
        y     -- (N,) int64  0-based labels
    """
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))
    encoder.to(device)
    encoder.eval()

    emb_list, lbl_list = [], []
    total = len(dataset)
    t0 = time.time()

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x   = x.to(device)
            emb = encoder(x).float()  # cast back to FP32 before numpy
            emb_list.append(emb.cpu().numpy())
            lbl_list.append(np.asarray(y))
            done = min((i + 1) * batch_size, total)
            if done % (batch_size * 20) == 0 or done >= total:
                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed > 0 else 0
                eta     = (total - done) / rate if rate > 0 else 0
                print(f"    {done:,}/{total:,}  ({done / total * 100:.0f}%)"
                      f"  {elapsed:.0f}s  eta {eta/60:.0f}m", flush=True)

    X = np.vstack(emb_list).astype(np.float32)
    y = np.concatenate(lbl_list).astype(np.int64)
    print(f"  Done: {time.time() - t0:.1f}s  X={X.shape}")
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hdf5-path",     type=Path, default=HDF5_PATH)
    parser.add_argument("--position",      default="Bag",
                        choices=["Bag", "Hand", "Hips", "Torso"])
    parser.add_argument("--sample-limit",  type=int, default=None,
                        help="Stratified window limit per split (None = full dataset)")
    parser.add_argument("--encoder",       default="moment",
                        choices=["moment", "fallback"])
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--batch-size",      type=int, default=512,
                        help="Head training batch size")
    parser.add_argument("--extract-batch-size", type=int, default=64,
                        help="Batch size for frozen encoder forward pass (keep low to avoid OOM)")
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--device",        default="cuda",
                        help="cpu | cuda | cuda:0 (default: cuda)")
    parser.add_argument("--patience",      type=int,   default=10,
                        help="Early-stopping patience in epochs (0 = disabled)")
    parser.add_argument("--output-dir",    type=Path, default=DEFAULT_OUTD)
    parser.add_argument("--force-extract", action="store_true",
                        help="Re-extract embeddings even if cache exists")
    args = parser.parse_args()

    # ---- imports ----
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("ERROR: torch not installed"); sys.exit(1)
    try:
        from sklearn.metrics import classification_report, f1_score
    except ImportError:
        print("ERROR: scikit-learn not installed"); sys.exit(1)

    from featureflyers_shl.data.dataset   import SHLWindowDataset
    from featureflyers_shl.models.foundation import get_encoder

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

    if not args.hdf5_path.exists():
        print(f"ERROR: HDF5 not found: {args.hdf5_path}"); sys.exit(1)

    lim_str  = str(args.sample_limit) if args.sample_limit else "full"
    run_name = (f"foundation_{args.encoder}_pos{args.position}"
                f"_s{lim_str}_ep{args.epochs}")
    run_dir  = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFoundation head")
    print(f"  encoder      : {args.encoder}")
    print(f"  position     : {args.position}")
    print(f"  sample_limit : {lim_str}")
    print(f"  device       : {device}")
    print(f"  epochs       : {args.epochs}")
    print(f"  output       : {run_dir.relative_to(REPO_ROOT)}\n")

    # ---- load encoder ----
    print(f"Loading {args.encoder} encoder ...", flush=True)
    t0 = time.time()
    encoder = get_encoder(args.encoder)
    encoder.eval()
    embed_dim      = encoder.embed_dim
    is_placeholder = getattr(encoder, "is_placeholder", False)
    if is_placeholder:
        print(f"  [WARNING] fallback encoder is NOT a real foundation model.")
        print(f"  Results are not comparable to MOMENT.")
    print(f"  embed_dim={embed_dim}  loaded in {time.time() - t0:.1f}s\n")

    # ---- datasets ----
    print("Building datasets (preloading into RAM) ...", flush=True)
    t_ds = time.time()
    # Force preload: 392K windows = ~7 GB RAM (tiny vs 152 GiB available).
    # Without preload, each __getitem__ does a random HDF5 seek → CPU bottleneck.
    ds_train = SHLWindowDataset(
        args.hdf5_path, split="train",
        position=args.position, sample_limit=args.sample_limit, seed=args.seed,
        preload=True,
    )
    ds_val = SHLWindowDataset(
        args.hdf5_path, split="validation",
        position=args.position, sample_limit=args.sample_limit, seed=args.seed + 1,
        preload=True,
    )
    print(f"  Train : {len(ds_train):,} windows  |  Val : {len(ds_val):,} windows"
          f"  ({time.time() - t_ds:.0f}s preload)\n")

    # ---- embedding extraction (cached) ----
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(split: str) -> Path:
        lim = f"_s{args.sample_limit}" if args.sample_limit else ""
        return EMBED_DIR / f"{split}_{args.position}_{args.encoder}{lim}.npz"

    def _get_embeddings(split: str, ds) -> tuple:
        cache = _cache_path(split)
        if cache.exists() and not args.force_extract:
            print(f"  [CACHE] {split} embeddings from {cache.name}", flush=True)
            d = np.load(cache)
            print(f"    X={d['X'].shape}")
            return d["X"], d["y"]
        print(f"  [EXTRACT] {split} ({len(ds):,} windows) ...", flush=True)
        X, y = _extract_embeddings(encoder, ds, args.extract_batch_size, device)
        np.savez_compressed(cache, X=X, y=y)
        print(f"  Saved {cache.relative_to(REPO_ROOT)}")
        return X, y

    print("Extracting / loading embeddings ...")
    X_train, y_train = _get_embeddings("train", ds_train)
    X_val,   y_val   = _get_embeddings("validation", ds_val)
    print()

    # ---- class weights ----
    counts  = np.bincount(y_train, minlength=N_CLASSES).astype(np.float32)
    weights = torch.tensor(
        1.0 / np.where(counts > 0, counts, 1.0), dtype=torch.float32, device=device
    )
    weights = weights / weights.sum() * N_CLASSES

    # ---- data loaders over cached embeddings ----
    train_lds = TensorDataset(torch.from_numpy(X_train),
                              torch.from_numpy(y_train).long())
    val_lds   = TensorDataset(torch.from_numpy(X_val),
                              torch.from_numpy(y_val).long())
    lkw = dict(batch_size=args.batch_size, num_workers=0,
               pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_lds, shuffle=True,  **lkw)
    val_loader   = DataLoader(val_lds,   shuffle=False, **lkw)

    # ---- head: embed_dim -> 256 -> N_CLASSES ----
    head = nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(256, N_CLASSES),
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"Head: {embed_dim} -> 256 -> {N_CLASSES}  ({n_params:,} params)\n")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ---- training loop ----
    print(f"{'Ep':>4}  {'Loss':>8}  {'TrainAcc':>8}  "
          f"{'ValAcc':>7}  {'MacroF1':>8}  {'Time':>5}")
    print("-" * 52)

    best_f1    = 0.0
    best_state = None
    no_improve = 0
    history    = []
    t_train    = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()

        head.train()
        tot_loss = tot_correct = tot_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tot_loss    += loss.item() * len(yb)
            tot_correct += (logits.argmax(1) == yb).sum().item()
            tot_n       += len(yb)

        tr_loss = tot_loss / tot_n
        tr_acc  = tot_correct / tot_n

        head.eval()
        preds_buf, lbls_buf = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds_buf.append(head(xb.to(device)).argmax(1).cpu().numpy())
                lbls_buf.append(yb.numpy())
        val_preds  = np.concatenate(preds_buf)
        val_labels = np.concatenate(lbls_buf)
        val_acc    = float((val_preds == val_labels).mean())
        val_f1     = float(f1_score(val_labels, val_preds,
                                    average="macro", zero_division=0))

        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()
        ep_time = time.time() - t_ep

        print(f"{epoch:>4}  {tr_loss:>8.4f}  {tr_acc:>7.3%}  "
              f"{val_acc:>7.3%}  {val_f1:>8.4f}  {ep_time:>4.1f}s")

        history.append(dict(
            epoch=epoch, train_loss=round(tr_loss, 5), train_acc=round(tr_acc, 4),
            val_acc=round(val_acc, 4), val_macro_f1=round(val_f1, 4),
            lr=round(lr_now, 8), epoch_time_s=round(ep_time, 1),
        ))

        if val_f1 >= best_f1:
            best_f1    = val_f1
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break

    total_time = time.time() - t_train
    print(f"\nTraining done in {total_time:.1f}s   best val macro-F1 = {best_f1:.4f}")

    # ---- final eval on best head ----
    head.load_state_dict(best_state)
    head.eval()
    preds_buf, lbls_buf = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds_buf.append(head(xb.to(device)).argmax(1).cpu().numpy())
            lbls_buf.append(yb.numpy())
    final_preds  = np.concatenate(preds_buf)
    final_labels = np.concatenate(lbls_buf)
    final_f1     = float(f1_score(final_labels, final_preds,
                                  average="macro", zero_division=0))
    final_acc    = float((final_preds == final_labels).mean())

    present = sorted(set(final_labels.tolist()))
    report  = classification_report(
        final_labels, final_preds,
        labels=present, target_names=[LABEL_MAP[l] for l in present],
        zero_division=0,
    )
    print(f"\n{'=' * 55}")
    print(f"Best head  Accuracy: {final_acc:.4f}   Macro-F1: {final_f1:.4f}")
    delta = final_f1 - XGB_POOL_BASELINE_F1
    sign  = "+" if delta >= 0 else ""
    print(f"vs XGBoost pool baseline (F1=0.6389):  {sign}{delta:.4f}")
    print(f"{'=' * 55}")
    print(report)

    # ---- artefacts ----
    torch.save(best_state, run_dir / "head.pt")
    config = dict(
        encoder=args.encoder, is_placeholder=is_placeholder,
        position=args.position, sample_limit=args.sample_limit,
        embed_dim=embed_dim, n_head_params=n_params,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, dropout=args.dropout, seed=args.seed,
        patience=args.patience, device=str(device),
        n_train=len(ds_train), n_val=len(ds_val),
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    metrics = dict(
        encoder=args.encoder, is_placeholder=is_placeholder,
        best_val_macro_f1=round(final_f1, 4),
        best_val_accuracy=round(final_acc, 4),
        xgb_pool_baseline_f1=XGB_POOL_BASELINE_F1,
        delta_vs_baseline=round(final_f1 - XGB_POOL_BASELINE_F1, 4),
        total_time_s=round(total_time, 1),
        epochs_trained=len(history),
        early_stopped=(args.patience > 0 and no_improve >= args.patience),
        history=history,
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "classification_report.txt").write_text(report)

    print(f"\nSaved to {run_dir.relative_to(REPO_ROOT)}/")
    print(f"  head.pt  config.json  metrics.json  classification_report.txt")


if __name__ == "__main__":
    main()
