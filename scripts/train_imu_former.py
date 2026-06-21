#!/usr/bin/env python3
"""
Stage 8 — Train IMUFormer end-to-end on raw SHL windows.

IMUFormer is a purpose-built architecture with three parallel multi-scale 1-D
CNN branches (one per sensor group: Acc/Gyr/Mag) fused via a lightweight
Transformer.  It trains from scratch on the raw (500, 9) windows — no
foundation model, no pre-extracted embeddings.

Advantages over Stage 6 (MOMENT + XGB):
  - End-to-end: the representation is learned for this specific task
  - Inductive bias: Acc/Gyr/Mag branches match IMU physics
  - Lightweight: ~2.8M params vs 341M MOMENT
  - Multi-scale: captures 50 ms → 650 ms temporal patterns simultaneously
  - Temporal context inside each 5-second window via Transformer

Expected result
---------------
  Quick run (stratified 40k/class, ~20 min):  ~0.70–0.73 macro-F1
  Full pool run  (1.57 M windows,  ~1.5 h):   ~0.73–0.78 macro-F1

Usage
-----
# Quick validation run (stratified 40k/class from pool, ~20 min)
python scripts/train_imu_former.py \\
    --fusion pool --stratify-per-class 40000 \\
    --epochs 50 --batch-size 512 --device cuda

# Full pool training (~1.5 h)
python scripts/train_imu_former.py \\
    --fusion pool --epochs 60 --batch-size 512 --device cuda

# Single-position smoke test (Bag, 5000 windows, 5 epochs)
python scripts/train_imu_former.py \\
    --position Bag --sample-limit 5000 --epochs 5 --device cuda

# Generate submission from saved checkpoint
python scripts/train_imu_former.py \\
    --predict-test \\
    --model-path outputs/execution-output/<run>/best_model.pt \\
    --output outputs/execution-output/submissions/stage8_submission.txt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT   = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HDF5_PATH    = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
DEFAULT_OUTD = REPO_ROOT / "outputs" / "execution-output"

LABEL_MAP      = {0:"Still", 1:"Walking", 2:"Run", 3:"Bike",
                  4:"Car",   5:"Bus",     6:"Train", 7:"Metro"}
N_CLASSES      = 8
LABEL_OFFSET   = 1
POSITIONS_ALL  = ["Bag", "Hand", "Hips", "Torso"]
WIN_SIZE       = 500


# ---------------------------------------------------------------------------
# Preprocessing (same as train_foundation_head for consistency)
# ---------------------------------------------------------------------------

def _per_window_norm(X: np.ndarray) -> np.ndarray:
    """Z-score each window independently. X: (N, T, C) → (N, T, C)."""
    X = X.astype(np.float32, copy=True)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return ((X - mu) / sd).astype(np.float32)


# ---------------------------------------------------------------------------
# Class-balanced stratified resampling
# ---------------------------------------------------------------------------

def _stratify_balanced(
    X: np.ndarray,
    y: np.ndarray,
    n_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for cls in range(N_CLASSES):
        idx = np.where(y == cls)[0]
        if len(idx) > n_per_class:
            idx = rng.choice(idx, n_per_class, replace=False)
        keep.append(idx)
    perm = np.concatenate(keep)
    rng.shuffle(perm)
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_split(
    hdf5_path: Path,
    split: str,
    positions: list[str],
    sample_limit: int | None,
    seed: int,
    stratify_per_class: int | None,
    norm: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess windows for a list of positions."""
    from featureflyers_shl.data.dataset import SHLWindowDataset

    X_parts, y_parts = [], []
    for pos in positions:
        t0 = time.time()
        ds = SHLWindowDataset(
            hdf5_path, split=split,
            position=pos,
            sample_limit=sample_limit,
            seed=seed, preload=True,
        )
        raw_X, raw_y = ds._X, ds._labels
        del ds

        # Drop windows that contain NaN in the raw signal.
        nan_mask = np.isnan(raw_X).any(axis=(1, 2))
        if nan_mask.any():
            n_bad = int(nan_mask.sum())
            raw_X = raw_X[~nan_mask]
            raw_y = raw_y[~nan_mask]
            print(f"  {pos}: {len(raw_X):,} windows  ({time.time()-t0:.0f}s)  "
                  f"[dropped {n_bad} NaN windows]", flush=True)
        else:
            print(f"  {pos}: {len(raw_X):,} windows  ({time.time()-t0:.0f}s)", flush=True)

        X_parts.append(raw_X)
        y_parts.append(raw_y)

    X = np.concatenate(X_parts, axis=0)   # (N, T, 9)
    y = np.concatenate(y_parts, axis=0)   # (N,) 0-based

    if stratify_per_class:
        before = len(X)
        X, y = _stratify_balanced(X, y, stratify_per_class, seed)
        dist = np.bincount(y, minlength=N_CLASSES).tolist()
        print(f"  [STRATIFY] {before:,} → {len(X):,}  cap={stratify_per_class:,}")
        print(f"  {[f'{LABEL_MAP[i]}={n}' for i, n in enumerate(dist)]}")

    if norm:
        print(f"  Normalising ({len(X):,} windows) ...", flush=True)
        X = _per_window_norm(X)

    # (N, T, 9) → (N, 9, T) for 1-D conv
    X = X.transpose(0, 2, 1).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(
    model,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device,
) -> tuple[dict, float, float, list]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import f1_score

    counts  = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float32)
    weights = torch.tensor(1.0 / np.where(counts > 0, counts, 1.0),
                           dtype=torch.float32, device=device)
    weights = weights / weights.sum() * N_CLASSES

    # Sanity-check: NaN in Hips raw data was the known root cause of NaN loss.
    assert not np.isnan(X_tr).any(), "X_tr contains NaN — check raw data NaN-drop step"
    assert not np.isnan(X_va).any(), "X_va contains NaN"

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long())
    va_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va).long())
    # num_workers=0: avoids NaN from pin_memory+multiprocessing with large arrays in RAM.
    # Data is already in RAM so worker overhead exceeds any I/O benefit.
    kw = dict(num_workers=0, pin_memory=False)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  **kw)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, **kw)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    print(f"\n{'Ep':>4}  {'Loss':>8}  {'Tr-Acc':>7}  "
          f"{'Va-Acc':>7}  {'MacroF1':>8}  {'Time':>5}")
    print("-" * 52)

    best_f1 = 0.0
    best_acc = 0.0
    best_state: dict = {}
    no_improve = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot_loss = tot_correct = tot_n = 0

        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss    += loss.item() * len(yb)
            tot_correct += (logits.argmax(1) == yb).sum().item()
            tot_n       += len(yb)

        tr_loss = tot_loss / tot_n
        tr_acc  = tot_correct / tot_n

        model.eval()
        pb, lb = [], []
        with torch.no_grad():
            for xb, yb in va_ld:
                pb.append(model(xb.to(device)).argmax(1).cpu().numpy())
                lb.append(yb.numpy())
        va_preds  = np.concatenate(pb)
        va_labels = np.concatenate(lb)
        va_acc    = float((va_preds == va_labels).mean())
        va_f1     = float(f1_score(va_labels, va_preds,
                                   average="macro", zero_division=0))
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()
        ep_t = time.time() - t0

        print(f"{epoch:>4}  {tr_loss:>8.4f}  {tr_acc:>7.3%}  "
              f"{va_acc:>7.3%}  {va_f1:>8.4f}  {ep_t:>4.1f}s")

        history.append(dict(
            epoch=epoch, train_loss=round(tr_loss, 5), train_acc=round(tr_acc, 4),
            val_acc=round(va_acc, 4), val_macro_f1=round(va_f1, 4),
            lr=round(lr_now, 8), epoch_time_s=round(ep_t, 1),
        ))

        if va_f1 >= best_f1:
            best_f1    = va_f1
            best_acc   = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            print(f"\nEarly stop at epoch {epoch} (no F1 gain for {patience} epochs)")
            break

    return best_state, best_f1, best_acc, history


# ---------------------------------------------------------------------------
# Prediction / submission mode
# ---------------------------------------------------------------------------

def _run_predict_test(args) -> None:
    import torch
    import h5py
    from featureflyers_shl.models.imu_former import build_imu_former

    if args.model_path is None or not args.model_path.exists():
        print(f"ERROR: --model-path not found: {args.model_path}"); sys.exit(1)
    if args.output is None:
        print("ERROR: --output required for --predict-test"); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.model_path}", flush=True)
    bundle = torch.load(args.model_path, map_location="cpu")

    cfg   = bundle["config"]
    model = build_imu_former(
        n_classes=N_CLASSES,
        d=cfg["d"],
        n_heads=cfg["n_heads"],
        n_tf_layers=cfg["n_tf_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(bundle["model_state"])
    model.to(device).eval()
    print(f"  Model loaded  ({model.n_params/1e6:.1f}M params)")

    BATCH = 512
    all_preds = []
    t_start = time.time()

    with h5py.File(str(args.hdf5_path), "r") as hf:
        ds = hf["test"]["data"]
        n_total  = ds.shape[0]
        n_predict = min(n_total, args.limit) if args.limit else n_total
        print(f"Test windows: {n_predict:,} / {n_total:,}\n")

        import torch
        for start in range(0, n_predict, BATCH):
            end       = min(start + BATCH, n_predict)
            raw_batch = ds[start:end].astype(np.float32)       # (B, 500, 9)
            X_pp = _per_window_norm(raw_batch)                  # (B, 500, 9)
            X_ct = torch.from_numpy(X_pp.transpose(0, 2, 1)).to(device)  # (B, 9, 500)
            with torch.no_grad():
                preds = model(X_ct).argmax(1).cpu().numpy()
            all_preds.append(preds)

            done = end
            elapsed = time.time() - t_start
            rate  = done / elapsed if elapsed > 0 else 1
            eta   = (n_predict - done) / rate
            if done % (BATCH * 20) == 0 or done >= n_predict:
                print(f"  {done:,}/{n_predict:,}  {elapsed:.0f}s  eta {eta/60:.1f}m",
                      flush=True)

    predictions = np.concatenate(all_preds)          # 0-indexed
    labels      = predictions + LABEL_OFFSET          # 1-indexed

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for lbl in labels:
            f.write(",".join([str(lbl)] * WIN_SIZE) + "\n")

    print(f"\nWritten : {args.output}  ({len(labels):,} lines)")
    print(f"Total   : {time.time()-t_start:.1f}s")
    print(f"Label range: [{labels.min()}, {labels.max()}]  (expected 1-8)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data
    parser.add_argument("--hdf5-path",     type=Path, default=HDF5_PATH)
    parser.add_argument("--position",      default="Bag",
                        choices=["Bag", "Hand", "Hips", "Torso"])
    parser.add_argument("--fusion",        default="none", choices=["none", "pool"])
    parser.add_argument("--sample-limit",  type=int, default=None)
    parser.add_argument("--stratify-per-class", type=int, default=None,
                        help="Cap each class at N training samples after loading.")
    # Model
    parser.add_argument("--d",             type=int, default=128,
                        help="Per-branch hidden dim; Transformer sees 3*d.")
    parser.add_argument("--n-heads",       type=int, default=4)
    parser.add_argument("--n-tf-layers",   type=int, default=2)
    parser.add_argument("--dropout",       type=float, default=0.1)
    # Training
    parser.add_argument("--epochs",        type=int,   default=60)
    parser.add_argument("--batch-size",    type=int,   default=512)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--patience",      type=int,   default=12)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--output-dir",    type=Path,  default=DEFAULT_OUTD)
    # Prediction
    parser.add_argument("--predict-test",  action="store_true")
    parser.add_argument("--model-path",    type=Path, default=None)
    parser.add_argument("--output",        type=Path, default=None)
    parser.add_argument("--limit",         type=int,  default=None)

    args = parser.parse_args()

    if args.predict_test:
        _run_predict_test(args)
        return

    import torch
    from sklearn.metrics import classification_report, f1_score
    from featureflyers_shl.models.imu_former import build_imu_former

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    positions_train = POSITIONS_ALL if args.fusion == "pool" else [args.position]
    pos_tag   = "Pool" if args.fusion == "pool" else args.position
    lim_str   = str(args.sample_limit) if args.sample_limit else "full"
    strat_tag = f"_strat{args.stratify_per_class}" if args.stratify_per_class else ""

    run_name = (
        f"imuformer_d{args.d}tf{args.n_tf_layers}"
        f"_pos{pos_tag}_s{lim_str}{strat_tag}_ep{args.epochs}"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nIMUFormer — Stage 8")
    print(f"  fusion    : {args.fusion}  positions={positions_train}")
    print(f"  d={args.d}, n_heads={args.n_heads}, n_tf_layers={args.n_tf_layers}, "
          f"dropout={args.dropout}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"  stratify  : {args.stratify_per_class or 'none'}")
    print(f"  device    : {device}")
    print(f"  output    : {run_dir.relative_to(REPO_ROOT)}\n")

    # Build model
    model = build_imu_former(
        n_classes=N_CLASSES,
        d=args.d,
        n_heads=args.n_heads,
        n_tf_layers=args.n_tf_layers,
        dropout=args.dropout,
    )
    total_params = model.n_params
    print(f"IMUFormer: {total_params/1e6:.2f}M parameters\n")

    # Load training data
    print("Loading training data ...", flush=True)
    t0 = time.time()
    X_tr, y_tr = _load_split(
        args.hdf5_path, "train",
        positions=positions_train,
        sample_limit=args.sample_limit,
        seed=args.seed,
        stratify_per_class=args.stratify_per_class,
    )
    print(f"Training set: {X_tr.shape}  ({time.time()-t0:.0f}s)")

    # Load validation data (always Bag, full)
    print("\nLoading validation data (Bag) ...", flush=True)
    t0 = time.time()
    X_va, y_va = _load_split(
        args.hdf5_path, "validation",
        positions=["Bag"],
        sample_limit=None,
        seed=args.seed + 1,
        stratify_per_class=None,
    )
    print(f"Validation set: {X_va.shape}  ({time.time()-t0:.0f}s)")

    # Train
    t_train = time.time()
    best_state, best_f1, best_acc, history = _train(
        model, X_tr, y_tr, X_va, y_va,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=device,
    )
    total_time = time.time() - t_train

    # Final eval
    model.load_state_dict(best_state)
    model.to(device).eval()

    EVAL_BS = args.batch_size
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    va_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va).long())
    va_ld = DataLoader(va_ds, batch_size=EVAL_BS, num_workers=0)
    pb, lb = [], []
    with torch.no_grad():
        for xb, yb in va_ld:
            pb.append(model(xb.to(device)).argmax(1).cpu().numpy())
            lb.append(yb.numpy())
    final_preds  = np.concatenate(pb)
    final_labels = np.concatenate(lb)
    final_f1  = float(f1_score(final_labels, final_preds, average="macro", zero_division=0))
    final_acc = float((final_preds == final_labels).mean())

    present = sorted(set(final_labels.tolist()))
    report  = classification_report(
        final_labels, final_preds,
        labels=present, target_names=[LABEL_MAP[l] for l in present],
        zero_division=0,
    )

    print(f"\n{'='*60}")
    print(f"  IMUFormer  d={args.d}  tf_layers={args.n_tf_layers}")
    print(f"  Accuracy : {final_acc:.4f}")
    print(f"  Macro-F1 : {final_f1:.4f}")
    print(f"  vs Stage6 hybrid (0.697): {final_f1 - 0.697:+.4f}")
    print(f"  vs submitted baseline (0.6481): {final_f1 - 0.6481:+.4f}")
    print(f"{'='*60}")
    print(report)

    # Save artifacts
    config = dict(
        d=args.d, n_heads=args.n_heads, n_tf_layers=args.n_tf_layers,
        dropout=args.dropout,
    )
    torch.save(
        dict(model_state=best_state, config=config),
        run_dir / "best_model.pt",
    )

    metrics = dict(
        architecture="IMUFormer",
        d=args.d, n_heads=args.n_heads, n_tf_layers=args.n_tf_layers,
        n_params=total_params,
        fusion=args.fusion, positions=positions_train,
        stratify_per_class=args.stratify_per_class,
        sample_limit=args.sample_limit,
        best_val_macro_f1=round(final_f1, 4),
        best_val_accuracy=round(final_acc, 4),
        stage6_hybrid_f1=0.697,
        submitted_baseline_f1=0.6481,
        delta_vs_stage6=round(final_f1 - 0.697, 4),
        delta_vs_baseline=round(final_f1 - 0.6481, 4),
        total_train_time_s=round(total_time, 1),
        epochs_trained=len(history),
        history=history,
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "classification_report.txt").write_text(report)

    print(f"\nArtifacts saved to {run_dir.relative_to(REPO_ROOT)}/")
    print("  best_model.pt  config.json  metrics.json  classification_report.txt")


if __name__ == "__main__":
    main()
