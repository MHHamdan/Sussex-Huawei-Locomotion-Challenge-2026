#!/usr/bin/env python3
"""
Stage-5 foundation-model ablation trainer + submission generator.

Trains a lightweight head on top of FROZEN foundation-model embeddings.
Supports multiple preprocessing modes, embedding strategies, head types,
and a pool-fusion mode that stacks all 4 sensor positions as independent
training samples (submission-compatible).

Bug fix vs. stage-4
-------------------
The previous MOMENT encoder silently transposed (B,C,T) → (B,T,C), causing
MOMENT to interpret 500 sensor readings as "channels" and the 9 axes as
time steps.  This is corrected in the new MomentEncoder (foundation.py).

Preprocessing options
---------------------
  --norm none            no normalisation (original sensor units)
  --norm per-window      z-score each window independently, per channel
  --norm channel-global  fit channel mean/std on training set, apply to both splits
  --norm train-stats     same as channel-global but caches stats to disk for reuse

  --include-magnitude    append Acc / Gyr / Mag Euclidean magnitudes (3 extra channels)
  --include-delta        append first differences of all current channels

Fusion modes
------------
  --fusion none   single position (--position selects which; default Bag)
  --fusion pool   stack Bag/Hand/Hips/Torso as independent training samples
                  test-compatible: test/data has no per-position split

Embedding strategies
--------------------
  --embed-strategy mean_pool       global mean over patch tokens → (B, 1024)
  --embed-strategy last_patch      last patch token / last-quarter mean → (B, 1024)
  --embed-strategy sensorwise      per-channel MOMENT embeddings, concat → (B, C×1024)
  --embed-strategy flatten_patches alias of sensorwise

Head types
----------
  --head linear          nn.Linear(embed_dim, 8)
  --head mlp             Linear→ReLU→Dropout→Linear  (default)
  --head residual_mlp    2-block residual MLP with LayerNorm
  --head xgb             XGBoost (sklearn); encoder stays frozen
  --head logistic        LogisticRegression (sklearn)

Stat-only mode
--------------
  --stat-only  skip embedding; train on 354 statistical/spectral features only.
               Used for apples-to-apples comparison against foundation model.

Hybrid mode
-----------
  --hybrid-stat-features  concatenate 354-dim statistical features with embeddings

Prediction (submission) mode
----------------------------
  --predict-test          load a saved model.joblib artifact and predict on test set
  --model-path PATH       path to the model.joblib artifact
  --output PATH           output submission file
  --limit N               only predict first N test windows (smoke test)

Ablation table (Bag position, 20 000 windows, seed=42)
-------------------------------------------------------
  A: stat-only + XGB          (use --stat-only --head xgb)
  B: MOMENT embedding + XGB   (Exp E, already done: F1=0.6968)
  C: MOMENT hybrid + XGB      (Exp F, already done: F1=0.7329)

Baseline to beat: XGBoost pool, macro-F1=0.6389, accuracy=68.4%

Usage examples
--------------
# Apples-to-apples A: stat-only XGB, 20k Bag
python scripts/train_foundation_head.py \\
    --position Bag --sample-limit 20000 --stat-only --head xgb --device cuda:1

# Smoke test (5 k windows, 3 epochs)
python scripts/train_foundation_head.py \\
    --position Bag --sample-limit 5000 --encoder moment \\
    --norm per-window --embed-strategy mean_pool \\
    --head mlp --epochs 3 --batch-size 512 --device cuda:1

# Full pool hybrid (best submission config, ~5h first run)
python scripts/train_foundation_head.py \\
    --encoder moment --norm per-window --embed-strategy mean_pool \\
    --head xgb --hybrid-stat-features --fusion pool --device cuda

# Smoke test submission (first 1000 lines)
python scripts/train_foundation_head.py \\
    --predict-test \\
    --model-path outputs/execution-output/<run>/model.joblib \\
    --output outputs/execution-output/submissions/smoke.txt \\
    --limit 1000

# Full submission
python scripts/train_foundation_head.py \\
    --predict-test \\
    --model-path outputs/execution-output/<run>/model.joblib \\
    --output outputs/execution-output/submissions/FeatureFlyers_foundation_hybrid_pool_full.txt
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
STATS_DIR    = REPO_ROOT / "dataset" / "processed" / "train_channel_stats"
FEAT_DIR     = REPO_ROOT / "dataset" / "processed" / "features"
DEFAULT_OUTD = REPO_ROOT / "outputs" / "execution-output"

LABEL_MAP = {0: "Still", 1: "Walking", 2: "Run", 3: "Bike",
             4: "Car",   5: "Bus",     6: "Train", 7: "Metro"}
N_CLASSES      = 8
LABEL_OFFSET   = 1          # 0-indexed predictions → 1-indexed submission labels
POSITIONS_ALL  = ["Bag", "Hand", "Hips", "Torso"]
WIN_SIZE       = 500

XGB_POOL_BASELINE_F1 = 0.6389

PYTORCH_HEADS = {"linear", "mlp", "residual_mlp"}
SKLEARN_HEADS = {"xgb", "logistic"}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _add_magnitude(X: np.ndarray) -> np.ndarray:
    """X: (N, T, C) → (N, T, C+3) with Acc/Gyr/Mag magnitudes appended."""
    acc = np.linalg.norm(X[:, :, 0:3], axis=2, keepdims=True)
    gyr = np.linalg.norm(X[:, :, 3:6], axis=2, keepdims=True)
    mag = np.linalg.norm(X[:, :, 6:9], axis=2, keepdims=True)
    return np.concatenate([X, acc, gyr, mag], axis=2)


def _add_delta(X: np.ndarray) -> np.ndarray:
    """X: (N, T, C) → (N, T, 2C) with first differences appended."""
    delta = np.diff(X, axis=1, prepend=X[:, :1, :])
    return np.concatenate([X, delta], axis=2)


def preprocess_windows(
    X_raw: np.ndarray,
    norm: str,
    include_magnitude: bool,
    include_delta: bool,
    train_mu: np.ndarray | None = None,
    train_sd: np.ndarray | None = None,
) -> np.ndarray:
    """Apply preprocessing to (N, T, 9) raw windows. Returns (N, T, C_new) float32."""
    X = X_raw.astype(np.float32, copy=True)

    if include_magnitude:
        X = _add_magnitude(X)
    if include_delta:
        X = _add_delta(X)

    if norm == "per-window":
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-8
        X = (X - mu) / sd

    elif norm in ("channel-global", "train-stats"):
        if train_mu is None or train_sd is None:
            raise ValueError("train_mu/train_sd required for channel-global norm")
        X = (X - train_mu) / (train_sd + 1e-8)

    return X.astype(np.float32)


def compute_train_stats(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std over all windows and time steps."""
    N, T, C = X_train.shape
    flat = X_train.reshape(-1, C)
    return flat.mean(axis=0), flat.std(axis=0)


# ---------------------------------------------------------------------------
# Stat feature helpers
# ---------------------------------------------------------------------------

def _extract_stat_features(X_raw: np.ndarray) -> np.ndarray:
    """Extract 354-dim stat/spectral features from raw (N, T, 9) windows."""
    from featureflyers_shl.features.statistical import extract_batch
    print(f"  [STAT] extracting {len(X_raw):,} windows ...", flush=True)
    t0 = time.time()
    F  = extract_batch(X_raw)
    print(f"  [STAT] done in {time.time() - t0:.1f}s  F={F.shape}")
    return F.astype(np.float32)


def _load_stat_from_cache(split: str, position: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load precomputed stat features for a full split/position.

    Cache format: dataset/processed/features/{split}_{position}.npz
      X: (N, 354) float32   (aligned with SHLWindowDataset full-dataset order)
      y: (N,) int64         (1-based labels; convert to 0-based with -1)
    """
    cache = FEAT_DIR / f"{split}_{position}.npz"
    if not cache.exists():
        raise FileNotFoundError(
            f"Precomputed feature cache not found: {cache}\n"
            f"Run: python scripts/precompute_features.py --positions {position}"
        )
    print(f"  [STAT] loading from cache: {cache.name}", flush=True)
    d = np.load(cache)
    X = d["X"].astype(np.float32)
    y = (d["y"] - 1).astype(np.int64)   # 1-based → 0-based
    print(f"    X={X.shape}  y={y.shape}")
    return X, y


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def _embedding_cache_path(
    split: str,
    position: str,
    encoder_name: str,
    norm_tag: str,
    mag_tag: str,
    dlt_tag: str,
    embed_strategy: str,
    sample_limit: int | None,
) -> Path:
    lim = f"_s{sample_limit}" if sample_limit else ""
    return EMBED_DIR / (
        f"{split}_{position}_{encoder_name}"
        f"_norm{norm_tag}{mag_tag}{dlt_tag}_{embed_strategy}{lim}.npz"
    )


def _extract_embeddings(
    encoder,
    X_proc: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward (N, T, C_proc) preprocessed windows through the frozen encoder.
    Internally transposes each batch to (B, C, T) before encoding.
    Returns (X_emb: (N, embed_dim) float32, y: (N,) int64).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    X_ct = np.transpose(X_proc, (0, 2, 1)).astype(np.float32)
    ds   = TensorDataset(
        torch.from_numpy(X_ct),
        torch.from_numpy(y.astype(np.int64)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    encoder.to(device)
    encoder.eval()

    emb_list, lbl_list = [], []
    total = len(ds)
    t0    = time.time()

    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader):
            xb  = xb.to(device)
            emb = encoder(xb).float()
            emb_list.append(emb.cpu().numpy())
            lbl_list.append(yb.numpy())
            done = min((i + 1) * batch_size, total)
            if done % (batch_size * 20) == 0 or done >= total:
                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed > 0 else 0
                eta     = (total - done) / rate if rate > 0 else 0
                print(f"    {done:,}/{total:,}  ({done / total * 100:.0f}%)"
                      f"  {elapsed:.0f}s  eta {eta/60:.1f}m", flush=True)

    X_emb = np.vstack(emb_list).astype(np.float32)
    y_out = np.concatenate(lbl_list).astype(np.int64)
    print(f"  Done: {time.time() - t0:.1f}s  X_emb={X_emb.shape}")
    return X_emb, y_out


def _get_embeddings_for(
    split: str,
    position: str,
    X_pp: np.ndarray,
    y: np.ndarray,
    encoder,
    extract_batch_size: int,
    device,
    norm_tag: str,
    mag_tag: str,
    dlt_tag: str,
    embed_strategy: str,
    encoder_name: str,
    sample_limit: int | None,
    force_extract: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached embeddings or extract and cache them for one split/position."""
    cache = _embedding_cache_path(
        split, position, encoder_name,
        norm_tag, mag_tag, dlt_tag, embed_strategy, sample_limit,
    )
    if cache.exists() and not force_extract:
        print(f"  [CACHE] {split}/{position} embeddings from {cache.name}", flush=True)
        d = np.load(cache)
        print(f"    X={d['X'].shape}")
        return d["X"].astype(np.float32), d["y"].astype(np.int64)

    print(f"  [EXTRACT] {split}/{position} ({len(X_pp):,} windows) ...", flush=True)
    X_emb, y_out = _extract_embeddings(encoder, X_pp, y, extract_batch_size, device)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X_emb, y=y_out)
    print(f"  Saved {cache.name}")
    return X_emb, y_out


# ---------------------------------------------------------------------------
# PyTorch head training
# ---------------------------------------------------------------------------

def _train_pytorch_head(
    head,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    dropout: float,
    patience: int,
    device,
) -> tuple[object, float, float, list, bool]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import f1_score

    counts  = np.bincount(y_train, minlength=N_CLASSES).astype(np.float32)
    weights = torch.tensor(1.0 / np.where(counts > 0, counts, 1.0),
                           dtype=torch.float32, device=device)
    weights = weights / weights.sum() * N_CLASSES

    train_ds = TensorDataset(torch.from_numpy(X_train),
                             torch.from_numpy(y_train).long())
    val_ds   = TensorDataset(torch.from_numpy(X_val),
                             torch.from_numpy(y_val).long())
    lkw = dict(batch_size=batch_size, num_workers=0,
               pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True,  **lkw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **lkw)

    head = head.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=lr * 0.01)

    print(f"{'Ep':>4}  {'Loss':>8}  {'TrainAcc':>8}  "
          f"{'ValAcc':>7}  {'MacroF1':>8}  {'Time':>5}")
    print("-" * 52)

    best_f1 = 0.0; best_acc = 0.0; best_state = None; no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
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
            best_acc   = val_acc
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}: "
                  f"no improvement for {patience} epochs.")
            return best_state, best_f1, best_acc, history, True

    return best_state, best_f1, best_acc, history, False


# ---------------------------------------------------------------------------
# Sklearn head training
# ---------------------------------------------------------------------------

def _train_sklearn_head(
    head_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    xgb_device: str = "cpu",
) -> tuple[object, float, float]:
    from sklearn.metrics import f1_score

    if head_type == "xgb":
        from xgboost import XGBClassifier
        counts = np.bincount(y_train, minlength=N_CLASSES).astype(np.float64)
        total  = counts.sum()
        sample_weight = total / (N_CLASSES * counts[y_train])

        model = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            tree_method="hist",
            device=xgb_device,
            n_jobs=-1,
            random_state=seed,
            verbosity=1,
        )
        print("  Fitting XGBoost ...", flush=True)
        t0 = time.time()
        model.fit(X_train, y_train, sample_weight=sample_weight)
        print(f"  XGBoost fit done in {time.time() - t0:.1f}s")

    elif head_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            max_iter=2000, class_weight="balanced",
            C=1.0, solver="lbfgs", multi_class="multinomial",
            random_state=seed, n_jobs=-1,
        )
        print("  Fitting LogisticRegression ...", flush=True)
        t0 = time.time()
        model.fit(X_train, y_train)
        print(f"  LR fit done in {time.time() - t0:.1f}s")

    else:
        raise ValueError(f"Unknown sklearn head: {head_type!r}")

    val_preds = model.predict(X_val)
    val_f1    = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
    val_acc   = float((val_preds == y_val).mean())
    return model, val_f1, val_acc


# ---------------------------------------------------------------------------
# Artifact save / load
# ---------------------------------------------------------------------------

def _save_artifact(
    run_dir: Path,
    *,
    head_type: str,
    sklearn_model=None,
    pytorch_state: dict | None = None,
    pytorch_head_config: dict | None = None,
    encoder_name: str,
    embed_strategy: str,
    norm: str,
    include_magnitude: bool,
    include_delta: bool,
    hybrid_stat_features: bool,
    stat_only: bool,
    fusion: str,
    positions: list[str],
    n_channels: int,
    embed_dim: int,
    feature_dim: int,
    train_mu: np.ndarray | None,
    train_sd: np.ndarray | None,
    seed: int,
    hdf5_path: str,
) -> Path:
    """
    Save a self-contained model bundle to run_dir/model.joblib.

    The bundle carries everything needed to reproduce predictions at test time:
    encoder config, preprocessing config, fusion mode, head weights, and label
    offset (+1 to convert 0-indexed predictions to 1–8 submission labels).
    """
    try:
        import joblib
    except ImportError:
        print("  [WARN] joblib not installed — model.joblib not saved")
        return run_dir / "model.joblib"

    bundle = dict(
        # --- encoder ---
        encoder_name=encoder_name,
        embed_strategy=embed_strategy,
        # --- preprocessing ---
        norm=norm,
        include_magnitude=include_magnitude,
        include_delta=include_delta,
        # --- feature construction ---
        hybrid_stat_features=hybrid_stat_features,
        stat_only=stat_only,
        # --- training scope ---
        fusion=fusion,
        positions=positions,
        # --- dimensions ---
        n_channels=n_channels,
        embed_dim=embed_dim,
        feature_dim=feature_dim,
        # --- channel-global norm params (None for per-window) ---
        train_mu=train_mu,
        train_sd=train_sd,
        # --- label convention ---
        label_offset=LABEL_OFFSET,   # add this to 0-indexed pred → 1-indexed label
        n_classes=N_CLASSES,
        label_map=LABEL_MAP,
        # --- reproducibility ---
        seed=seed,
        hdf5_path=hdf5_path,
        # --- head ---
        head_type=head_type,
        sklearn_model=sklearn_model,        # set for xgb / logistic
        pytorch_state=pytorch_state,        # set for linear / mlp / residual_mlp
        pytorch_head_config=pytorch_head_config,
    )
    out = run_dir / "model.joblib"
    joblib.dump(bundle, out)
    print(f"  Saved model.joblib  ({out.stat().st_size / 1e6:.1f} MB)")
    return out


# ---------------------------------------------------------------------------
# Prediction / submission mode
# ---------------------------------------------------------------------------

def _run_predict_test(args) -> None:
    """
    Load a model.joblib bundle and generate an SHL submission file.

    Test HDF5 shape: (92726, 500, 9) — already windowed, no per-position split.
    Submission format: 92 726 lines, each 500 comma-separated integers in 1–8.
    """
    import joblib
    import h5py

    if args.model_path is None or not args.model_path.exists():
        print(f"ERROR: --model-path not found: {args.model_path}")
        sys.exit(1)
    if args.output is None:
        print("ERROR: --output required for --predict-test")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Prediction mode")
    print(f"  model : {args.model_path}")
    print(f"  output: {args.output}")
    print(f"  limit : {args.limit or 'none (full 92 726 windows)'}")
    print(f"{'=' * 60}\n")

    bundle = joblib.load(args.model_path)

    head_type        = bundle["head_type"]
    stat_only        = bundle["stat_only"]
    encoder_name     = bundle["encoder_name"]
    embed_strategy   = bundle["embed_strategy"]
    norm             = bundle["norm"]
    include_magnitude = bundle["include_magnitude"]
    include_delta    = bundle["include_delta"]
    hybrid           = bundle["hybrid_stat_features"]
    n_channels       = bundle["n_channels"]
    label_offset     = bundle["label_offset"]
    train_mu         = bundle.get("train_mu")
    train_sd         = bundle.get("train_sd")

    if head_type not in SKLEARN_HEADS:
        print(f"ERROR: --predict-test only supports sklearn heads (xgb, logistic); "
              f"got '{head_type}'. For PyTorch heads, save head.pt and write a custom "
              f"inference script.")
        sys.exit(1)

    sklearn_model = bundle["sklearn_model"]
    if sklearn_model is None:
        print("ERROR: bundle contains no sklearn_model")
        sys.exit(1)

    # Load encoder
    if not stat_only:
        from featureflyers_shl.models.foundation import get_encoder
        import torch
        device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                  if args.device == "auto" else torch.device(args.device))
        print(f"Loading encoder '{encoder_name}' ...", flush=True)
        t0 = time.time()
        encoder = get_encoder(encoder_name, embed_strategy, n_channels=n_channels)
        encoder.eval()
        encoder.to(device)
        print(f"  Loaded in {time.time() - t0:.1f}s  embed_dim={encoder.embed_dim}")
    else:
        device = None

    # Open test HDF5
    if not args.hdf5_path.exists():
        print(f"ERROR: HDF5 not found: {args.hdf5_path}")
        sys.exit(1)

    BATCH = 512
    all_preds = []
    t_start = time.time()

    with h5py.File(str(args.hdf5_path), "r") as hf:
        ds = hf["test"]["data"]
        n_total = ds.shape[0]
        n_predict = min(n_total, args.limit) if args.limit else n_total
        print(f"Test windows to predict : {n_predict:,} / {n_total:,}\n")

        for start in range(0, n_predict, BATCH):
            end       = min(start + BATCH, n_predict)
            batch_raw = ds[start:end].astype(np.float32)   # (B, 500, 9)

            # Preprocess
            X_pp = preprocess_windows(
                batch_raw, norm, include_magnitude, include_delta,
                train_mu, train_sd,
            )

            # Build feature vector
            if not stat_only:
                import torch
                X_ct  = np.transpose(X_pp, (0, 2, 1)).astype(np.float32)  # (B, C, T)
                emb_parts = []
                inner_bs = bundle.get("extract_batch_size", 64)
                with torch.no_grad():
                    for i0 in range(0, len(X_ct), inner_bs):
                        xb = torch.from_numpy(X_ct[i0:i0 + inner_bs]).to(device)
                        emb_parts.append(encoder(xb).float().cpu().numpy())
                X_emb = np.concatenate(emb_parts, axis=0)                 # (B, embed_dim)
                if hybrid:
                    from featureflyers_shl.features.statistical import extract_batch
                    X_stat = extract_batch(batch_raw).astype(np.float32)   # (B, 354)
                    X_feat = np.concatenate([X_emb, X_stat], axis=1)
                else:
                    X_feat = X_emb
            else:
                from featureflyers_shl.features.statistical import extract_batch
                X_feat = extract_batch(batch_raw).astype(np.float32)

            preds = sklearn_model.predict(X_feat).astype(np.int32)   # 0-indexed
            all_preds.append(preds)

            done = end
            elapsed = time.time() - t_start
            rate    = done / elapsed if elapsed > 0 else 0
            eta     = (n_predict - done) / rate if rate > 0 else 0
            if done % (BATCH * 20) == 0 or done >= n_predict:
                print(f"  {done:,}/{n_predict:,}  ({done/n_predict*100:.0f}%)"
                      f"  {elapsed:.0f}s  eta {eta/60:.1f}m", flush=True)

    predictions = np.concatenate(all_preds)         # 0-indexed
    labels      = predictions + label_offset         # 1-indexed (1–8)

    # Validate label range
    bad = np.sum((labels < 1) | (labels > 8))
    if bad > 0:
        print(f"  [WARN] {bad} predictions outside 1–8 range!")
    else:
        print(f"  Label range check: all in 1–8  ({len(labels):,} windows)")

    # Write submission
    args.output.parent.mkdir(parents=True, exist_ok=True)
    t_write = time.time()
    with open(args.output, "w") as f:
        for lbl in labels:
            f.write(",".join([str(lbl)] * WIN_SIZE) + "\n")

    n_lines = len(labels)
    print(f"\n  Written : {args.output}")
    print(f"  Lines   : {n_lines:,}  (expected: {n_predict:,})")
    print(f"  Write   : {time.time() - t_write:.1f}s")
    print(f"  Total   : {time.time() - t_start:.1f}s")
    print(f"\nSmoke-test verification:")
    print(f"  Line count  : {n_lines}")
    print(f"  Fields/line : 500  (constant)")
    print(f"  Label range : [{int(labels.min())}, {int(labels.max())}]  (expected 1–8)")
    class_counts = np.bincount(labels, minlength=9)[1:]
    for i, (name, cnt) in enumerate(zip(LABEL_MAP.values(), class_counts)):
        print(f"    {name:<12}: {cnt:,}  ({cnt/n_lines*100:.1f}%)")


# ---------------------------------------------------------------------------
# Main — training mode
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- data / scope ---
    parser.add_argument("--hdf5-path",      type=Path, default=HDF5_PATH)
    parser.add_argument("--position",       default="Bag",
                        choices=["Bag", "Hand", "Hips", "Torso"],
                        help="Position for fusion=none training and for validation")
    parser.add_argument("--fusion",         default="none",
                        choices=["none", "pool"],
                        help="none: single position; pool: stack all 4 positions as "
                             "independent training samples (test-compatible)")
    parser.add_argument("--sample-limit",   type=int, default=None,
                        help="Stratified sample this many windows per position "
                             "(None = full dataset)")
    # --- encoder ---
    parser.add_argument("--encoder",        default="moment",
                        choices=["moment", "fallback", "chronos", "uni2ts"])
    parser.add_argument("--embed-strategy", default="mean_pool",
                        choices=["mean_pool", "last_patch", "sensorwise",
                                 "flatten_patches"])
    # --- preprocessing ---
    parser.add_argument("--norm",           default="none",
                        choices=["none", "per-window", "channel-global", "train-stats"])
    parser.add_argument("--include-magnitude", action="store_true")
    parser.add_argument("--include-delta",     action="store_true")
    # --- head / features ---
    parser.add_argument("--head",           default="mlp",
                        choices=["linear", "mlp", "residual_mlp", "xgb", "logistic"])
    parser.add_argument("--hybrid-stat-features", action="store_true",
                        help="Concatenate 354-dim statistical features with embeddings")
    parser.add_argument("--stat-only",      action="store_true",
                        help="Skip embedding; train on 354 stat/spectral features only. "
                             "For apples-to-apples comparison against foundation model.")
    # --- PyTorch hyper-parameters ---
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch-size",     type=int,   default=512)
    parser.add_argument("--extract-batch-size", type=int, default=64,
                        help="Batch size for frozen encoder forward pass")
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--dropout",        type=float, default=0.3)
    parser.add_argument("--hidden",         type=int,   default=256)
    parser.add_argument("--patience",       type=int,   default=10)
    # --- misc ---
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--xgb-device",     default="cpu",
                        help="Device for XGBoost fitting (default: cpu to avoid GPU OOM on large datasets)")
    parser.add_argument("--output-dir",     type=Path,  default=DEFAULT_OUTD)
    parser.add_argument("--force-extract",  action="store_true",
                        help="Re-run embedding extraction even if cache exists")
    # --- prediction mode ---
    parser.add_argument("--predict-test",   action="store_true",
                        help="Prediction mode: load artifact, predict test set, "
                             "write submission")
    parser.add_argument("--model-path",     type=Path, default=None,
                        help="Path to model.joblib (required for --predict-test)")
    parser.add_argument("--output",         type=Path, default=None,
                        help="Output submission file (required for --predict-test)")
    parser.add_argument("--limit",          type=int, default=None,
                        help="Only predict first N test windows (smoke test)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Prediction mode — entirely separate path
    # ------------------------------------------------------------------
    if args.predict_test:
        _run_predict_test(args)
        return

    # ------------------------------------------------------------------
    # Training mode — imports
    # ------------------------------------------------------------------
    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed"); sys.exit(1)
    try:
        from sklearn.metrics import classification_report, f1_score
    except ImportError:
        print("ERROR: scikit-learn not installed"); sys.exit(1)

    from featureflyers_shl.data.dataset   import SHLWindowDataset
    from featureflyers_shl.models.foundation import get_encoder, build_pytorch_head

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    if not args.hdf5_path.exists():
        print(f"ERROR: HDF5 not found: {args.hdf5_path}"); sys.exit(1)

    if args.stat_only and args.head not in SKLEARN_HEADS:
        print(f"ERROR: --stat-only only supports sklearn heads (xgb, logistic); "
              f"got '{args.head}'")
        sys.exit(1)

    if args.fusion == "pool" and args.norm in ("channel-global", "train-stats"):
        print("ERROR: channel-global / train-stats norm is not supported in pool mode "
              "(stats would need to be computed from the combined 4-position training set). "
              "Use --norm per-window or --norm none for pool fusion.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Positions & run name
    # ------------------------------------------------------------------
    positions_train = POSITIONS_ALL if args.fusion == "pool" else [args.position]
    val_position    = "Bag"          # always Bag-only validation for fair comparison

    lim_str  = str(args.sample_limit) if args.sample_limit else "full"
    norm_tag = args.norm.replace("-", "")
    mag_tag  = "_mag"    if args.include_magnitude    else ""
    dlt_tag  = "_delta"  if args.include_delta        else ""
    hyb_tag  = "_hybrid" if args.hybrid_stat_features else ""
    so_tag   = "_statonly" if args.stat_only          else ""
    pos_tag  = "Pool" if args.fusion == "pool" else args.position

    encoder_tag = "stat" if args.stat_only else args.encoder
    run_name = (
        f"foundation_{encoder_tag}_pos{pos_tag}"
        f"_{args.embed_strategy}_norm{norm_tag}{mag_tag}{dlt_tag}"
        f"_{args.head}{hyb_tag}{so_tag}_s{lim_str}"
        + (f"_ep{args.epochs}" if args.head in PYTORCH_HEADS else "")
    )
    if args.stat_only:
        run_name = (
            f"foundation_statonly_pos{pos_tag}"
            f"_{args.head}{hyb_tag}_s{lim_str}"
            + (f"_ep{args.epochs}" if args.head in PYTORCH_HEADS else "")
        )

    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFoundation ablation — training mode")
    print(f"  encoder        : {'stat-only (no embedding)' if args.stat_only else args.encoder}")
    print(f"  embed_strategy : {args.embed_strategy if not args.stat_only else 'n/a'}")
    print(f"  fusion         : {args.fusion}  positions={positions_train}")
    print(f"  norm           : {args.norm}")
    print(f"  magnitude      : {args.include_magnitude}")
    print(f"  delta          : {args.include_delta}")
    print(f"  head           : {args.head}")
    print(f"  stat-only      : {args.stat_only}")
    print(f"  hybrid stats   : {args.hybrid_stat_features}")
    print(f"  sample_limit   : {lim_str} per position")
    print(f"  val_position   : {val_position}")
    print(f"  device         : {device}")
    print(f"  output         : {run_dir.relative_to(REPO_ROOT)}\n")

    # ------------------------------------------------------------------
    # Load encoder (skip for stat-only)
    # ------------------------------------------------------------------
    encoder        = None
    embed_dim      = 0
    is_placeholder = False

    if not args.stat_only:
        print(f"Loading {args.encoder} encoder ...", flush=True)
        t0 = time.time()
        try:
            # n_channels determined after preprocessing; use 9 as placeholder for
            # initial load (affects embed_dim only for sensorwise strategy — corrected
            # after preprocessing if needed).
            encoder = get_encoder(
                args.encoder, args.embed_strategy,
                n_channels=9 + (3 if args.include_magnitude else 0)
                              * (1 + (1 if args.include_delta else 0))
                              + (9 if args.include_delta else 0),
            )
        except ImportError as exc:
            print(f"\n[ALT ENCODER] {exc}\n  → aborting.")
            sys.exit(1)
        encoder.eval()
        embed_dim      = encoder.embed_dim
        is_placeholder = getattr(encoder, "is_placeholder", False)
        if is_placeholder:
            print("  [WARNING] fallback encoder is NOT a real foundation model.")
        print(f"  embed_dim={embed_dim}  loaded in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Build training feature matrix (pool: one position at a time)
    # ------------------------------------------------------------------
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    X_tr_parts: list[np.ndarray] = []
    y_tr_parts: list[np.ndarray] = []

    train_mu: np.ndarray | None = None
    train_sd: np.ndarray | None = None

    print("Building TRAINING features ...", flush=True)
    t_feat = time.time()

    for pos in positions_train:
        print(f"\n  --- Position: {pos} ---", flush=True)

        # Load raw windows for this position
        is_full = args.sample_limit is None
        t0 = time.time()
        ds_pos = SHLWindowDataset(
            args.hdf5_path, split="train",
            position=pos,
            sample_limit=args.sample_limit,
            seed=args.seed, preload=True,
        )
        print(f"  Loaded {len(ds_pos):,} windows from {pos} in {time.time()-t0:.0f}s")
        X_raw_pos = ds_pos._X       # (N, T, 9) float32
        y_raw_pos = ds_pos._labels  # (N,) int64  0-based

        # Compute channel stats for first position (pool=first pos ≈ Bag stats)
        if args.norm in ("channel-global", "train-stats") and train_mu is None:
            X_tmp = X_raw_pos.astype(np.float32, copy=True)
            if args.include_magnitude:
                X_tmp = _add_magnitude(X_tmp)
            if args.include_delta:
                X_tmp = _add_delta(X_tmp)
            train_mu, train_sd = compute_train_stats(X_tmp)
            del X_tmp
            if args.norm == "train-stats":
                STATS_DIR.mkdir(parents=True, exist_ok=True)
                stats_tag  = (f"pos{pos_tag}"
                              + ("_mag" if args.include_magnitude else "")
                              + ("_delta" if args.include_delta else ""))
                stats_file = STATS_DIR / f"train_stats_{stats_tag}.npz"
                if not stats_file.exists():
                    np.savez_compressed(stats_file, mu=train_mu, sd=train_sd)
                    print(f"  [STATS] saved to {stats_file.name}")

        if args.stat_only:
            # Stat-only: use precomputed cache for full dataset; extract for limited
            if is_full:
                X_stat_pos, _ = _load_stat_from_cache("train", pos)
                X_tr_parts.append(X_stat_pos)
                y_tr_parts.append(y_raw_pos)
            else:
                X_stat_pos = _extract_stat_features(X_raw_pos)
                X_tr_parts.append(X_stat_pos)
                y_tr_parts.append(y_raw_pos)
        else:
            # Embedding mode
            X_pp_pos = preprocess_windows(
                X_raw_pos, args.norm, args.include_magnitude, args.include_delta,
                train_mu, train_sd,
            )
            n_channels = X_pp_pos.shape[2]

            X_emb_pos, y_emb_pos = _get_embeddings_for(
                split="train", position=pos,
                X_pp=X_pp_pos, y=y_raw_pos,
                encoder=encoder,
                extract_batch_size=args.extract_batch_size,
                device=device,
                norm_tag=norm_tag, mag_tag=mag_tag, dlt_tag=dlt_tag,
                embed_strategy=args.embed_strategy,
                encoder_name=args.encoder,
                sample_limit=args.sample_limit,
                force_extract=args.force_extract,
            )

            if args.hybrid_stat_features:
                if is_full:
                    X_stat_pos, _ = _load_stat_from_cache("train", pos)
                else:
                    X_stat_pos = _extract_stat_features(X_raw_pos)
                X_feat_pos = np.concatenate([X_emb_pos, X_stat_pos], axis=1)
                print(f"  Hybrid dim: {X_feat_pos.shape[1]}  "
                      f"(emb={X_emb_pos.shape[1]} + stat={X_stat_pos.shape[1]})")
            else:
                X_feat_pos = X_emb_pos

            X_tr_parts.append(X_feat_pos)
            y_tr_parts.append(y_emb_pos)

        del ds_pos, X_raw_pos

    X_tr = np.concatenate(X_tr_parts, axis=0)
    y_tr = np.concatenate(y_tr_parts, axis=0)
    n_channels = (9 + (3 if args.include_magnitude else 0)
                  + (9 + (3 if args.include_magnitude else 0)) * (1 if args.include_delta else 0))
    print(f"\nTraining set: {X_tr.shape}  "
          f"({time.time() - t_feat:.0f}s total)")

    # ------------------------------------------------------------------
    # Build validation feature matrix (always Bag-only)
    # ------------------------------------------------------------------
    print(f"\nBuilding VALIDATION features (Bag) ...", flush=True)
    t_val = time.time()

    is_full_val = args.sample_limit is None
    ds_val = SHLWindowDataset(
        args.hdf5_path, split="validation",
        position=val_position,
        sample_limit=args.sample_limit,
        seed=args.seed + 1, preload=True,
    )
    print(f"  Loaded {len(ds_val):,} val windows in {time.time()-t_val:.0f}s")
    X_val_raw   = ds_val._X
    y_val_raw   = ds_val._labels

    if args.stat_only:
        if is_full_val:
            X_va, _ = _load_stat_from_cache("validation", val_position)
        else:
            X_va = _extract_stat_features(X_val_raw)
        y_va = y_val_raw
    else:
        X_val_pp = preprocess_windows(
            X_val_raw, args.norm, args.include_magnitude, args.include_delta,
            train_mu, train_sd,
        )
        X_va_emb, y_va = _get_embeddings_for(
            split="validation", position=val_position,
            X_pp=X_val_pp, y=y_val_raw,
            encoder=encoder,
            extract_batch_size=args.extract_batch_size,
            device=device,
            norm_tag=norm_tag, mag_tag=mag_tag, dlt_tag=dlt_tag,
            embed_strategy=args.embed_strategy,
            encoder_name=args.encoder,
            sample_limit=args.sample_limit,
            force_extract=args.force_extract,
        )
        if args.hybrid_stat_features:
            if is_full_val:
                X_va_stat, _ = _load_stat_from_cache("validation", val_position)
            else:
                X_va_stat = _extract_stat_features(X_val_raw)
            X_va = np.concatenate([X_va_emb, X_va_stat], axis=1)
        else:
            X_va = X_va_emb

    del ds_val, X_val_raw

    print(f"Validation set: {X_va.shape}  y={y_va.shape}")

    # ------------------------------------------------------------------
    # Train head
    # ------------------------------------------------------------------
    t_train = time.time()
    n_head_params = 0
    history       = []
    early_stopped = False
    sklearn_model = None
    pytorch_state = None
    pytorch_head_config = None

    if args.head in PYTORCH_HEADS:
        from featureflyers_shl.models.foundation import build_pytorch_head
        head = build_pytorch_head(
            args.head, X_tr.shape[1], N_CLASSES, args.hidden, args.dropout)
        n_head_params = sum(p.numel() for p in head.parameters())
        print(f"\nHead ({args.head}): {X_tr.shape[1]} → {N_CLASSES}  "
              f"({n_head_params:,} params)\n")

        best_state, best_f1, best_acc, history, early_stopped = _train_pytorch_head(
            head, X_tr, y_tr, X_va, y_va,
            args.epochs, args.batch_size, args.lr, args.dropout,
            args.patience, device,
        )
        total_time = time.time() - t_train

        # Final eval
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import f1_score
        head.load_state_dict(best_state)
        head.eval()
        val_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va).long())
        val_ld = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
        pb, lb = [], []
        with torch.no_grad():
            for xb, yb in val_ld:
                pb.append(head(xb.to(device)).argmax(1).cpu().numpy())
                lb.append(yb.numpy())
        final_preds  = np.concatenate(pb)
        final_labels = np.concatenate(lb)
        final_f1     = float(f1_score(final_labels, final_preds,
                                      average="macro", zero_division=0))
        final_acc    = float((final_preds == final_labels).mean())

        torch.save(best_state, run_dir / "head.pt")
        pytorch_state = best_state
        pytorch_head_config = dict(
            head_type=args.head,
            embed_dim=X_tr.shape[1],
            n_classes=N_CLASSES,
            hidden=args.hidden,
            dropout=args.dropout,
        )

    elif args.head in SKLEARN_HEADS:
        from sklearn.metrics import f1_score, classification_report
        sklearn_model, best_f1, best_acc = _train_sklearn_head(
            args.head, X_tr, y_tr, X_va, y_va, args.seed,
            xgb_device=args.xgb_device)
        total_time = time.time() - t_train

        final_preds  = sklearn_model.predict(X_va)
        final_labels = y_va
        final_f1     = float(f1_score(final_labels, final_preds,
                                      average="macro", zero_division=0))
        final_acc    = float((final_preds == final_labels).mean())

    else:
        raise ValueError(f"Unknown head {args.head!r}")

    from sklearn.metrics import classification_report, f1_score

    print(f"\nTraining done in {total_time:.1f}s")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    present = sorted(set(final_labels.tolist()))
    report  = classification_report(
        final_labels, final_preds,
        labels=present, target_names=[LABEL_MAP[l] for l in present],
        zero_division=0,
    )

    stat_desc = "(stat-only)" if args.stat_only else (
        "(hybrid: emb+stat)" if args.hybrid_stat_features else "(embedding only)")
    print(f"\n{'=' * 60}")
    print(f"  Encoder    : {'stat-only' if args.stat_only else args.encoder}"
          f"  {stat_desc}")
    print(f"  Strategy   : {args.embed_strategy if not args.stat_only else 'n/a'}")
    print(f"  Norm       : {args.norm}")
    print(f"  Fusion     : {args.fusion}  positions={positions_train}")
    print(f"  Head       : {args.head}")
    print(f"  Feature dim: {X_tr.shape[1]}")
    print(f"  Accuracy   : {final_acc:.4f}")
    print(f"  Macro-F1   : {final_f1:.4f}")
    delta = final_f1 - XGB_POOL_BASELINE_F1
    sign  = "+" if delta >= 0 else ""
    print(f"  vs XGB pool baseline F1=0.6389: {sign}{delta:.4f}")
    print(f"{'=' * 60}")
    print(report)

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------
    config = dict(
        encoder=args.encoder if not args.stat_only else "stat-only",
        is_placeholder=is_placeholder,
        embed_strategy=args.embed_strategy if not args.stat_only else "n/a",
        stat_only=args.stat_only,
        fusion=args.fusion,
        positions=positions_train,
        position_val=val_position,
        sample_limit=args.sample_limit,
        norm=args.norm,
        include_magnitude=args.include_magnitude,
        include_delta=args.include_delta,
        hybrid_stat_features=args.hybrid_stat_features,
        n_channels=n_channels,
        embed_dim=embed_dim,
        feature_dim=int(X_tr.shape[1]),
        n_head_params=n_head_params,
        head=args.head,
        hidden=args.hidden if args.head in PYTORCH_HEADS else None,
        epochs=args.epochs if args.head in PYTORCH_HEADS else None,
        lr=args.lr if args.head in PYTORCH_HEADS else None,
        dropout=args.dropout if args.head in PYTORCH_HEADS else None,
        patience=args.patience if args.head in PYTORCH_HEADS else None,
        seed=args.seed, device=str(device),
        n_train=len(X_tr), n_val=len(X_va),
    )
    metrics = dict(
        encoder=args.encoder if not args.stat_only else "stat-only",
        is_placeholder=is_placeholder,
        embed_strategy=args.embed_strategy if not args.stat_only else "n/a",
        stat_only=args.stat_only,
        hybrid_stat_features=args.hybrid_stat_features,
        fusion=args.fusion,
        positions=positions_train,
        norm=args.norm,
        include_magnitude=args.include_magnitude,
        include_delta=args.include_delta,
        n_channels=n_channels,
        embed_dim=embed_dim,
        feature_dim=int(X_tr.shape[1]),
        head=args.head,
        position=args.position,
        sample_limit=args.sample_limit if args.sample_limit else "full",
        best_val_macro_f1=round(final_f1, 4),
        best_val_accuracy=round(final_acc, 4),
        xgb_pool_baseline_f1=XGB_POOL_BASELINE_F1,
        delta_vs_baseline=round(final_f1 - XGB_POOL_BASELINE_F1, 4),
        total_time_s=round(total_time, 1),
        epochs_trained=len(history),
        early_stopped=early_stopped,
        history=history,
    )

    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "classification_report.txt").write_text(report)

    # Save complete model artifact
    _save_artifact(
        run_dir,
        head_type=args.head,
        sklearn_model=sklearn_model,
        pytorch_state=pytorch_state,
        pytorch_head_config=pytorch_head_config,
        encoder_name=args.encoder if not args.stat_only else "stat-only",
        embed_strategy=args.embed_strategy,
        norm=args.norm,
        include_magnitude=args.include_magnitude,
        include_delta=args.include_delta,
        hybrid_stat_features=args.hybrid_stat_features,
        stat_only=args.stat_only,
        fusion=args.fusion,
        positions=positions_train,
        n_channels=n_channels,
        embed_dim=embed_dim,
        feature_dim=int(X_tr.shape[1]),
        train_mu=train_mu,
        train_sd=train_sd,
        seed=args.seed,
        hdf5_path=str(args.hdf5_path),
    )

    print(f"\nSaved to {run_dir.relative_to(REPO_ROOT)}/")
    print("  config.json  metrics.json  classification_report.txt  "
          "head.pt / head_sklearn.pkl / model.joblib")


if __name__ == "__main__":
    main()
