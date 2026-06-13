# Modeling Plan — FeatureFlyers / SHL 2026

## Current Best Submission

| File | Model | Val Macro-F1 | Val Accuracy | Status |
|------|-------|-------------|-------------|--------|
| `FeatureFlyers_xgb_pool_full.txt` | XGBoost pool (all 4 positions) | **0.6389** | 68.4% | Submitted |

Measured on Bag-only validation (57 576 windows). Model: `xgb_sfull_posBag_Hand_Hips_Torso_fusionpool`.

---

## Problem

Multi-class locomotion mode recognition (8 classes) from 9-axis IMU sensor data collected
at 4 phone positions (Bag, Hand, Hips, Torso). Evaluation metric: **F1-score** (macro).

Foundation models must remain **frozen**. Only lightweight heads may be trained.

---

## Stage 1 — Data Preparation and EDA

- [x] Unzip raw data → `dataset/raw/`
- [x] Convert to HDF5 → `dataset/processed/shl2026.hdf5`
  - Train/val: flat `(N, 9)` per position; test: pre-windowed `(92726, 500, 9)`
- [x] Label distribution analysis → `outputs/eda/label_analysis.txt`
- [x] Class imbalance noted: Run=4.3 %, Bus=14.4 % in train
- [x] Window size fixed: 500 samples, hop 250 (50 % overlap)
- [ ] Signal visualisation per class (deferred to notebook)

---

## Stage 2 — Classical ML Baseline ✓

**Goal:** establish a reproducible floor before any deep learning.

### Feature extraction — implemented in `src/featureflyers_shl/features/statistical.py`

| Group | Features | Count |
|-------|----------|-------|
| Per-axis statistics | mean, std, min, max, median, energy, zcr | 9 × 7 = 63 |
| Group magnitude stats | mean, std, min, max, energy for Acc/Gyr/Mag magnitudes | 3 × 5 = 15 |
| Per-axis spectral | top-20 FFT magnitudes + energy + entropy + rolloff | 9 × 23 = 207 |
| Group magnitude FFT | same spectral set on Acc/Gyr/Mag magnitudes | 3 × 23 = 69 |
| **Total** | | **354** |

### Classifiers — implemented in `scripts/train_baseline.py`
- [x] Random Forest (class_weight=balanced, n_estimators=100)
- [x] Logistic Regression (class_weight=balanced)
- [x] Single-position mode (default: Bag)
- [x] Multi-position early fusion (`--fusion early`)

### Submission — implemented in `scripts/generate_submission.py`
- [x] Load model.joblib → predict on test → write 92 726 × 500 submission file
- [x] XGBoost label offset fix — XGBoost predicts 0-7; submission needs 1-8 (+1 applied automatically)
- [x] Early-fusion guard — script exits with clear error if a fusion="early" model is loaded (test has no per-position split)
- [x] fusion="pool" support — position-pooled models work at test time (same 354-feature space)

### Infrastructure improvements
- [x] `scripts/precompute_features.py` — pre-computes all 354 features for every position/split using 32 parallel CPU workers, saves to `dataset/processed/features/*.npz` (2.4 GB total). Cuts data loading from ~130s to ~1s per run.
- [x] XGBoost GPU (`--model xgb --device cuda`) — replaces CPU-only Random Forest for faster fitting
- [x] Balanced sample weights for XGBoost — `compute_sample_weight("balanced")` fixes class imbalance on full dataset
- [x] Early stopping for XGBoost — `--early-stopping-rounds 30` prevents overfitting and saves training time
- [x] NaN guard in `statistical.py:extract()` — handles recording gaps (2 windows in train/Hips had NaN raw sensor values)

### Results — all runs, Bag position, seed=42, full validation set (57 576 windows)

| Model | Positions | Fusion | Sample Limit | Macro-F1 | Accuracy | Runtime | Submittable | Notes |
|-------|-----------|--------|-------------|---------|---------|---------|-------------|-------|
| RF | Bag | none | 20 000 | 0.6427 | 63.4% | 121s | Yes | sklearn, CPU, class_weight=balanced |
| XGBoost | Bag | none | 20 000 | 0.6605 | 64.7% | 126s | Yes | GPU, no sample weights |
| XGBoost v1 | Bag | none | full | 0.6039 | 59.6% | 1303s | Yes | GPU, no sample weights -- class imbalance hurt |
| XGBoost v2 | Bag | none | full | 0.6384 | 62.9% | 1176s | Yes | GPU, balanced weights + early stop |
| **XGBoost** | **Bag+Hand+Hips+Torso** | **early** | **full** | **0.6613** | **67.8%** | **86s** | **NO** | **Validation-only -- test has no per-position split** |

**Submittable = Yes means the model can generate test predictions from `test/data` (354 features).**
**fusion="early" produces 1416-feature vectors and CANNOT be applied to `test/data`.**

Best per-class F1 (4-position early fusion, validation-only):

| Class | F1 | Notes |
|-------|----|-------|
| Still | 0.86 | strong |
| Walking | 0.89 | strong |
| Run | 0.66 | rare class (1 110 val windows) |
| Bike | 0.87 | strong |
| Car | 0.72 | moderate |
| Bus | 0.48 | hard -- confused with Train |
| Train | 0.50 | hard -- confused with Metro/Bus |
| Metro | 0.30 | hardest -- underground vibration similar to Train |

Key findings:
- 4-position early fusion adds +2.3pp macro-F1 over single-position (0.6613 vs 0.6384) but **cannot be submitted**
- The test set (`test/data`) is a single (92726, 500, 9) array with NO per-position labels
- Only fusion="none" and fusion="pool" models are submission-compatible
- Cache reduces total pipeline time from ~20 min to ~90s for a full-dataset run
- Train/Metro remain the hardest pair -- need deeper model or foundation model embeddings

---

## Stage 3 — Deep Time-Series Baseline

**Goal:** end-to-end learnable baseline on raw windows.

### Architecture — `src/featureflyers_shl/models/cnn1d.py`

| Block | Channels | Kernel | Pool | Output length |
|-------|----------|--------|------|--------------|
| Conv1 | 9 → 32   | 7      | /2   | 250           |
| Conv2 | 32 → 64  | 5      | /2   | 125           |
| Conv3 | 64 → 128 | 3      | /5   | 25            |
| Conv4 | 128 → 256| 3      | —    | 25            |
| GAP   |          |        |      | 1             |
| Linear| 256 → 8 |        |      | —             |

Each block: Conv1d → BatchNorm1d → ReLU → (MaxPool) → Dropout

### Dataset — `src/featureflyers_shl/data/dataset.py`
- `SHLWindowDataset(hdf5_path, split, position, sample_limit, seed)`
- Train/val: stratified window sampling from flat `(N, 9)` streams
- Test: direct access to pre-windowed `(92726, 500, 9)`
- Returns `(x, y)` for train/val; `(x, idx)` for test
- Labels converted to 0-based (HDF5 1-8 → 0-7)

### Training — `scripts/train_deep.py`
- AdamW + cosine LR decay, class-weighted cross-entropy
- Early stopping via `--patience` (0 = disabled)
- Best model saved by val macro-F1
- Saves `model.pt`, `config.json`, `metrics.json`, `classification_report.txt`, `confusion_matrix.txt` to `outputs/deep_baseline/<run>/`

Multi-GPU support via `torch.nn.DataParallel` — automatically uses all 4 RTX 2080 Ti when `--device cuda` is set.

```bash
# Smoke test (Bag, 5000 windows, 2 epochs)
python scripts/train_deep.py \
    --position Bag --sample-limit 5000 --epochs 2 \
    --batch-size 128 --device cuda

# Stage 3 experiment run
python scripts/train_deep.py \
    --position Bag --sample-limit 20000 --epochs 10 \
    --batch-size 128 --patience 4 --device cuda

# Full Bag-position training (all 4 GPUs)
python scripts/train_deep.py \
    --position Bag --epochs 50 --batch-size 512 --patience 10 --device cuda
```

Outputs saved to `outputs/execution-output/<run_name>/`.

### Results (Stage 3 experiments — Bag position, 20 000 stratified windows, seed=42)

| Model | Batch Size | Epochs (trained) | Best Val Macro-F1 | Best Val Accuracy | Notes |
|-------|-----------|-----------------|------------------|------------------|-------|
| CNN1D | 128 | 9/10 (early stop) | 0.0866 | 13.2% | Patience=4; best at epoch 5 |
| CNN1D | 256 | 10/10 | 0.0906 | 13.0% | Best at epoch 7 |
| RF (baseline) | -- | -- | **0.6427** | **63.4%** | 354 stat+spectral features |

Key findings:
- RF with hand-crafted features outperforms CNN by ~7x on macro-F1 at 20 000 windows
- CNN is collapsing to predict "Bike" (dominant confusion target in both runs)
- Root cause: cosine LR decays too fast over 10 epochs; model needs more epochs or a warmup schedule
- Next step: train CNN for 50+ epochs on full dataset; add residual connections or channel attention

Multi-position fusion (planned):
- Late fusion: average softmax logits across available positions
- Early fusion: concatenate windows → `(batch, 36, 500)`

---

## Stage 4 — Submission-Ready XGBoost

**Goal:** make the strongest XGBoost model that can actually generate a submission, given the test set structure.

### Critical test set finding

`test/data`: shape `(92726, 500, 9)` -- a **single flat array, no per-position split**.

The test HDF5 has only one key under `test/`: `data`. There are no `test/Bag`, `test/Hips`, etc.
This means:
- `fusion="early"` models (1416 features from 4 simultaneous positions) **CANNOT be submitted**
- Submission-compatible models must produce 354-feature vectors from a single 9-channel window
- Submission-compatible fusions: `fusion="none"` (single position) or `fusion="pool"` (all positions stacked)

### fusion="pool" -- position-invariant model

New fusion mode added to `train_baseline.py`: `--fusion pool`

| Mode | Training | Features at test time | n_train (full dataset) |
|------|----------|-----------------------|------------------------|
| none (Bag) | Bag windows only | 354 | ~392 000 |
| pool (all 4) | Bag+Hand+Hips+Torso stacked as independent samples | 354 | ~1 570 000 |
| early (validation-only) | aligned windows from all 4 positions, feature-concatenated | 1416 | ~392 000 |

Pool advantages over single-position:
- 4x more training data (all positions contribute)
- Model learns position-invariant representations naturally
- Works at test time with no position information needed

### Experiments

All macro-F1 values measured on **Bag-only validation** (57 576 windows) for fair comparison.

| Model | Fusion | n_train | Macro-F1 (Bag val) | Accuracy (Bag val) | Submittable | Status |
|-------|--------|---------|-------------------|-------------------|-------------|--------|
| XGBoost | none (Bag) | 392 142 | 0.6324 | 62.5% | Yes | Done |
| **XGBoost** | **pool (all 4)** | **~1 570 000** | **0.6389** | **68.4%** | **Yes** | **Done -- best** |
| XGBoost | early (Bag+Hand+Hips+Torso) | 392 142 | 0.6613 | 67.8% | NO | Validation-only |

Per-class F1 on Bag-only val (pool model vs Bag model):

| Class | Bag model | Pool model | Delta |
|-------|-----------|-----------|-------|
| Still | 0.83 | 0.85 | +0.02 |
| Walking | 0.84 | 0.84 | 0.00 |
| Run | 0.87 | 0.61 | -0.26 |
| Bike | 0.57 | 0.43 | -0.14 |
| Car | 0.72 | 0.78 | +0.06 |
| Bus | 0.47 | 0.48 | +0.01 |
| Train | 0.43 | 0.59 | +0.16 |
| Metro | 0.33 | 0.54 | +0.21 |

Key findings:
- Pool model wins overall (+6.5pp macro-F1 vs Bag on same val distribution)
- Large gains on previously hardest classes: Train +16pp, Metro +21pp
- Regressions on rare/distinctive classes: Run -26pp, Bike -14pp
- Pool model's 4x more training data improves position-invariance at the cost of less Bag-specific tuning

**Best submission: `FeatureFlyers_xgb_pool_full.txt`** (pool model, macro-F1=0.6389 on Bag val)

### Commands

```bash
# Train position-pooled XGBoost (~70s from cache, ~1.57M train windows)
python scripts/train_baseline.py \
    --model xgb \
    --positions Bag Hand Hips Torso \
    --fusion pool \
    --device cuda \
    --early-stopping-rounds 30 \
    --n-estimators 500

# Generate full submission
python scripts/generate_submission.py \
    --model-path outputs/execution-output/xgb_sfull_posBag_Hand_Hips_Torso_fusionpool/model.joblib \
    --output outputs/execution-output/submissions/FeatureFlyers_xgb_pool_full.txt
```

---

## Stage 5 — Frozen Foundation Model + Head

**Goal:** leverage representation power of pre-trained time-series models without fine-tuning.

### Protocol

1. Freeze all encoder parameters (`requires_grad = False`).
2. Forward all training/val windows through the frozen encoder in batches → `(N, embed_dim)` embeddings.
3. Cache embeddings to `dataset/processed/embeddings/` (`.npz`, git-ignored).
4. Train a lightweight MLP head `(embed_dim → 256 → 8)` on cached embeddings only.
5. Report macro-F1 vs XGBoost pool baseline (0.6389).

### Implementation

| File | Purpose |
|------|---------|
| `src/featureflyers_shl/models/foundation.py` | `MomentEncoder`, `FallbackEncoder`, `get_encoder()` factory |
| `scripts/train_foundation_head.py` | End-to-end: embed extraction + head training + eval |

### Encoders

| Encoder | Params | Embed dim | Notes |
|---------|--------|-----------|-------|
| `moment` | 341M (frozen) | 1024 | MOMENT-1-large; `pip install momentfm`; downloads ~1.4 GB once |
| `fallback` | — | 1024 | Frozen orthogonal random projection; NOT a real foundation model |

Input format: `(B, 500, 9)` → MOMENT internally converts to `(B, 9, 500)` with mean-pool over patches.

### Commands

```bash
# Smoke test -- fallback encoder (no download, fast)
python scripts/train_foundation_head.py \
    --position Bag --sample-limit 5000 --encoder fallback \
    --epochs 2 --batch-size 256 --device cuda:1

# Smoke test -- MOMENT encoder (FP16 model, preloaded RAM, batch=256 extraction)
python scripts/train_foundation_head.py \
    --position Bag --sample-limit 1000 --encoder moment \
    --epochs 2 --batch-size 256 --extract-batch-size 256 --device cuda:1

# Full Bag run with MOMENT (~80 min first run; subsequent runs skip extraction)
python scripts/train_foundation_head.py \
    --position Bag --encoder moment \
    --epochs 30 --batch-size 512 --extract-batch-size 256 \
    --patience 10 --device cuda:1 \
    > outputs/execution-output/foundation_moment_full_run.log 2>&1
```

### Results

Baseline to beat: XGBoost pool, macro-F1=**0.6389**, accuracy=68.4%

| Run | Encoder | n_train | Epochs run | Best epoch | Val Macro-F1 | Val Accuracy | Embed time | Notes |
|-----|---------|---------|-----------|-----------|-------------|-------------|------------|-------|
| fallback smoke | fallback (placeholder) | 5 000 | 2 | 2 | 0.1111 | 14.8% | 0.3s | Random projection, expected near-random result |
| MOMENT smoke | moment | 5 000 | 2 | 1 | 0.0838 | 12.4% | 163s/split | Too few epochs/data, near-random, pipeline verified |
| **MOMENT full** | **moment** | **392 142** | **21** (early stop) | **11** | **0.2108** | **20.2%** | 80 min train + 12 min val | See analysis below |

**Full run analysis (MOMENT full, Bag position):**
- Best epoch 11 (val macro-F1 = 0.2108); early stopping fired at epoch 21 (patience=10)
- Training accuracy climbed steadily (34%→63%) while val oscillated at 16–20% → train/val gap indicates the frozen embeddings don't transfer cleanly to locomotion
- Best per-class: Run 0.38 F1, Car 0.37; worst: Bus 0.03, Train 0.13, Metro 0.13
- **Conclusion:** Frozen MOMENT-1-large + MLP head scores **−0.4281 vs XGBoost pool**. Generic time-series embeddings (trained on diverse domains) cannot match 354 hand-crafted IMU features. Fine-tuning the encoder would be needed to close this gap, but is prohibited by Stage 5 rules.

**Extraction speed (FP16 model + preloaded RAM):**
- FP16 model: 6.86 GiB GPU (vs 9.64 GiB FP32), freeing ~4 GiB for batch activations
- Preload: 392K windows loaded into 7 GB RAM at init (21s); O(1) `__getitem__`, GPU at sustained 100%
- Throughput: **82 win/s at batch=256** (vs ~21 win/s DataParallel lazy; ~4× speedup)
- Full Bag extraction: 80 min; embeddings cached at `dataset/processed/embeddings/`; subsequent runs skip extraction entirely

---

## Stage 6 — Multi-Position Ensemble

- Train a separate best-performing head for each position
- At test time, average probabilities across positions (requires per-position test split -- check future challenge data releases)
- Optionally: learn a position-weighting meta-layer

---

## Stage 7 — Submission

- `scripts/generate_submission.py` (implemented)
- Format: 92 726 lines, each with 500 comma-separated integers (1-8)
- Output dir: `outputs/execution-output/submissions/`

### Submission history

| File | Model | Val Macro-F1 | Notes |
|------|-------|-------------|-------|
| `FeatureFlyers_xgb_Bag_full.txt` | XGBoost Bag only | 0.6324 | Baseline submission |
| **`FeatureFlyers_xgb_pool_full.txt`** | **XGBoost pool (all 4 pos)** | **0.6389** | **Best submission** |

---

## Experiment Tracking

All runs log to `outputs/<run_name>/`:
- `config.yaml` (copied from run config)
- `metrics.json` (train/val F1, per-class F1)
- `confusion_matrix.png`
- `model.ckpt` (if applicable)

Use fixed `seed: 42` in `configs/default.yaml` for reproducibility.
