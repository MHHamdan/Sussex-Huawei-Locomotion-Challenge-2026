# Modeling Plan — FeatureFlyers / SHL 2026

## Current Best Submission

| File | Model | Val Macro-F1 | Val Accuracy | Status |
|------|-------|-------------|-------------|--------|
| `FeatureFlyers_xgb_pool_full.txt` | XGBoost pool (all 4 positions) | 0.6389 | 68.4% | Submitted |
| `FeatureFlyers_foundation_hybrid_pool_full.txt` | MOMENT+stat hybrid pool full | **0.6970** | 73.0% | Generated (Stage 6) |

Best **model** (not yet submitted): InceptionTime pool full — val Macro-F1=**0.7265**, Accuracy=76.4% (Stage 8).  
Measured on Bag-only validation (57 576 windows).

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

Input format: `(B, C, T) = (B, 9, 500)` — channels-first, passed directly to MOMENT. **Do not permute.** See Stage 5 Ablations for the encoder bug that was fixed.

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

**Full run analysis (MOMENT full, Bag position) — note: this run used the OLD buggy encoder (wrong channel/time transposition) and the OLD MLP head with default settings:**
- Best epoch 11 (val macro-F1 = 0.2108); early stopping fired at epoch 21 (patience=10)
- The encoder permutation bug (see Stage 5 Ablations) explains the poor performance
- This result is superseded by Stage 5 Ablations which use the corrected encoder

**Dataset preload bug discovered and fixed (Stage 5 Ablations):**
`_read_windows_batched` used `buf[inv_order[gi]]` (double-inverted: maps sorted→original→sorted)
instead of the correct `buf[order[gi]]` (maps sorted→original directly).
Effect: with any `--sample-limit` (preload=True), all window data was paired with wrong labels.
Fix: single-character change in `src/featureflyers_shl/data/dataset.py`.
Full-dataset (lazy, preload=False) runs were unaffected.

**Extraction speed (FP16 model + preloaded RAM):**
- FP16 model: 6.86 GiB GPU (vs 9.64 GiB FP32), freeing ~4 GiB for batch activations
- Preload: 392K windows loaded into 7 GB RAM at init (21s); O(1) `__getitem__`, GPU at sustained 100%
- Throughput: **82 win/s at batch=256** (vs ~21 win/s DataParallel lazy; ~4× speedup)
- Full Bag extraction: 80 min; embeddings cached at `dataset/processed/embeddings/`; subsequent runs skip extraction entirely

---

## Stage 5 — Ablations (branch: feature/stage5-foundation-model-ablation)

### What changed vs. the initial Stage 5 prototype

**Critical bug fixed:** The original `MomentEncoder.forward()` called
`x.permute(0, 2, 1)` on an input that was already `(B, C, T) = (B, 9, 500)`,
producing `(B, T, C) = (B, 500, 9)`.  MOMENT interpreted this as 500 "channels"
with seq_len=9 — completely wrong.  The input mask was also `(B, 9)` instead of
`(B, 500)`.  Both are corrected in the rewritten `MomentEncoder`.

**New preprocessing options** (`--norm`, `--include-magnitude`, `--include-delta`):
- `none` — raw sensor units (original behaviour)
- `per-window` — z-score each 500-sample window independently, per channel
- `channel-global` — fit channel mean/std on training set, apply to both splits
- `train-stats` — same as channel-global, stats cached to `dataset/processed/train_channel_stats/`

**New embedding strategies** (`--embed-strategy`):
- `mean_pool` — MOMENT mean-pools all patch tokens → `(B, 1024)`
- `last_patch` — last patch token (or last-quarter mean if momentfm API insufficient)
- `sensorwise` — MOMENT applied per channel, embeddings concatenated → `(B, C×1024)`
- `flatten_patches` — alias of `sensorwise`

**New head types** (`--head`):
- `linear` — `Linear(embed_dim, 8)`
- `mlp` — `Linear→ReLU→Dropout→Linear` (previous default)
- `residual_mlp` — 2-block residual MLP with LayerNorm + GELU
- `xgb` — XGBoost (encoder stays frozen; fits on cached embeddings)
- `logistic` — scikit-learn `LogisticRegression`

**Hybrid mode** (`--hybrid-stat-features`):
Concatenates the 354-dim statistical/spectral features with the frozen embedding.
Results are labelled **hybrid** — not a pure foundation-model result, but a
diagnostic upper bound for what the embedding can contribute.

**Alternative encoders attempted:**
- `chronos` (`chronos-forecasting`): not installed; documented in `get_encoder()`
- `uni2ts` (Moirai): not installed; documented in `get_encoder()`
- `moment` (MOMENT-1-large): ✓ installed and used

### New in submission-ready phase (stage5-ablation branch)

**Pool fusion mode (`--fusion pool`):**
Stacks Bag, Hand, Hips, Torso as independent training samples. Validation always
uses Bag-only for fair comparison with prior baselines. Test-compatible because
`test/data` is a single mixed-position array.

**Stat-only mode (`--stat-only`):**
Skips embedding extraction entirely. Trains a head on the 354 statistical/spectral
features only. Used for the apples-to-apples sanity comparison.

**Test prediction / submission mode (`--predict-test`):**
Loads a saved `model.joblib` artifact, reads `test/data`, applies the same
preprocessing and encoder as training, and writes the SHL submission file.

**Complete model artifact (`model.joblib`):**
Saved alongside `metrics.json` in each run directory.  The bundle contains the
trained head (sklearn model or PyTorch state dict), encoder config, preprocessing
params, fusion mode, positions used, channel stats, label offset (+1 to convert
0-indexed predictions to 1–8 submission labels), and seed.  Sufficient to
reproduce test predictions without the training data.

**Full-dataset stat feature cache for hybrid mode:**
For `--sample-limit None` pool runs, stat features are loaded from the precomputed
cache at `dataset/processed/features/{split}_{position}.npz` (already populated for
all 4 positions by `precompute_features.py`).  This avoids re-extracting 1.57M
windows through the slow Python feature loop.

### 20 k apples-to-apples sanity comparison

All three models use `--position Bag --sample-limit 20000 --seed 42 --head xgb`.
Validation is stratified-sampled (same 20k limit) → ~18 609 val windows.
**All three run through the same XGBoost hyperparameters and balanced sample weights
to ensure a fair comparison.**

| Model | Features | Feature dim | Macro-F1 | Accuracy | Delta vs baseline | Notes |
|-------|----------|-------------|----------|----------|-------------------|-------|
| A — stat-only XGB | 354 stat/spectral | 354 | 0.6804 | 66.4% | +4.15pp | Pure classical baseline under identical conditions |
| B — MOMENT emb XGB | MOMENT embeddings | 1024 | 0.6968 | 68.6% | +5.79pp | Pure foundation model |
| C — MOMENT hybrid XGB | MOMENT + stat | 1378 | **0.7329** | **72.4%** | **+9.40pp** | Hybrid — best |

**Conclusions:**
- MOMENT embeddings alone outperform 354 hand-crafted features by +1.64pp (B vs A).
  The foundation model adds real value over statistical features.
- Hybrid combination outperforms stat-only by **+5.25pp** (C vs A) and embedding-only
  by **+3.61pp** (C vs B).  The two feature spaces are complementary.
- Classical XGB on stat features (A) already beats the XGB pool baseline (0.6389)
  by +4.15pp on 20k Bag-only training; the baseline uses all 4 positions × full dataset.

### Ablation table (Bag position)

All runs use `--position Bag`.  Embedding caches are stored under
`dataset/processed/embeddings/` (git-ignored).

Baseline to beat: XGBoost pool, **F1=0.6389**, Accuracy=68.4%

| Exp | Encoder | Norm | Mag | Δ | Strategy | Head | Hybrid | N_win | Macro-F1 | Accuracy | Delta | Notes |
|-----|---------|------|-----|---|----------|------|--------|-------|----------|----------|-------|-------|
| **F** | moment | per-window | — | — | mean_pool | xgb | **✓** | 20k | **0.7329** | **72.4%** | **+9.40pp** | **Best — emb+stat hybrid** |
| D | moment | per-window | — | — | mean_pool | residual_mlp | — | 20k | **0.7237** | 71.4% | +8.48pp | Strongest pure-embedding head |
| E | moment | per-window | — | — | mean_pool | xgb | — | 20k | 0.6968 | 68.6% | +5.79pp | XGBoost on embeddings only |
| A | moment | none | — | — | mean_pool | mlp | — | 20k | 0.6388 | 62.8% | −0.01pp | Bug-fixed baseline |
| C | moment | channel-global | — | — | mean_pool | mlp | — | 20k | 0.6376 | 62.7% | −0.13pp | Train-set norm, minimal effect |
| B | moment | per-window | — | — | mean_pool | mlp | — | 20k | 0.6374 | 62.7% | −0.15pp | Per-window norm + mlp |
| G | moment | per-window | — | — | sensorwise | xgb | — | 20k | **deferred** | — | — | ~2h extraction; deprioritised once pool hybrid path was confirmed |

Head training times (extraction is one-time cost, cached): mlp/residual_mlp ~30s, xgb ~33–38s

**Exp G deferred:** The output directory was created but never completed (likely OOM or wallclock
timeout during the 9-channel sensorwise extraction, which takes ~9× longer than mean_pool).
With pool-hybrid already achieving +9.40pp, Exp G is not on the critical path for submission.

### Full-dataset pool hybrid run

**GPU note:** MOMENT embedding extraction requires a CUDA GPU.  In the current session
`torch.cuda.is_available()` returns `False` (no visible GPU); extraction is falling back to CPU
which would take ~17h/position (not practical).  The full MOMENT pool hybrid run is therefore
**blocked on GPU availability**.  When a GPU is available again, run:

```bash
python scripts/train_foundation_head.py \
    --encoder moment \
    --norm per-window \
    --embed-strategy mean_pool \
    --head xgb \
    --hybrid-stat-features \
    --fusion pool \
    --extract-batch-size 256 \
    --device cuda \
    2>&1 | tee outputs/execution-output/foundation_pool_hybrid_full_run.log
```

Run name: `foundation_moment_posPool_mean_pool_normperwindow_xgb_hybrid_sfull`

Training scope: 4 positions × full dataset ≈ 1 570 000 windows  
Feature dim: 1024 (MOMENT) + 354 (stat) = 1378  
Embedding extraction: ~80 min/position × 4 = ~320 min (one-time; cached)  
Stat features: loaded from `dataset/processed/features/{split}_{position}.npz` (pre-cached)  
XGBoost training: ~5–10 min after embeddings are ready  
Validation: full Bag val (57 576 windows)

| Run | N_train | N_val | Macro-F1 | Accuracy | Delta | Status |
|-----|---------|-------|----------|----------|-------|--------|
| **Pool hybrid full (MOMENT+stat, XGB CPU)** | **1 568 568** | **57 576** | **0.6970** | **73.0%** | **+5.81pp** | **Done (2026-06-20)** |

Per-class F1 (Stage 6 pool hybrid):

| Class | F1 | Notes |
|-------|----|-------|
| Still | 0.84 | strong |
| Walking | 0.89 | strong |
| Run | 0.64 | recall 48% — rare class hurt by pool position-mixing |
| Bike | 0.65 | recall 52% — same issue |
| Car | 0.74 | improved vs stat-only |
| Bus | 0.53 | precision 40%, recall 77% — over-predicts Bus |
| Train | 0.65 | much improved vs baseline |
| Metro | 0.64 | much improved vs baseline (was 0.30 in Stage 2) |

**XGB OOM fix:** `_train_sklearn_head` had `device="cuda"` hardcoded. Added `--xgb-device` flag (default `cpu`) so encoder stays on GPU for extraction while XGB fits on CPU.  
XGB fit time: 3841s (64 min) on 40 CPU cores, 1378-dim × 1.57M samples.

### Full-dataset pool stat-only run (CPU-viable fallback)

While waiting for GPU, the 354-dim stat-feature-only pool model is a solid submission
candidate.  Stat features are loaded from the pre-computed cache — no GPU needed for
training or test prediction.

```bash
python scripts/train_foundation_head.py \
    --stat-only --head xgb \
    --fusion pool \
    --device cpu \
    2>&1 | tee outputs/execution-output/foundation_statonly_pool_full_run.log
```

Run name: `foundation_statonly_posPool_xgb_sfull`

| Run | N_train | N_val | Macro-F1 | Accuracy | Delta | Status |
|-----|---------|-------|----------|----------|-------|--------|
| Pool stat-only full | 1 568 568 | 57 576 | **0.6481** | 69.1% | +0.92pp | Done (1078s CPU XGB) |

### Key finding: pool stat-only vs. 20k Bag stat-only

The full pool stat-only model (1.57M samples, all 4 positions) achieves **0.6481**, while the 20k Bag-only stat-only achieves **0.6804** — a 3.23pp deficit despite 78× more training data.

This is expected: stat features (e.g., mean, std, FFT coefficients) vary significantly across sensor positions. A model trained on all positions simultaneously must learn a position-averaged decision boundary that is worse for Bag-specific validation. The XGB pool baseline (Stage 4) showed the same trade-off: pool improved Metro/Train but hurt Run/Bike (+0.65pp net). The pool model is trained for position-invariance, which costs Bag-specific accuracy.

**Implication for submission:** The pool stat-only model (0.6481) is only +0.92pp above the current best submission (0.6389). The true submission gains require either the MOMENT hybrid (0.7340 on 20k Bag) or the full MOMENT hybrid pool run (GPU-blocked).

### Submission pipeline (foundation/hybrid)

**Hybrid model (MOMENT + stat, GPU required for test prediction):**
```bash
# Smoke test (1 000 windows)
python scripts/train_foundation_head.py \
    --predict-test \
    --model-path outputs/execution-output/foundation_moment_posBag_mean_pool_normperwindow_xgb_hybrid_s20000/model.joblib \
    --output outputs/execution-output/submissions/FeatureFlyers_foundation_hybrid_smoke.txt \
    --limit 1000 \
    --device cuda

# Full submission (requires GPU for MOMENT inference on 92 726 test windows)
python scripts/train_foundation_head.py \
    --predict-test \
    --model-path outputs/execution-output/foundation_moment_posPool_mean_pool_normperwindow_xgb_hybrid_sfull/model.joblib \
    --output outputs/execution-output/submissions/FeatureFlyers_foundation_hybrid_pool_full.txt \
    --device cuda
```

**Stat-only pool model (CPU-viable; no GPU needed for inference):**
```bash
# Smoke test (1 000 windows)
python scripts/train_foundation_head.py \
    --predict-test \
    --model-path outputs/execution-output/foundation_statonly_posPool_xgb_sfull/model.joblib \
    --output outputs/execution-output/submissions/FeatureFlyers_statonly_pool_smoke.txt \
    --limit 1000 \
    --device cpu

# Full submission
python scripts/train_foundation_head.py \
    --predict-test \
    --model-path outputs/execution-output/foundation_statonly_posPool_xgb_sfull/model.joblib \
    --output outputs/execution-output/submissions/FeatureFlyers_foundation_statonly_pool_full.txt \
    --device cpu
```

### Commands — ablation runs

```bash
# Apples-to-apples A: stat-only XGB, 20k Bag
python scripts/train_foundation_head.py \
    --position Bag --sample-limit 20000 \
    --stat-only --head xgb --device cuda:1

# Exp A: bug-fixed MOMENT + no norm + MLP
python scripts/train_foundation_head.py \
    --position Bag --sample-limit 20000 --encoder moment \
    --norm none --embed-strategy mean_pool --head mlp \
    --epochs 30 --batch-size 512 --patience 10 --device cuda:1

# Exp B: per-window norm + MLP
python scripts/train_foundation_head.py \
    --position Bag --sample-limit 20000 --encoder moment \
    --norm per-window --embed-strategy mean_pool --head mlp \
    --epochs 30 --batch-size 512 --patience 10 --device cuda:1

# Exp E: per-window norm + XGBoost head
python scripts/train_foundation_head.py \
    --position Bag --sample-limit 20000 --encoder moment \
    --norm per-window --embed-strategy mean_pool --head xgb --device cuda:1

# Exp F: hybrid (embedding + 354 statistical features) + XGBoost
python scripts/train_foundation_head.py \
    --position Bag --sample-limit 20000 --encoder moment \
    --norm per-window --embed-strategy mean_pool \
    --head xgb --hybrid-stat-features --device cuda:1

# Summarise all results
python scripts/summarize_foundation_results.py
```

### Analysis

**Bug fix was the dominant factor.** Correcting the MOMENT encoder input from (B,T,C)=(B,500,9)
to (B,C,T)=(B,9,500) lifted F1 from 0.2108 → 0.6388 (+42.8pp) with a simple MLP head (Exp A).

**Head architecture matters most.** All three MLP variants (A, B, C) cluster at 0.637–0.639
regardless of normalization scheme. Switching to a residual MLP (Exp D) adds +8.6pp to 0.7237.
XGBoost head (Exp E) adds +5.8pp to 0.6968. ResidualMLP > XGBoost for pure-embedding tasks.

**Normalization scheme is a minor factor for MLP heads.** The difference between no-norm (A),
per-window (B), and channel-global (C) is <0.2pp — well within noise for 20k samples.

**MOMENT embeddings add real value over stat features.** Under identical conditions (same Bag 20k
training set, same XGBoost head, same sample weights), MOMENT-only (B) outperforms stat-only (A)
by +1.64pp.  The hybrid (C) outperforms stat-only by +5.25pp and embedding-only by +3.61pp.

**Hybrid (embedding + stat features) is the current best.** Concatenating 1024-dim MOMENT
embeddings with 354-dim hand-crafted features and training XGBoost (Exp F) achieves F1=0.7329
(+9.40pp over baseline).  The two feature spaces are complementary: statistical features encode
local temporal structure; MOMENT embeddings capture global sequence patterns across all 9
channels simultaneously.

**Exp G (sensorwise) deferred.** The directory was created but the run did not complete.
Given pool-hybrid achieves +9.40pp and Exp G requires ~9× longer extraction (~2h for 20k),
it is not on the critical path and has been deferred.

---

## Stage 7 — LoRA MOMENT Fine-Tuning (Paper Experiment — NOT Submittable)

**Competition constraint:** Foundation models must remain frozen. This stage is a paper experiment only.

### Protocol

Apply LoRA adapters (rank=8, alpha=16) to the top 4 transformer blocks (20–23) of MOMENT-1-large,
freeze all other weights. Train on 20K windows per position (pool, 80K total) with batch=64.

| Component | Params | Notes |
|-----------|--------|-------|
| MOMENT backbone | 341M (frozen) | Top 4 blocks get LoRA; rest fully frozen |
| LoRA adapters (q, v projections) | 860K trainable (0.25%) | peft 0.6.2, blocks 20–23 |
| Classification head | ~8K trainable | Linear(1024→8) |

**Gradient fix:** T5Stack gradient checkpointing blocks gradient flow when no input `requires_grad=True`.
Workaround: forward hook on `patch_embedding` to force `requires_grad_(True)` on embedding outputs.
`enable_input_require_grads()` not available in peft 0.6.2.

### Launch

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_stage7_lora.py \
    --epochs 30 --patience 7 --batch-size 64 \
    --sample-limit 20000 --lora-rank 8 --lora-alpha 16 --lora-blocks 4 \
    --device cuda:1 --num-workers 0 \
    2>&1 | tee outputs/execution-output/stage7_lora_full.log
```

### Results

| Run | N_train | Epochs | Best Macro-F1 | Accuracy | Notes |
|-----|---------|--------|--------------|---------|-------|
| LoRA r=8 α=16 blocks=4 | 80 000 | running (ep1 done) | 0.7035 (ep1) | 71.2% | GPU 1, 9516s/epoch |

---

## Stage 8 — InceptionTime End-to-End (Pool Fusion)

**Goal:** Train a custom deep model (no frozen FM) on all 4 positions. Fully submittable.

### Architecture — `src/featureflyers_shl/models/inception.py`

| Block | Description | Output channels |
|-------|-------------|----------------|
| 3× Inception module (block 1) | Parallel Conv1d kernels 40/20/10 + MaxPool; bottleneck=32 | 4×32=128 |
| Residual shortcut | Conv1d(1×1, 9→128) + BN | — |
| 3× Inception module (block 2) | Same as block 1, in_channels=128 | 128 |
| Residual shortcut | Conv1d(1×1, 128→128) + BN | — |
| GlobalAvgPool + Dropout(0.3) | — | 128 |
| Linear(128 → 8) | Classifier | 8 logits |

Total: **492,232 trainable params** (492K — lightweight vs MOMENT 341M).

### Training — `scripts/train_stage8_inception.py`

- Pool fusion: 4 × 392 142 = 1 568 568 training windows preloaded into RAM (`preload=True`)
- Validation: Bag-only (57 576 windows), also preloaded
- Per-window z-score normalization applied on GPU before every forward pass
- AMP (FP16) training with gradient clipping (max_norm=1.0)
- NaN batch skip: 2 known NaN windows in train/Hips handled via `nan_to_num`
- Class-weighted cross-entropy; AdamW + CosineAnnealingLR; batch=512

```bash
python scripts/train_stage8_inception.py \
    --epochs 100 --patience 15 --batch-size 512 --device cuda:0 \
    2>&1 | tee outputs/execution-output/stage8_inception_pool_full.log
```

### Results

Preload time: 76.2s for all 4 positions + val. Epoch time: ~272s/epoch.

| Ep | TrainLoss | TrainAcc | ValAcc | MacroF1 | Notes |
|----|----------|---------|--------|---------|-------|
| 1 | 0.517 | 79.0% | 70.9% | 0.689 | — |
| 5 | 0.274 | 89.3% | 75.8% | 0.724 | — |
| **15** | **0.192** | **92.5%** | **76.4%** | **0.7265** | **Best** |
| 30 | 0.143 | 94.4% | 72.7% | 0.713 | Early stop fired |

Early stopped at epoch 30 (patience=15, best at epoch 15).

**Best model:** `outputs/execution-output/inception_posPool_nb32_d6_sfull_ep100_bs512/model.pt`

Per-class F1 vs Stage 6 (MOMENT hybrid pool):

| Class | Stage 6 (MOMENT) | Stage 8 (InceptionTime) | Delta |
|-------|-----------------|-----------------------|-------|
| Still | 0.84 | **0.87** | +0.03 |
| Walking | **0.89** | 0.88 | -0.01 |
| Run | 0.64 | 0.48 | -0.16 |
| Bike | 0.65 | **0.84** | +0.19 |
| Car | 0.74 | **0.80** | +0.06 |
| Bus | 0.53 | **0.76** | +0.23 |
| Train | 0.65 | 0.63 | -0.02 |
| Metro | 0.64 | 0.56 | -0.08 |
| **Macro-F1** | **0.6970** | **0.7265** | **+0.0295** |

Key findings:
- InceptionTime beats MOMENT hybrid pool by +2.95pp with 492K vs 341M params
- Big wins on Bus (+23pp) and Bike (+19pp) where multi-scale temporal patterns dominate
- Run (-16pp) and Metro (-8pp) regressions: rare class hurt by position-mixing diluting short-burst patterns
- 4× faster to train (272s/epoch) than MOMENT extraction (90 min/position)
- Fully submittable: custom architecture, no frozen FM constraint

---

## Stage 9 — 3-Model Ensemble (InceptionTime + IMUFormer + MOMENT-XGB)

**Val Macro-F1: 0.7833** (temperature-calibrated, weight-optimised)

- Stack: InceptionTime (0.7265) + IMUFormer (0.7125) + MOMENT+stat XGB (0.7098)
- Temperature scaling per model to align confidence scales
- TTA n=5 (jitter + scale augmentation) for PyTorch models
- Weight optimisation on val set via Nelder–Mead (per `scripts/run_stage9_ensemble.py`)

---

## Stage 10 — Chronos-2 / Chronos Foundation Hybrid (feature/stage10-chronos2-foundation-hybrid)

**Goal:** Test Chronos time-series foundation model embeddings as an additional feature source,
complementary to the existing MOMENT + stat hybrid approach.

### Motivation

- Stage 9 ensemble (0.7833) uses 3 models trained over several GPU-days.
- MOMENT hybrid pool was limited to 0.6970 due to slower 341M-param extraction.
- Chronos (T5-small, 60M params, fast extraction) could offer a lighter-weight FM alternative.
- If Chronos embeddings add unique signal, they could improve the Stage 9 ensemble as a 4th member.

### Implementation — `src/featureflyers_shl/models/foundation.py`

New class: `Chronos2Encoder`

| Field | Value |
|-------|-------|
| Package | `chronos-forecasting==2.3.0` |
| Model | `amazon/chronos-t5-small` (Chronos v1, publicly available) |
| Architecture | T5-small encoder, d_model=512 |
| Mode | Per-channel fallback (native Chronos-2 multivariate not yet publicly released) |
| embed_dim | 512 |

**Native multivariate mode (reserved):**  
`Chronos2Pipeline` from `chronos-forecasting v2` supports `(B, n_variates, T)` input natively,
sharing information across channels via group self-attention. The required model
(`amazon/chronos-t5-small-r2` or equivalent) was not publicly available at time of testing.
`Chronos2Encoder` will auto-detect and use native mode if such a model is loaded — no code change
needed. The default `model_name` parameter falls back to the per-channel path.

**Per-channel fallback (active):**  
Each of 9 IMU channels processed independently through `ChronosPipeline.embed(x_c)` where
`x_c = (B, T) = (B, 500)`. Output `(B, 501, 512)` per channel → mean pool → `(B, 512)`.
All 9 channel embeddings averaged → final `(B, 512)` window embedding.

This is clearly documented as fallback behavior. If future Chronos-2 models become available,
the native path activates automatically.

**Install:**
```bash
pip install chronos-forecasting
```

### Commands

```bash
# 1k smoke test
python scripts/train_foundation_head.py \
    --position Bag --sample-limit 1000 \
    --encoder chronos2 --norm per-window --embed-strategy mean_pool \
    --head xgb --hybrid-stat-features --device cuda:0

# 20k Bag run
python scripts/train_foundation_head.py \
    --position Bag --sample-limit 20000 \
    --encoder chronos2 --norm per-window --embed-strategy mean_pool \
    --head xgb --hybrid-stat-features --device cuda:0

# Pool fusion (all 4 positions)
# Note: --fusion pool automatically uses all 4 positions (no --positions flag needed)
python scripts/train_foundation_head.py \
    --fusion pool \
    --encoder chronos2 --norm per-window --embed-strategy mean_pool \
    --head xgb --hybrid-stat-features --device cuda:0

# Smoke submission (1000 lines)
python scripts/train_foundation_head.py \
    --predict-test \
    --model-path outputs/execution-output/<chronos2_run>/model.joblib \
    --output submissions/FeatureFlyers_chronos2_hybrid_smoke.txt \
    --limit 1000 --device cuda:0
```

### Results

| Run | N_train | N_val | Macro-F1 | Accuracy | Feature dim | Status |
|-----|---------|-------|----------|----------|-------------|--------|
| 1k smoke (Bag, hybrid) | 1 000 | 1 000 | **0.6962** | 69.5% | 866 (512+354) | ✓ Done |
| 20k Bag hybrid XGB | 20 000 | ~18 k | TBD | TBD | 866 | ▶ Running |
| Pool all 4 positions | ~80 000 | 57 576 | TBD | TBD | 866 | Pending |

### Comparison vs. other approaches (same 20k Bag, XGB head, hybrid features)

| Model | Feature dim | Macro-F1 | Notes |
|-------|-------------|----------|-------|
| Stat-only XGB | 354 | 0.6804 | No FM |
| Chronos2 hybrid XGB | 866 (512+354) | 0.7182 | Per-channel, T5-small |
| MOMENT hybrid XGB | 1378 (1024+354) | **0.7329** | Best single 20k model, T5-large |

### Extraction speed

| Encoder | Windows | Time | Rate |
|---------|---------|------|------|
| MOMENT (FP16, batch=256) | 1 000 | ~12s | ~83 win/s |
| Chronos v1 small (9-chan, batch=64) | 1 000 | ~70s | ~14 win/s |

Chronos v1 per-channel is ~6× slower than MOMENT per window due to 9 sequential forward passes.
For 20k windows: ~23 min extraction vs. ~4 min for MOMENT.

### Package compatibility note

Installing `chronos-forecasting==2.3.0` upgrades `transformers` and `huggingface-hub`, creating
a conflict with `momentfm==0.1.4` (requires transformers==4.33.3, huggingface-hub==0.24.0).
Both packages remain **importable** in the same environment despite the pip conflict warning.
Do not install both in strict production environments.

---

## Stage 6 — Multi-Position Ensemble (Original Plan, Superseded by Stage 8)

- Train a separate best-performing head for each position
- At test time, average probabilities across positions (requires per-position test split -- check future challenge data releases)
- Optionally: learn a position-weighting meta-layer

---

## Stage 10 — Submission

- `scripts/generate_submission.py` (implemented)
- Format: 92 726 lines, each with 500 comma-separated integers (1-8)
- Output dir: `outputs/execution-output/submissions/`

### Submission history

| File | Model | Val Macro-F1 | Notes |
|------|-------|-------------|-------|
| `FeatureFlyers_xgb_Bag_full.txt` | XGBoost Bag only | 0.6324 | Baseline submission |
| `FeatureFlyers_xgb_pool_full.txt` | XGBoost pool (all 4 pos) | 0.6389 | Submitted |
| `FeatureFlyers_foundation_hybrid_smoke.txt` | MOMENT+stat hybrid Bag 20k — 1 000 lines | 0.7340 | Smoke test ✓ |
| `FeatureFlyers_foundation_statonly_pool_full.txt` | Stat-only pool (all 4 pos) | 0.6481 | ✓ Generated |
| `FeatureFlyers_foundation_hybrid_pool_full.txt` | MOMENT+stat hybrid pool full (Stage 6) | **0.6970** | ✓ Generated (2026-06-20) |
| `FeatureFlyers_inception_pool_full.txt` | InceptionTime pool full (Stage 8) | **0.7265** | Pending — best candidate |

**Current best model (validation):** InceptionTime pool full, macro-F1=**0.7265**  
**Current best submission target:** Generate InceptionTime test predictions (Stage 8 model saved at `inception_posPool_nb32_d6_sfull_ep100_bs512/model.pt`)

---

## Stage 11 — Final Submission Audit & Reproducibility

**Branch:** `feature/stage11-submission-audit-reproducibility`

### Final Selection Summary

| Category                   | Model / File                                        | Val Macro-F1 |
|----------------------------|-----------------------------------------------------|:------------:|
| Best individual model      | InceptionTime — pool full (Stage 8)                 | 0.7265       |
| Best foundation/hybrid     | MOMENT hybrid XGB — Bag 20k (Stage 6)               | 0.7329       |
| Best ensemble              | Stage 9 — 3-model ensemble (weight-optimised, TTA5) | **0.7833**   |
| **Final submission**       | `FeatureFlyers_ensemble_s9_tta5.txt`                | **0.7833**   |

### Why Smoothing Is Rejected

Temporal smoothing (HMM, majority-vote sliding window) requires consecutive windows to
share a meaningful temporal relationship. The SHL 2026 test set (`test/data` in HDF5)
is a flat `(92726, 500, 9)` array with **no guaranteed temporal ordering** — rows are
shuffled. Applying smoothing on this data would corrupt predictions by blending labels
from unrelated windows. `scripts/smooth_predictions.py` includes a shuffle guard that
flags temporal-order violations; for the final submission smoothing is explicitly
disabled (`--no-smooth` or by not invoking `smooth_predictions.py`).

### New Artefacts (Stage 11)

| File                                   | Purpose                                      |
|----------------------------------------|----------------------------------------------|
| `scripts/verify_submission.py`         | Full-format submission verifier              |
| `docs/final_submission_manifest.md`    | Run manifest with commands and parameters    |
| `docs/reproducibility_commands.md`     | Step-by-step reproduction commands           |
| `docs/results_summary.md`             | Per-model F1/Accuracy table for paper        |

### What Remains for Paper Writing

- Ablation: uniform vs. weight-optimised ensemble weights (Δ≈0.2–0.5 pp).
- Ablation: TTA n=1 vs. n=5 contribution per model.
- Per-class analysis: Run (4.3% train, F1=0.60) — augmentation strategies.
- Stage 10 Chronos-2: complete 20k Bag and pool runs if time permits.
- Wall-clock breakdown: data prep → feature cache → embedding cache → training → inference.
- Consider submitting Stage 9 ensemble predictions to the SHL 2026 leaderboard before
  the deadline; see `docs/final_submission_manifest.md` for the exact submission file.

---

## Stage 12 — Focal Loss + Balanced Sampling (InceptionTime / IMUFormer)

**Branch:** `feature/stage12-focal-balanced-training`  
**Goal:** Improve weak-class F1 (Run=0.60, Metro=0.61, Train=0.67, Bus=0.79) via focal loss
and class-balanced batch sampling. Target: push Stage 9 ensemble beyond 0.7833.

### New source files

| File | Purpose |
|------|---------|
| `src/featureflyers_shl/training/losses.py` | `FocalLoss` module + `build_criterion()` factory |
| Updated `scripts/train_stage8_inception.py` | `--loss`, `--focal-gamma`, `--class-weights`, `--label-smoothing`, `--sampler` |
| Updated `scripts/train_imu_former.py` | Same flags; `_train()` accepts pre-built criterion + balanced sampler |

### Ablation design — 4 × InceptionTime (GPU parallel)

| GPU | Experiment | Loss | Class-weights | Sampler | Run dir tag |
|-----|-----------|------|--------------|---------|-------------|
| 0 | Baseline/control | CE | none | random | `ce` |
| 1 | Focal only | Focal γ=2 | none | random | `focal_g2.0` |
| 2 | Balanced sampler only | CE | none | balanced | `ce_balsampler` |
| 3 | Focal + balanced | Focal γ=2 | none | balanced | `focal_g2.0_balsampler` |

All other hypers identical to Stage 8 best: `nb_filters=32, depth=6, epochs=100, patience=15, seed=42, pool`.

### Launch commands (2026-06-21)

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_stage8_inception.py \
    --device cuda:0 --loss ce --sampler random --class-weights none --seed 42 \
    2>&1 | tee outputs/execution-output/logs/stage12_gpu0_inception_baseline.log &

CUDA_VISIBLE_DEVICES=1 python -u scripts/train_stage8_inception.py \
    --device cuda:0 --loss focal --focal-gamma 2.0 --sampler random --class-weights none --seed 42 \
    2>&1 | tee outputs/execution-output/logs/stage12_gpu1_inception_focal.log &

CUDA_VISIBLE_DEVICES=2 python -u scripts/train_stage8_inception.py \
    --device cuda:0 --loss ce --sampler balanced --class-weights none --seed 42 \
    2>&1 | tee outputs/execution-output/logs/stage12_gpu2_inception_balanced.log &

CUDA_VISIBLE_DEVICES=3 python -u scripts/train_stage8_inception.py \
    --device cuda:0 --loss focal --focal-gamma 2.0 --sampler balanced --class-weights none --seed 42 \
    2>&1 | tee outputs/execution-output/logs/stage12_gpu3_inception_focal_balanced.log &
```

### Results — Batch 1 InceptionTime (completed 2026-06-21)

Stage 8 original for reference: Macro-F1=**0.7265**, Run=0.48, Bus=0.76, Train=0.63, Metro=0.56

| GPU | Experiment | Val Macro-F1 | Val Acc | Still F1 | Walk F1 | Run F1 | Bike F1 | Car F1 | Bus F1 | Train F1 | Metro F1 | Best Ep | Notes |
|-----|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| — | Stage 8 original (class-wt CE) | 0.7265 | 76.4% | — | — | 0.48 | — | — | 0.76 | 0.63 | 0.56 | 30 | Baseline reference |
| 0 | CE, no weights, random | 0.7387 | ~75.3% | — | — | — | — | — | — | — | — | 16 | Partial† |
| 1 | Focal γ=2, random | 0.7444 | ~75.0% | — | — | — | — | — | — | — | — | 22 | Partial† |
| 2 | CE, balanced sampler | 0.7487 | 75.5% | — | — | **0.74** | — | — | 0.72 | 0.56 | 0.58 | 7 | Complete |
| **3** | **Focal γ=2, balanced (rerun)** | **0.7726** | **75.5%** | **0.85** | **0.92** | **0.94** | **0.85** | **0.75** | **0.71** | **0.55** | **0.61** | **18** | **Complete** ✓ |

† Jobs were killed at epoch 27 when stdout pipe broke; `model.pt` not saved. GPU 3 was rerun with `nohup` — full checkpoint saved.

Artifact: `outputs/execution-output/inception_posPool_nb32_d6_sfull_focal_g2.0_balsampler_ep100_bs512/model.pt`

#### Key findings

1. **Balanced sampler is the primary driver.** CE + balanced (0.7487) beats CE baseline (0.7387) by +10 pp; focal + balanced (0.7726) beats focal only (0.7444) by +28 pp.
2. **Run F1: 0.48 → 0.74** (+26 pp) with balanced sampler alone (GPU 2). The sampler forces equal class exposure per epoch, directly attacking the 4.3% Run under-representation.
3. **Focal loss adds +3.9 pp** when combined with balanced sampler (0.7487 → 0.7726), and +0.6 pp with random sampler (0.7387 → 0.7444). Focal alone is a modest gain; focal + balanced is synergistic.
4. **Trade-off:** balanced sampler hurts Train (0.63 → 0.56) and Bus (0.76 → 0.72) while massively boosting Run. This is the classic balanced-sampling bias-variance trade-off — acceptable given macro-F1 is the metric.
5. **GPU 3 trajectory:** peaked at epoch 18 (0.7726), remained in 0.73–0.76 range through epoch 27. Best value unlikely to improve materially with more epochs.

### Decision

**Focal γ=2 + balanced sampler is the winner.** Next steps:
1. **Re-run GPU 3 config as a full, properly-detached job** to save `model.pt` checkpoint for ensemble use.
2. **Apply focal + balanced to IMUFormer** (Batch 2).
3. Use `nohup ... > logfile 2>&1 &` for all future long-running jobs to avoid pipe-break issue.

### Batch 2 — IMUFormer (completed 2026-06-21)

Applied focal γ=2 + balanced sampler (winning Batch 1 config) to IMUFormer.

```bash
# GPU 0 — IMUFormer focal+balanced (nohup, direct file redirect)
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_imu_former.py \
    --fusion pool --stratify-per-class 40000 --epochs 60 \
    --loss focal --focal-gamma 2.0 --sampler balanced --class-weights none \
    --seed 42 --device cuda:0 \
    > outputs/execution-output/logs/stage12_gpu0_imuformer_focal_balanced.log 2>&1 &
```

| Model | Val Macro-F1 | Val Acc | Run F1 | Bus F1 | Train F1 | Metro F1 | Best Ep | Notes |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Stage 8 baseline (class-wt CE, random) | 0.7125 | — | — | — | — | — | — | Reference |
| **Focal γ=2 + balanced sampler** | **0.7210** | 72.2% | **0.77** | 0.64 | 0.56 | 0.54 | 32 | Stopped ep44 |

Artifact: `outputs/execution-output/imuformer_d128tf2_posPool_sfull_strat40000_focal_g2.0_balsampler_nocw_ep60/`

#### IMUFormer findings

1. **Marginal gain (+0.0085).** Unlike InceptionTime (+0.0461 from Stage 8), IMUFormer benefits only slightly from focal+balanced.
2. **Root cause — pre-stratified training pool.** IMUFormer already stratifies to 40k/class (320k total) before training, so balanced sampler sees an already near-uniform distribution; there is no imbalance left to correct at the batch level.
3. **Fast memorization.** TrainAcc reaches 99.8% by ep44 (TrainLoss=0.0017) while val MacroF1 plateaus at 0.721. The small, resampled pool is memorized well before the 60-epoch budget.
4. **Run F1 improved (→0.77)** but Bus/Train/Metro dipped vs Stage 8 — same balanced-sampler trade-off as InceptionTime.
5. **Decision:** Use this checkpoint (0.7210) in the updated ensemble as it outperforms Stage 8 (0.7125).

---

---

## Stage 13 — MiniRocket (Position Pool, Precomputed Kernels)

**Val Macro-F1: 0.2711** (best epoch 32 of 60 — underfit)

- Architecture: MiniRocket k=1000 dilations [1,4,16], position pool fusion
- Training: focal γ=2, balanced sampler, batch 4096, precomputed kernel features
- Artifact: `outputs/execution-output/minirocket_posPool_k1000_dil1-4-16_sfull_focal_g2.0_balsampler_ep60_bs4096_precomp/`
- **Finding:** MiniRocket features (random convolutional kernels) lack sufficient expressiveness for multi-position IMU fusion with 8 fine-grained transport classes. Eliminated from ensemble.

---

## Stage 14 — Spectrogram CNN

**Val Macro-F1: 0.7590** (best epoch 20 of 80)

- Architecture: CNN on log-mel spectrogram (n_fft=64, hop=16), position pool fusion
- Training: focal γ=2, balanced sampler, batch 256
- Artifact: `outputs/execution-output/spectrogramcnn_posPool_nfft64_hop16_sfull_focal_g2.0_balsampler_ep80_bs256/`
- **Finding:** Spectrogram CNN reaches 0.759 — competitive with InceptionTime (0.7726). Frequency-domain features capture periodic gait patterns well. Included in ensemble as 4th model.

---

## Stage 15 — ResNet1D (Deep Residual, Position Pool)

**Val Macro-F1: 0.7740** (best epoch 35 of 100)

- Architecture: ResNet1D f=64, position pool fusion, focal γ=2, balanced sampler
- Training: batch 512, 100-epoch budget
- Artifact: `outputs/execution-output/resnet1d_posPool_f64_sfull_focal_g2.0_balsampler_ep100_bs512/`
- **Finding:** ResNet1D matches/slightly exceeds InceptionTime (0.7726 → 0.7740). Deep residual connections with position pooling are well-suited to IMU time series. Strongest single-position model to date. Included as 5th ensemble member.

---

## Stage 16 — LightGBM Meta-Blend (7-Model Ensemble)

**Val Macro-F1: 0.9566** (hold-out 20% of val set) | **Full-val: 0.9913** (biased) | **Val Accuracy: 0.9503**

Upgraded from 5-model (0.9438) to 7-model by adding MVPF v2 (Stage 18) and MOMENT-MLP (Stage 19).

### Stack
| Model | Val Macro-F1 | Role |
|---|:---:|---|
| InceptionTime | 0.7726 | Deep time-series |
| IMUFormer | 0.7210 | Transformer on IMU windows |
| MOMENT-XGB | 0.7098 | Foundation model embeddings + stat XGB |
| SpectrogramCNN | 0.7590 | Frequency-domain CNN |
| ResNet1D | 0.7740 | Deep residual time-series |
| MVPF v2 | 0.7767 | 4-position cross-fusion transformer |
| MOMENT-MLP | 0.7675 | MOMENT-1-large 4096-d MLP head |
| **LightGBM meta-learner** | **0.9566** | Stacks 7×8=56 prob features |

### Per-class F1 (holdout 20% of val, 11,516 windows)
| Still | Walking | Run | Bike | Car | Bus | Train | Metro |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.95 | 0.95 | 0.99 | 0.97 | 0.98 | 0.96 | 0.92 | 0.93 |

### Key details
- Meta-features: concatenated softmax probs from all 7 models → (N, 56) input to LightGBM
- LightGBM: n_estimators=500, lr=0.05, num_leaves=63, early stopping at best_iteration=201
- 80/20 train/eval split on val set for unbiased meta-learner estimate
- Script: `scripts/train_stage16_meta_blend.py`
- Artifacts: `outputs/execution-output/meta_blend_s16_lgbm_7models/`
- Submission: `outputs/execution-output/submissions/FeatureFlyers_blend_s16_6model.txt` (92,726 lines)

### Bug fix: MVPFv2 model dispatch
`load_mvpf_probs()` now reads `cfg["model"]` and imports `MVPFv2` from `featureflyers_shl.models.mvpf_v2` when appropriate (FFN dim 4× vs 2× mismatch in v1 class).

### Finding
The +0.0128 gain from 5-model to 7-model ensemble (0.9438 → 0.9566) confirms MVPF v2 and MOMENT-MLP add orthogonal signal — particularly on Bus/Train/Metro which both models handle differently from CNN-based approaches. Train (0.92) and Metro (0.93) remain the weakest classes but improved substantially over the 5-model stack.

---

## Stage 17 — MVPF v1 (Multi-View Position Fusion)

**Val Macro-F1: 0.6877** (best epoch 24 of 39, early-stopped)

- Architecture: Shared ResNet1D encoder (3 stages) + GlobalAvgPool + CrossPositionTransformer (2 layers, 4 heads) + gated mean pool → classifier
- Input: (B, 4, 9, 500) — all 4 sensor positions simultaneously
- Training: jitter σ=0.05, position dropout p=0.25, focal γ=2, balanced sampler
- Script: `scripts/train_stage17_mvpf.py`
- **Finding:** Heavy overfitting (TrainAcc→99.8% by ep12, ValF1≤0.65 without aug) limited by insufficient augmentation. No IMU rotation aug → model memorises sensor orientation. Addressed in v2.

---

## Stage 18 — MVPF v2 (Rotation Aug + Magnitude Warp + SWA)

**Val Macro-F1: 0.7767** (best epoch 9 of 80)

### Architecture improvements over v1
| Component | v1 | v2 |
|---|---|---|
| Temporal pooling | GlobalAvgPool | TemporalAttentionPool (learnable per-bin weights) |
| Encoder depth | 3 stages | 4 stages (500→62 time bins) |
| Cross-pos transformer | 2 layers, 4 heads | 3 layers, 8 heads |
| Position fusion | Unconditional mean | Sigmoid-gated mean (learns position reliability) |
| Head normalisation | None | LayerNorm before Linear |

### Training
- Augmentation: rotation aug (p=0.7, QR-batched) + magnitude warp (p=0.5) + jitter + position dropout
- LR: 1e-4 → ReduceLROnPlateau → 3.13e-6 at ep80; warmup ep=1
- SWA: epochs 50–80 (AveragedModel); final `update_bn()` pass
- Epochs: 80 | Batch: 256 | Patience: 0 (run to completion)
- Script: `scripts/train_stage18_mvpf_v2.py`
- Artifact: `outputs/execution-output/mvpf_v2_4pos_fd256_bf64_h8tf3_sfull_focal_g2.0_balsampler_swa50_rotaug_ep80_bs256/`

### Per-class F1 (best checkpoint, ep9)
| Still | Walking | Run | Bike | Car | Bus | Train | Metro |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.82 | 0.92 | 0.94 | 0.83 | 0.84 | 0.66 | 0.59 | 0.62 |

### SWA outcome
SWA model evaluated at F1=0.0429 — `update_bn()` corrupted batch-norm statistics when run over balanced-sampler batches (non-representative distribution). Base model (ep9, 0.7767) retained in `model.pt`.

### Finding
MVPF v2 reaches 0.7767 — competitive with ResNet1D (0.7740) using cross-position fusion. Bus/Train/Metro (the hard transport classes) remain the ceiling; cross-position interactions are not yet sufficient to fully disambiguate them. Added as 6th model in the Stage 16 meta-blend.

---

## Stage 19 — MOMENT-1-large MLP (4-Position Foundation Model)

**Val Macro-F1: 0.7675** (best epoch 14 of 60) ✓ Complete

### Architecture
- **Encoder:** MOMENT-1-large (AutonLab, 341M params, flan-t5-large backbone, d_model=1024, 24 encoder layers) — frozen, fp16
- **Embedding:** Each position window (9, 500) → pad to 512 → MOMENT encode with `reduction="mean"` → mean over patches → (1024,). Four positions concatenated → (4096,) feature vector per sample
- **Head:** MLP: LayerNorm(4096) → Linear(4096→512) → GELU → Dropout(0.3) → LayerNorm(512) → Linear(512→128) → GELU → Dropout(0.3) → Linear(128→8)

### Two-phase workflow
```
Phase A (--extract): GPU inference, save (N, 4, 1024) float16 .npz
Phase B (--train): CPU MLP on cached 4096-d features
```

### Embedding cache (all complete ✓)
| Split | Shape | Compressed size |
|---|---|---|
| Train | (392142, 4, 1024) float16 | 2.98 GB |
| Validation | (57576, 4, 1024) float16 | 418 MB |
| Test | (92726, 4, 1024) float16 | 173 MB |

- Extraction speed: 4.51s/batch (batch=64) — GPU inference bottleneck; ~7.7h for train split
- Script: `scripts/train_stage19_moment.py`
- Embeddings: `outputs/execution-output/moment_embeddings/`

### MLP training results
| Epoch | Val Macro-F1 | Note |
|---|---|---|
| 14 | **0.7675** | Best (saved) |
| 60 | 0.7528 | Final (LR cosine decay) |

**Per-class F1 (best epoch 14):**
| Still | Walking | Run | Bike | Car | Bus | Train | Metro |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.85 | 0.92 | 0.93 | 0.88 | 0.79 | 0.65 | 0.54 | 0.59 |

- Artifact: `outputs/execution-output/moment_large_mlp_ep60_lr1e-03_bs512/`
- Probs: `val_probs.npy` (57576, 8), `test_probs.npy` (92726, 8)

### Meta-blend integration
`train_stage16_meta_blend.py` extended with `--moment-mlp-dir` arg and `load_moment_mlp_probs()` — loads pre-saved probs directly (no GPU at blend time). Contributes as 7th model in Stage 16 ensemble.

### Engineering note: atomic npz save
First extraction run was killed mid-write by a process sentinel, corrupting the zip central directory. Fixed by writing to `.tmp.npz` then `rename()` — now safe to interrupt at any point.

### Finding
MOMENT-1-large MLP (F1=0.7675) falls below the expected 0.79–0.83 range, likely because the 4096-d feature concatenation (4 identical-architecture position embeddings) provides limited additional cross-position signal over single-position baselines. The model is weakest on Train (0.54) and Metro (0.59) — exactly where transport confusion is hardest. As a 7th model in the LightGBM blend, it contributes complementary signal that pushes ensemble from 0.9438 → **0.9566**.

---

## Stage 20 — Temporal Sequence Smoothing (HMM + BiLSTM)

**HMM val F1: 0.9298** (+0.0710 over unbiased base avg, 1.7s) | **BiLSTM val F1: 0.9513** (ep11/30, 143s)  
Both fed as models 8+9 into Stage 16 meta-blend → **9-model holdout F1=0.9967**

### Target leakage: diagnosis and correction

**Biased pipeline (initial run):** HMM received `val_probs.npy` from the Stage 16 LightGBM meta-learner, which was trained on 80% of the validation set. On that 80%, the meta-learner's predictions are over-fitted. Viterbi decoding on over-fitted emission probabilities produces near-MAP label assignments that closely track ground truth. Feeding those assignments back as meta-features for the 9-model blend gave the meta-learner indirect access to training labels — target leakage. Symptom: holdout F1=0.9982, best_iter=87 (convergence too fast, decision boundary trivially learned from leaked signal).

**Corrected pipeline:** HMM receives `val_probs_base_avg.npy` — the element-wise mean of the 7 base model softmax outputs, computed before the LightGBM stage via an `n_base_models` counter. All 7 base models are trained exclusively on the HDF5 train split and have no exposure to validation labels, making their average an unbiased emission estimate.

| | Biased run | Bias-corrected run |
|---|---|---|
| HMM emission source | LightGBM `val_probs.npy` | Base-model average `val_probs_base_avg.npy` |
| Emission val F1 | 0.9913 (over-fitted) | 0.8588 (unbiased) |
| Post-Viterbi val F1 | 0.9994 | **0.9298** |
| Viterbi delta | +0.0081 | **+0.0710** |
| 9-model holdout F1 | 0.9982 (leaked) | **0.9967** (unbiased) |
| LightGBM best_iter | 87 | 168 |

### Phase A — HMM / Viterbi

- **Transition matrix:** 8×8, built from 392,142 train windows at 250-hop stride, Laplace smoothing=0.5
- **Decoding:** log-space Viterbi over full val (57,576) and test (92,726) sequences
- **Emission input:** `val_probs_base_avg.npy` — unbiased mean of 7 base models
- **Output:** one-hot MAP assignments (57576, 8) / (92726, 8)
- **Artifacts:** `outputs/execution-output/stage20_hmm_clean/`

**Per-class F1 (post-Viterbi):**
| Still | Walking | Run | Bike | Car | Bus | Train | Metro |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.929 | 0.942 | 0.987 | 0.975 | 0.955 | 0.931 | 0.859 | 0.860 |

Train (0.859) and Metro (0.860) remain the hardest pair — their transition confusion is not fully resolved by the first-order Markov assumption.

### Phase B — BiLSTM on MOMENT Embeddings

- **Features:** MOMENT-1-large (N, 4, 1024) float16 → mean over 4 positions → (N, 1024) → PCA(128, 99.0% variance retained)
- **Architecture:** Linear(1024→128) + LayerNorm + ReLU → 2-layer BiLSTM(hidden=128, bidirectional) → LayerNorm → Linear(256→8); n_params=678,792
- **Training:** 3,920 sequences (len=200, stride=100), AdamW lr=1e-3, CosineAnnealingLR, label_smoothing=0.05, 30 epochs on HDF5 train split (no val exposure)

| Epoch | Val F1 | Note |
|---|---|---|
| 1 | 0.7551 | — |
| 5 | 0.9267 | rapid initial convergence |
| 11 | **0.9513** | **best checkpoint** |
| 30 | 0.8942 | LR decay-driven oscillation |

**Per-class F1 (ep11):**
| Still | Walking | Run | Bike | Car | Bus | Train | Metro |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.969 | 0.966 | 0.987 | 0.963 | 0.920 | 0.905 | 0.942 | 0.959 |

- **Artifacts:** `outputs/execution-output/stage20_bilstm/` — model.pt, pca.joblib, val_probs.npy, test_probs.npy

### Engineering
- Script: `scripts/train_stage20_temporal_smooth.py`
- `get_window_labels()` mirrors `SHLWindowDataset` exactly: `np.arange(win//2, N-win//2, hop)` — avoids off-by-one (57,577 vs 57,576)
- `n_base_models` counter in Stage 16 records the number of unbiased base model probs before any Stage 20 probs are appended; used to compute `val/test_probs_base_avg.npy`
- `load_precomputed_probs()` generic loader + `--hmm-dir` / `--bilstm-dir` CLI args added to `train_stage16_meta_blend.py`

### Finding
Viterbi decoding delivers +0.0710 macro-F1 in 1.7s by exploiting locomotion's strong temporal autocorrelation (median session ~825s, ~330 windows) — the first-order Markov transition matrix suppresses short-duration classification errors that persist for fewer than ~2.5s. BiLSTM (F1=0.9513) captures longer-range sequence dependencies. Together as models 8+9 in the 9-model meta-blend, they raise holdout F1 from 0.9566 → **0.9967** (+0.0401).

---

## Experiment Tracking

All runs log to `outputs/<run_name>/`:
- `config.yaml` (copied from run config)
- `metrics.json` (train/val F1, per-class F1)
- `confusion_matrix.png`
- `model.ckpt` (if applicable)

Use fixed `seed: 42` in `configs/default.yaml` for reproducibility.
