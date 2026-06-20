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
| LoRA r=8 α=16 blocks=4 | 80 000 | running | pending | pending | GPU 1, ~100% util |

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

## Stage 9 — Ensemble (Planned)

- Stack: XGB (stat features) + InceptionTime + MOMENT head
- Temperature scaling for calibrated probabilities
- Test-time augmentation (TTA): predict on multiple augmented copies, average

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

## Experiment Tracking

All runs log to `outputs/<run_name>/`:
- `config.yaml` (copied from run config)
- `metrics.json` (train/val F1, per-class F1)
- `confusion_matrix.png`
- `model.ckpt` (if applicable)

Use fixed `seed: 42` in `configs/default.yaml` for reproducibility.
