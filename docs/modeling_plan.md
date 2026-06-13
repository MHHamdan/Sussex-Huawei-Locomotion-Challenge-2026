# Modeling Plan — FeatureFlyers / SHL 2026

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

### Infrastructure improvements
- [x] `scripts/precompute_features.py` — pre-computes all 354 features for every position/split using 32 parallel CPU workers, saves to `dataset/processed/features/*.npz` (2.4 GB total). Cuts data loading from ~130s to ~1s per run.
- [x] XGBoost GPU (`--model xgb --device cuda`) — replaces CPU-only Random Forest for faster fitting
- [x] Balanced sample weights for XGBoost — `compute_sample_weight("balanced")` fixes class imbalance on full dataset
- [x] Early stopping for XGBoost — `--early-stopping-rounds 30` prevents overfitting and saves training time
- [x] NaN guard in `statistical.py:extract()` — handles recording gaps (2 windows in train/Hips had NaN raw sensor values)

### Results — all runs, Bag position, seed=42, full validation set (57 576 windows)

| Model | Positions | Sample Limit | Macro-F1 | Accuracy | Runtime | Notes |
|-------|-----------|-------------|---------|---------|---------|-------|
| RF | Bag | 20 000 | 0.6427 | 63.4% | 121s | sklearn, CPU, class_weight=balanced |
| XGBoost | Bag | 20 000 | 0.6605 | 64.7% | 126s | GPU, no sample weights |
| XGBoost v1 | Bag | full | 0.6039 | 59.6% | 1303s | GPU, no sample weights — class imbalance hurt |
| XGBoost v2 | Bag | full | 0.6384 | 62.9% | 1176s | GPU, balanced weights + early stop |
| **XGBoost** | **Bag+Hand+Hips+Torso** | **full** | **0.6613** | **67.8%** | **86s** | GPU, 4-pos early fusion, cached features |

Best per-class F1 (4-position early fusion):

| Class | F1 | Notes |
|-------|----|-------|
| Still | 0.86 | strong |
| Walking | 0.89 | strong |
| Run | 0.66 | rare class (1 110 val windows) |
| Bike | 0.87 | strong |
| Car | 0.72 | moderate |
| Bus | 0.48 | hard — confused with Train |
| Train | 0.50 | hard — confused with Metro/Bus |
| Metro | 0.30 | hardest — underground vibration similar to Train |

Key findings:
- 4-position fusion adds +2.3pp macro-F1 over single-position (0.6613 vs 0.6384)
- Cache reduces total pipeline time from ~20 min to ~90s for a full-dataset run
- Train/Metro remain the hardest pair — need deeper model or foundation model embeddings

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

## Stage 4 — Frozen Foundation Model + Head

**Goal:** leverage representation power of pre-trained time-series models without fine-tuning.

### Candidate foundation models

| Model | Type | Input format | Notes |
|-------|------|-------------|-------|
| [MOMENT](https://github.com/moment-timeseries-foundation-model/moment) | Transformer | `(batch, T, C)` patches | Time-series foundation model / Time-series BERT-style |
| [Moirai](https://github.com/SalesforceAIResearch/uni2ts) | Transformer | Multivariate time series | Salesforce Uni2TS; includes Moirai models |
| [TimesFM](https://github.com/google-research/timesfm) | Transformer | Univariate time series | Google Research; Apache-2.0 |
| [TimeSformer](https://github.com/facebookresearch/TimeSformer) | Video Transformer | Video frames / adaptable | Can be adapted by reshaping sensor windows, but repo is archived |

### Protocol (all models)
1. Freeze all foundation model parameters (`requires_grad = False`).
2. Forward-pass windows through the model → embedding vector(s).
3. Attach trainable head: Linear(d_model → 8) or MLP(d_model → 256 → 8).
4. Train only the head (seconds/minutes not hours).
5. No data augmentation on foundation model inputs to avoid distribution shift.

### Embedding strategies
- CLS token / mean pool of patch embeddings
- Per-sensor embedding → concatenate 9 streams → head
- Multi-position: extract embeddings per position, concat, single head

---

## Stage 5 — Multi-Position Ensemble

- Train a separate best-performing head for each position
- At test time, average probabilities across positions
- Optionally: learn a position-weighting meta-layer

---

## Stage 6 — Submission

- `scripts/generate_submission.py` (to be created)
- Format: `FeatureFlyers_predictions.txt` — comma-separated label integers, one line
- Final check: sample count must match test set (92 726 samples)

---

## Experiment Tracking

All runs log to `outputs/<run_name>/`:
- `config.yaml` (copied from run config)
- `metrics.json` (train/val F1, per-class F1)
- `confusion_matrix.png`
- `model.ckpt` (if applicable)

Use fixed `seed: 42` in `configs/default.yaml` for reproducibility.
