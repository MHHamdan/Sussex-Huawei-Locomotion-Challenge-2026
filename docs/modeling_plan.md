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

### Results (5 000 stratified windows, Bag, RF, seed=42)

| Mode | Macro-F1 | Accuracy |
|------|----------|---------|
| Single position (Bag) | ~0.57 | ~0.57 |
| 4-position early fusion | TBD | TBD |
| Full dataset | TBD | TBD |

Run is hardest to recall (rare + distinct). Metro is hardest to distinguish from Train.

---

## Stage 3 — Deep Time-Series Baseline

**Goal:** end-to-end learnable baseline on raw windows.

Architecture options:
- 1-D CNN (simple, fast)
- LSTM / GRU (captures sequential dependencies)
- Temporal Convolutional Network (TCN)

Inputs: (batch, 500, 9) float32 windows.
Output: 8-class softmax head.

Training:
- Adam, lr=1e-3, cosine decay
- Batch size 512, 50 epochs
- Class-weighted cross-entropy for imbalance

Multi-position fusion:
- Late fusion: average softmax logits across available positions
- Early fusion: concatenate windows → (batch, 500, 36)

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
