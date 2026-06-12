# Modeling Plan — FeatureFlyers / SHL 2026

## Problem

Multi-class locomotion mode recognition (8 classes) from 9-axis IMU sensor data collected
at 4 phone positions (Bag, Hand, Hips, Torso). Evaluation metric: **F1-score** (macro).

Foundation models must remain **frozen**. Only lightweight heads may be trained.

---

## Stage 1 — Data Preparation and EDA

- [x] Unzip raw data → `dataset/raw/`
- [x] Convert to HDF5 → `dataset/processed/shl2026.hdf5`
- [ ] Label distribution plots per split and position
- [ ] Signal visualisation: 10-second windows per class per sensor
- [ ] Sample-rate verification (expected ~100 Hz after windowing)
- [ ] Check class imbalance across positions
- [ ] Define window size and hop: 500 samples @ 500 Hz = 1 s windows

---

## Stage 2 — Classical ML Baseline

**Goal:** establish a reproducible floor before any deep learning.

### Feature extraction (per window, per sensor axis)

Statistical:
- mean, std, min, max, range
- RMS, zero-crossing rate
- skewness, kurtosis

Spectral (FFT):
- Top-K dominant frequencies and their magnitudes
- Spectral energy, spectral entropy, spectral rolloff

Cross-axis:
- Correlation coefficients between Acc_x/y/z
- Magnitude: sqrt(x²+y²+z²) for Acc, Gyr, Mag

### Classifiers
1. Random Forest (RF) — strong baseline, interpretable feature importances
2. Support Vector Machine (linear kernel, scaled features)

### Evaluation
- 5-fold cross-validation on training set
- Hold-out validation set for threshold tuning
- Report macro-F1, per-class F1, confusion matrix

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
| [MOMENT](https://github.com/moment-research/MOMENT) | Transformer | (batch, T, C) patches | Time-series BERT |
| [Moirai](https://github.com/SalesforceAIResearch/uni2ts) | Transformer | multi-variate TS | Salesforce, MIT license |
| [TimesFM](https://github.com/google-research/timesfm) | Transformer | univariate TS | Google, Apache-2 |
| [TimeSformer](https://github.com/facebookresearch/TimeSformer) | Video Transformer | adaptable | needs reshape |

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
