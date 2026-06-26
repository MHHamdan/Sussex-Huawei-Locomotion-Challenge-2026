# FeatureFlyers — Submission Workflow Summary for Co-Authors

**Challenge**: SHL 2026 — 8-class locomotion recognition from 9-axis IMU data (4 body positions).
**Metric**: Macro-F1 over 8 activity classes (Still, Walking, Run, Bike, Car, Bus, Train, Metro).
**Key rule**: The solution must be built around a **frozen pre-trained foundation model**. Foundation weights must not be updated. Only lightweight heads, classifiers, or meta-learners may be trained.

---

## The Data

The dataset is a single HDF5 file with three splits:

| Split | Shape | Labels |
|-------|-------|--------|
| Train | 392,142 windows × 500 samples × 9 channels | Yes (1–8) |
| Validation | 57,576 windows × 500 samples × 9 channels | Yes (1–8) |
| Test | 92,726 windows × 500 samples × 9 channels | **No** — submitted for scoring |

A "window" is 5 seconds of IMU at 100 Hz across 9 axes (3-axis accelerometer, gyroscope, magnetometer). The 4 body positions (Bag, Hand, Hips, Torso) are available for train/val; at test time only a single flat array is provided with no position labels and no temporal ordering — the rows are shuffled.

---

## Step 1 — Core Foundation Path (the primary claim)

**Model**: MOMENT-1-large (341M parameters, pre-trained on diverse time-series corpora, weights fully frozen).

**Embedding extraction**: Each 500-sample window is passed through MOMENT as a sequence of patches. MOMENT returns a 1024-d embedding per position. We extract embeddings for all 4 positions and concatenate → **4096-d vector per window**. Extraction runs in fp16, batch size 64.

| Split | Windows | Output file size |
|-------|--------:|:---:|
| Train | 392,142 | 2.8 GB |
| Validation | 57,576 | 418 MB |
| Test | 92,726 | 174 MB |

**MLP head trained on top** (Stage 19):

| Hyperparameter | Value |
|----------------|-------|
| Input dim | 4,096 (4 positions × 1,024) |
| Architecture | 3-layer MLP with dropout |
| Epochs | 60 (best checkpoint: epoch 8) |
| Learning rate | 1e-3 |
| Batch size | 512 |
| Optimizer | Adam |
| Seed | 42 |
| Val Macro-F1 | **0.7681** |

This path alone is a valid challenge submission. It is the foundation of our final result.

We also ran a hybrid variant (Stage 6) concatenating MOMENT embeddings with 354 hand-crafted statistical/spectral features and training an XGBoost head (300 trees, max depth 6, learning rate 0.1), reaching F1=0.7329.

---

## Step 2 — Auxiliary Deep Models (supporting ensemble members)

To improve ensemble diversity we trained five scratch-built models on the same 392,142 train windows. Per challenge rules these are **supporting members**, not the core claim. All five use **focal loss (γ=2) + balanced batch sampler** — critical for the Run class (4.3% of train), which rose from F1=0.48 to F1=0.94.

### InceptionTime (Stage 12)

| Hyperparameter | Value |
|----------------|-------|
| Architecture | 6 inception blocks, 32 filters per branch |
| Input | (9, 500) — 9 channels, 500 time steps |
| Epochs | 100 (best: epoch 18) |
| Batch size | 512 |
| Learning rate | 1e-3 (Adam) |
| Loss | Focal, γ=2 |
| Sampler | Balanced (equal class frequency per batch) |
| Seed | 42 |
| Val Macro-F1 | **0.7726** |

### IMUFormer (Stage 12)

| Hyperparameter | Value |
|----------------|-------|
| Architecture | Transformer encoder, d=128, 2 layers, 4 heads |
| Input | (9, 500) |
| Epochs | 60 (best: epoch 29) |
| Batch size | 512 |
| Learning rate | 3e-4 (Adam) |
| Loss | Focal, γ=2 |
| Sampler | Balanced; stratified train limit 40,000 windows |
| Seed | 42 |
| Val Macro-F1 | **0.7163** |

### SpectrogramCNN (Stage 14)

| Hyperparameter | Value |
|----------------|-------|
| Architecture | Log-mel spectrogram (nfft=64, hop=16) → 2D CNN |
| Input | (9, 500) converted to (9, F, T) spectrograms |
| Epochs | 80 (best: epoch 20) |
| Batch size | 256 |
| Loss | Focal, γ=2 |
| Sampler | Balanced |
| Seed | 42 |
| Val Macro-F1 | **0.7590** |

### ResNet1D (Stage 15)

| Hyperparameter | Value |
|----------------|-------|
| Architecture | Deep residual 1D CNN, 64 filters, skip connections |
| Input | (9, 500) |
| Epochs | 100 (best: epoch 35) |
| Batch size | 512 |
| Loss | Focal, γ=2 |
| Sampler | Balanced |
| Seed | 42 |
| Val Macro-F1 | **0.7740** (best individual model) |

### MVPF v2 — Multi-View Position Fusion (Stage 18)

| Hyperparameter | Value |
|----------------|-------|
| Architecture | Cross-position fusion transformer; fusion dim 256, base filters 64, 8 attention heads, 3 transformer layers |
| Input | (4, 9, 500) — all 4 positions stacked |
| Epochs | 80 (best: epoch 27) |
| Batch size | 256 |
| Loss | Focal, γ=2 |
| Sampler | Balanced |
| SWA | Stochastic weight averaging, start epoch 50 |
| Augmentation | Random rotation across positions |
| Seed | 42 |
| Val Macro-F1 | **0.7678** |

---

## Step 3 — LightGBM Meta-Blend (Stage 16)

Each of the 6 models (MOMENT-MLP + 5 auxiliary) produces **8 class probabilities** per window via inference with test-time augmentation (TTA n=3: jitter σ=0.02, scale ∈ [0.9, 1.1]). These are stacked into a **48-column feature matrix** (6 × 8) and a LightGBM classifier is trained as the final decision layer.

| Hyperparameter | Value |
|----------------|-------|
| Meta-learner | LightGBM classifier |
| Input features | 48 (6 models × 8 class probabilities) |
| Number of trees | 500 |
| Learning rate | 0.05 |
| Train split | 80% of validation set — 46,060 windows (stratified) |
| Holdout split | 20% of validation set — 11,516 windows (unseen) |
| TTA on base models | n=3 (jitter + scale, applied at inference) |
| Seed | 42 |
| Holdout Val Macro-F1 | **0.9490** |

**Per-class F1 on holdout:**

| Still | Walking | Run | Bike | Car | Bus | Train | Metro |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.95 | 0.95 | 0.99 | 0.97 | 0.98 | 0.95 | 0.89 | 0.91 |

The +17.5 pp gain over the best individual model comes from LightGBM learning which model to trust per class — MOMENT embeddings are most informative for transport modes (Train/Metro) while ResNet1D and InceptionTime are strongest for motion-intense activities (Run/Bike).

---

## Step 4 — Why We Did NOT Apply Temporal Smoothing

We explored HMM Viterbi and BiLSTM smoothing (Stage 20), which improved **validation** F1 to 0.9513. We did not apply them to the test submission for a hard technical reason:

**Test rows are shuffled.** Three independent proofs:

1. The HDF5 `test/data` key is a single flat array — no timestamps, no session boundaries, no position split.
2. Empirical check: lag-1 autocorrelation across 100 consecutive test window boundaries = **−0.158**. For ordered 100 Hz IMU data this should be ~0.997. A value near zero confirms random ordering.
3. Project documentation and the `smooth_predictions.py` shuffle guard independently document this constraint.

Applying Viterbi or BiLSTM to shuffled windows would propagate hidden states across unrelated activities, corrupting predictions. The Stage 20 val gains are internal diagnostics only.

---

## Final Submission

| Field | Value |
|-------|-------|
| File | `FeatureFlyers_blend_s16_lgbm.txt` |
| Format | 92,726 lines × 500 comma-separated integers (1–8) |
| Total predictions | 46,363,000 |
| Holdout Val Macro-F1 | **0.9490** |
| Method | Foundation-enhanced ensemble (frozen MOMENT + auxiliary models + LightGBM) |
| Foundation weights updated | **No** |
| Temporal smoothing applied | **No** |

---

## One-Line Summary

> We feed frozen MOMENT-1-large embeddings through a lightweight MLP head, combine the resulting probabilities with five auxiliary sensor models via LightGBM stacking, and submit the meta-blend predictions — achieving holdout Val Macro-F1 = **0.9490** without fine-tuning any foundation model weights or applying temporal smoothing to the shuffled test set.
