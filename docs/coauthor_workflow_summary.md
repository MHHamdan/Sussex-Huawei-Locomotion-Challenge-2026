# FeatureFlyers — Full Pipeline Guide for Co-Authors

## Challenge Context

**Competition**: Sussex-Huawei Locomotion Challenge 2026 (SHL 2026)
**Task**: 8-class locomotion mode recognition from raw 9-axis IMU sensor data
**Classes**: Still (1), Walking (2), Run (3), Bike (4), Car (5), Bus (6), Train (7), Metro (8)
**Evaluation metric**: Macro-F1 (equal weight per class regardless of frequency)
**Submission format**: One text file, 92,726 lines, each containing 500 comma-separated integer labels (1–8)

### The Competition Rule That Shapes Everything

> *"SHL 2026 is foundation-model focused. The final pipeline must clearly leverage an existing pre-trained foundation model in a frozen manner. Foundation model weights must not be fine-tuned or retrained. Only lightweight task-specific components such as heads, classifiers, calibration layers, or meta-learners can be trained. Deep models trained from scratch can be used as auxiliary baselines or supporting ensemble members, but they should not be presented as the core challenge solution by themselves."*

This means our submission must be described as:

**"A foundation-enhanced ensemble combining frozen MOMENT-1-large representations with lightweight trained heads and auxiliary sensor models."**

---

## System Overview

```
RAW DATA (HDF5)
      │
      ├─── Hand-crafted features (statistical + spectral)
      │         └──► XGBoost baseline (Stage 5)          F1 = 0.648
      │
      ├─── Frozen MOMENT-1-large embeddings (4096-d)
      │         ├──► + stat features → XGBoost (Stage 6)  F1 = 0.733
      │         └──► MLP head (Stage 19) ◄── CORE PATH    F1 = 0.768
      │
      ├─── Scratch-trained auxiliary models (Stages 12–18)
      │         ├── InceptionTime                          F1 = 0.773
      │         ├── IMUFormer                              F1 = 0.716
      │         ├── SpectrogramCNN                         F1 = 0.759
      │         ├── ResNet1D                               F1 = 0.774
      │         └── MVPF v2                                F1 = 0.768
      │
      └─── LightGBM meta-blend (Stage 16)
                └── 6 models × 8 class probs = 48 features
                    Holdout Val Macro-F1 = 0.9490  ◄── FINAL SUBMISSION
```

---

## Part 1 — The Dataset

### Raw Data Structure

The entire dataset is stored as a single HDF5 file: `dataset/processed/shl2026.hdf5`

```
shl2026.hdf5
├── train/
│   ├── Bag/   data: (N_bag, 500, 9)   labels: (N_bag,)
│   ├── Hand/  data: (N_hand, 500, 9)  labels: (N_hand,)
│   ├── Hips/  data: (N_hips, 500, 9)  labels: (N_hips,)
│   └── Torso/ data: (N_torso, 500, 9) labels: (N_torso,)
├── validation/
│   └── Bag/   data: (57576, 500, 9)   labels: (57576,)
└── test/
    └── data:  (46363000, 9)            ← flat, no windows, no labels
```

**Critical test-set fact**: The test split is stored as a single flat `(46,363,000, 9)` array — 46 million raw samples with no position labels, no timestamps, no session boundaries. We reshape it into non-overlapping 500-sample windows to get 92,726 windows. The order of these windows is **shuffled** — confirmed by lag-1 autocorrelation across window boundaries = −0.158 (expected ~0.997 for ordered 100 Hz IMU data).

### Windowing

| Parameter | Value |
|-----------|-------|
| Window size | 500 samples = 5 seconds at 100 Hz |
| Hop size (train/val) | 250 samples = 50% overlap |
| Hop size (test) | 500 samples = non-overlapping (no overlap possible without temporal order) |

### Split Sizes

| Split | Windows | Positions | Labels | Notes |
|-------|--------:|-----------|:------:|-------|
| Train | 392,142 | 4 (Bag, Hand, Hips, Torso) | Yes | Used to train all base models |
| Validation | 57,576 | 1 (Bag only) | Yes | 80% trains LightGBM meta-learner; 20% holdout F1 |
| Test | 92,726 | 1 (single flat array) | **No** | Submitted to organisers for official scoring |

### Sensor Channels

Each window has shape `(500, 9)` — 500 time steps × 9 channels:

| Channels | Sensor | Axes |
|----------|--------|------|
| 0–2 | Accelerometer | x, y, z |
| 3–5 | Gyroscope | x, y, z |
| 6–8 | Magnetometer | x, y, z |

### Class Imbalance

| Class | Train freq | Notes |
|-------|:---:|-------|
| Still | ~22% | Easy — very low variance |
| Walking | ~18% | Easy — strong step cadence |
| Run | **4.3%** | Hard — rare; F1=0.48 without balanced sampler |
| Bike | ~8% | Medium — periodic cadence |
| Car | ~14% | Hard — passive, vibration-only cues |
| Bus | ~6% | Hard — similar to Car/Train |
| Train | ~15% | Very hard — smooth ride, similar to Metro |
| Metro | ~15% | Very hard — similar to Train |

Run (4.3%) is the most imbalanced class and the primary motivation for the balanced sampler used in all deep models.

---

## Part 2 — Hand-Crafted Statistical / Spectral Features (Stage 5)

Before any deep learning, we extract 354 interpretable features per window per position. These are cached as `.npz` files and reused by multiple later stages.

### Feature Categories

| Category | Features | Examples |
|----------|:--------:|---------|
| Time-domain statistics | ~108 | Mean, std, min, max, RMS, skewness, kurtosis, zero-crossing rate, IQR — per channel |
| Frequency-domain (FFT) | ~180 | Top-20 dominant frequencies + magnitudes, spectral entropy, spectral centroid — per channel |
| Cross-channel | ~66 | Pairwise correlations between accelerometer axes, gyroscope axes |
| **Total** | **354** | Per window, per position |

### XGBoost Baselines (Stage 5)

| Run | Features | Val Macro-F1 |
|-----|---------|:---:|
| Bag only | 354 | 0.6324 |
| Pool (4 pos) | 4 × 354 = 1,416 | 0.6389 |
| Pool full train | 4 × 354 = 1,416 | 0.6481 |

**Hyperparameters** (XGBoost):

| Parameter | Value |
|-----------|-------|
| n_estimators | 300 |
| max_depth | 6 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| seed | 42 |

**Script**: `scripts/train_baseline.py`
**Cache**: `dataset/processed/features/{split}_{position}.npz`

---

## Part 3 — Foundation Model: MOMENT-1-large (Core)

### What is MOMENT?

MOMENT-1-large is a 341-million-parameter time-series foundation model pre-trained by MIT/Google on a large and diverse corpus of time-series data (ECG, motion, climate, traffic, finance). It uses a masked-patch Transformer architecture (similar to masked autoencoders for images) that learns general temporal representations.

**Key property for SHL 2026**: MOMENT weights are **never updated** in our pipeline — they are used purely as a frozen feature extractor. This makes it the compliant foundation-model core required by the challenge rules.

### Embedding Extraction

Each 500-sample window is divided into patches and passed through the MOMENT encoder. It returns a **1024-dimensional embedding** representing the global temporal pattern of that window.

We extract embeddings independently for each of the 4 body positions. For test (single position), embeddings are extracted from the flat array reshaped into 92,726 windows.

| Parameter | Value |
|-----------|-------|
| Model | `moment-research/MOMENT-1-large` |
| Parameters | 341M (frozen) |
| Output per position | 1,024-d float16 |
| Positions concatenated | 4 → **4,096-d** per window |
| Batch size | 64 |
| Precision | fp16 (half precision) |
| Script | `scripts/extract_moment_embeddings.py` |

**Output files** (local only, not committed — too large):

| File | Shape | Size |
|------|-------|-----:|
| `moment_embeddings/train_embeddings.npz` | (392142, 4, 1024) float16 | 2.8 GB |
| `moment_embeddings/validation_embeddings.npz` | (57576, 4, 1024) float16 | 418 MB |
| `moment_embeddings/test_embeddings.npz` | (92726, 4, 1024) float16 | 174 MB |

### Stage 6 — MOMENT + Statistical Hybrid → XGBoost

The earliest foundation-model result: concatenate the 1024-d MOMENT embedding (Bag position) with 354 hand-crafted statistical features → 1,378-d vector → XGBoost head.

| Run | Train samples | Val Macro-F1 |
|-----|-------------:|:---:|
| Bag position, 20k samples | 20,000 | 0.7329 |
| Pool all positions, strat 40k | ~160,000 | 0.6970 |

**Hyperparameters** (Stage 6 XGBoost):

| Parameter | Value |
|-----------|-------|
| n_estimators | 300 |
| max_depth | 6 |
| learning_rate | 0.1 |
| Feature dim | 1,024 (MOMENT) + 354 (stat) = 1,378 |
| Seed | 42 |

**Script**: `scripts/train_foundation_head.py`

### Stage 19 — MOMENT → MLP Head (Primary Foundation Path)

The best single-model result on the pure foundation path. Takes the 4,096-d frozen embedding (4 positions concatenated) and trains a 3-layer MLP classifier.

**Architecture**:
```
Input (4096-d)
    → Linear(4096, 1024) → BatchNorm → GELU → Dropout(0.3)
    → Linear(1024, 512)  → BatchNorm → GELU → Dropout(0.3)
    → Linear(512, 8)
    → Softmax
```

**Hyperparameters**:

| Parameter | Value |
|-----------|-------|
| Input dim | 4,096 (4 positions × 1,024-d MOMENT) |
| Hidden layers | 1,024 → 512 |
| Activation | GELU |
| Normalisation | BatchNorm after each linear layer |
| Dropout | 0.3 |
| Loss | CrossEntropy |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| LR schedule | Cosine annealing, min LR = 1e-5 |
| Batch size | 512 |
| Epochs | 60 (best checkpoint: **epoch 8**) |
| Trainable params | ~2.1M |
| Seed | 42 |
| Val Macro-F1 | **0.7681** |

**Per-class F1 (best checkpoint, epoch 8)**:

| Still | Walking | Run | Bike | Car | Bus | Train | Metro |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.85 | 0.92 | 0.93 | 0.88 | 0.79 | 0.65 | 0.54 | 0.59 |

Train (0.54) and Metro (0.59) are hardest — these are passive transport modes with very similar IMU signatures.

**Script**: `scripts/train_stage19_moment_mlp.py`
**Output dir**: `outputs/execution-output/moment_large_mlp_ep60_lr1e-03_bs512/`

> **This model alone is a valid SHL 2026 submission.** It uses only frozen foundation model weights + a lightweight MLP head. Val Macro-F1 = 0.7681.

---

## Part 4 — Auxiliary Deep Models (Supporting Ensemble Members)

Five scratch-trained models trained on the full 392,142-window training set. Per challenge rules, these are *supporting members* in the ensemble, not the primary claim. All five share the same training recipe:

- **Loss**: Focal loss with γ=2 (down-weights easy examples, focuses on hard/rare classes)
- **Sampler**: Balanced batch sampler (each batch contains equal representation of all 8 classes, regardless of dataset frequency)
- **Why**: Run class (4.3% of train) had F1=0.48 with standard CrossEntropy; balanced sampler raises it to F1=0.94

### 4.1 InceptionTime (Stage 12)

InceptionTime applies multiple parallel 1D convolutions with different kernel sizes (short, medium, long) to capture temporal patterns at multiple scales simultaneously. A bottleneck layer then fuses the multi-scale features.

**Architecture**:
```
Input (9, 500)
    → [6 × InceptionBlock(filters=32, kernels=[10,20,40], bottleneck=32)]
    → GlobalAveragePool
    → Linear(128, 8)
```

**Hyperparameters**:

| Parameter | Value |
|-----------|-------|
| Inception blocks | 6 |
| Filters per branch | 32 |
| Kernel sizes | 10, 20, 40 (short/medium/long) |
| Bottleneck size | 32 |
| Total params | ~492K |
| Loss | Focal, γ=2.0 |
| Sampler | Balanced |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 512 |
| Epochs | 100 (best: **epoch 18**) |
| Seed | 42 |
| Val Macro-F1 | **0.7726** |

**Script**: `scripts/train_stage8_inception.py` (with `--focal --focal-gamma 2.0 --balanced-sampler`)
**Output dir**: `outputs/execution-output/inception_posPool_nb32_d6_sfull_focal_g2.0_balsampler_ep100_bs512/`

### 4.2 IMUFormer (Stage 12)

A Transformer encoder that treats the 500-sample window as a sequence and applies global self-attention. Unlike convolutions, it can model long-range temporal dependencies across the full 5-second window.

**Architecture**:
```
Input (9, 500)
    → Patch embedding + positional encoding
    → [2 × TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256)]
    → CLS token classification head
    → Linear(128, 8)
```

**Hyperparameters**:

| Parameter | Value |
|-----------|-------|
| Model dimension (d) | 128 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Feedforward dim | 256 |
| Loss | Focal, γ=2.0 |
| Sampler | Balanced; stratified train limit 40,000 windows/position |
| Class weights | Off (no-class-weights flag) |
| Optimizer | Adam |
| Learning rate | 3e-4 |
| Batch size | 512 |
| Epochs | 60 (best: **epoch 29**) |
| Seed | 42 |
| Val Macro-F1 | **0.7163** |

**Script**: `scripts/train_imu_former.py` (with `--focal --focal-gamma 2.0 --balanced-sampler --no-class-weights`)
**Output dir**: `outputs/execution-output/imuformer_d128tf2_posPool_sfull_strat40000_focal_g2.0_balsampler_nocw_ep60/`

### 4.3 SpectrogramCNN (Stage 14)

Converts the time-series into a log-mel spectrogram (frequency vs. time image) and applies a 2D CNN. Captures frequency-domain patterns — e.g., the dominant step cadence of walking (~1.8 Hz), cycling cadence (~1.2 Hz) — that are invisible to time-domain models.

**Architecture**:
```
Input (9, 500)
    → STFT(n_fft=64, hop=16) → log-mel spectrogram (9, F, T)
    → [Conv2D blocks with BatchNorm + MaxPool]
    → GlobalAveragePool
    → Linear(256, 8)
```

**Hyperparameters**:

| Parameter | Value |
|-----------|-------|
| FFT size | 64 |
| Hop length | 16 |
| Output spectrogram shape | (9, 33, 32) approx |
| Loss | Focal, γ=2.0 |
| Sampler | Balanced |
| Optimizer | Adam |
| Batch size | 256 |
| Epochs | 80 (best: **epoch 20**) |
| Seed | 42 |
| Val Macro-F1 | **0.7590** |

**Script**: `scripts/train_spectrogram_cnn.py` (with `--focal --focal-gamma 2.0 --balanced-sampler`)
**Output dir**: `outputs/execution-output/spectrogramcnn_posPool_nfft64_hop16_sfull_focal_g2.0_balsampler_ep80_bs256/`

### 4.4 ResNet1D (Stage 15)

A deep 1D residual network with skip connections that allow gradients to flow cleanly through many layers. The best-performing individual model — residual connections prevent vanishing gradients and allow the network to learn very deep temporal patterns.

**Architecture**:
```
Input (9, 500)
    → Conv1D(9, 64, k=7) stem
    → [ResBlock(64) × 3] → MaxPool
    → [ResBlock(128) × 4] → MaxPool
    → [ResBlock(256) × 6] → MaxPool
    → [ResBlock(512) × 3]
    → GlobalAveragePool
    → Linear(512, 8)
```

**Hyperparameters**:

| Parameter | Value |
|-----------|-------|
| Base filters | 64 |
| Total depth | ~34 layers |
| Loss | Focal, γ=2.0 |
| Sampler | Balanced |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 512 |
| Epochs | 100 (best: **epoch 35**) |
| Seed | 42 |
| Val Macro-F1 | **0.7740** (best individual model) |

**Script**: `scripts/train_resnet1d.py` (with `--focal --focal-gamma 2.0 --balanced-sampler`)
**Output dir**: `outputs/execution-output/resnet1d_posPool_f64_sfull_focal_g2.0_balsampler_ep100_bs512/`

### 4.5 MVPF v2 — Multi-View Position Fusion (Stage 18)

The only model that explicitly sees all 4 body positions simultaneously and learns cross-position interactions. A position-specific encoder processes each of the 4 positions, then a cross-position Transformer fuses them. Stochastic Weight Averaging (SWA) smooths the final weight trajectory for better generalisation.

**Architecture**:
```
Input (4, 9, 500) — 4 positions stacked
    → [Position encoder: Conv1D stem + ResBlocks, output: (4, 64, T)]
    → Cross-position Transformer(d=256, heads=8, layers=3)
    → Classification head Linear(256, 8)
```

**Hyperparameters**:

| Parameter | Value |
|-----------|-------|
| Fusion dimension | 256 |
| Base filters | 64 |
| Attention heads | 8 |
| Transformer layers | 3 |
| Loss | Focal, γ=2.0 |
| Sampler | Balanced |
| Augmentation | Random rotation across positions (train only) |
| SWA | Stochastic weight averaging; start epoch 50 |
| Optimizer | Adam |
| Batch size | 256 |
| Epochs | 80 (best: **epoch 27**) |
| Seed | 42 |
| Val Macro-F1 | **0.7678** |

**Script**: `scripts/train_mvpf.py` (with `--focal --focal-gamma 2.0 --balanced-sampler --swa --swa-start 50 --rotation-aug`)
**Output dir**: `outputs/execution-output/mvpf_v2_4pos_fd256_bf64_h8tf3_sfull_focal_g2.0_balsampler_swa50_rotaug_ep80_bs256/`

---

## Part 5 — LightGBM Meta-Blend (Stage 16, Final Submission)

### Concept

Each of the 6 base models produces **8 softmax probabilities** per window — one per activity class. These probabilities are the model's "belief distribution" over all classes. We stack them into a 48-column table and train a LightGBM classifier that learns *which model to trust, per class, per context*.

```
Window W
    → MOMENT-MLP    → [0.02, 0.01, 0.01, 0.01, 0.85, 0.05, 0.03, 0.02]  ← 8 probs
    → InceptionTime → [0.03, 0.02, 0.01, 0.02, 0.78, 0.08, 0.04, 0.02]  ← 8 probs
    → IMUFormer     → [0.05, 0.03, 0.01, 0.01, 0.72, 0.10, 0.05, 0.03]  ← 8 probs
    → SpectrogramCNN→ [0.04, 0.02, 0.01, 0.02, 0.80, 0.07, 0.02, 0.02]  ← 8 probs
    → ResNet1D      → [0.02, 0.01, 0.01, 0.01, 0.88, 0.04, 0.02, 0.01]  ← 8 probs
    → MVPF v2       → [0.03, 0.02, 0.01, 0.01, 0.82, 0.06, 0.03, 0.02]  ← 8 probs
                                                                           ─────────
                                                                           48 features
    → LightGBM → "Car" (class 5)
```

### Test-Time Augmentation (TTA)

Before extracting probabilities, each PyTorch model sees each window 3 times with slight perturbations (n=3):
- **Jitter**: add Gaussian noise with σ=0.02 to each sample
- **Scale**: multiply all values by a random factor ∈ [0.9, 1.1]
- The 3 probability vectors are averaged → more stable predictions

TTA is NOT applied to MOMENT-MLP (embeddings are pre-extracted) or to the LightGBM meta-learner.

### Meta-Learner Training

| Parameter | Value |
|-----------|-------|
| Meta-learner | LightGBM multiclass classifier |
| Input | 48 features (6 models × 8 softmax probabilities) |
| Number of trees | 500 |
| Learning rate | 0.05 |
| Num leaves | 63 |
| Min child samples | 20 |
| Subsample | 0.8 |
| Column subsample | 0.8 |
| Training data | 80% of labelled val set = **46,060 windows** (stratified split) |
| Holdout data | 20% of labelled val set = **11,516 windows** (never seen during training) |
| Seed | 42 |

> **Why use validation for training the meta-learner?** The 6 base models were trained on the train split. When they generate probabilities on the val split, these are genuine out-of-training predictions — there is no data leakage. The meta-learner learns from the base models' generalisation behaviour, not their memorisation.

### Results

| Metric | Value |
|--------|:---:|
| Holdout Val Macro-F1 | **0.9490** |
| Full-val Macro-F1 (biased — includes training portion) | 0.9898 |
| Val Accuracy | 94.1% |

**Per-class F1 (holdout, 11,516 windows)**:

| Still | Walking | Run | Bike | Car | Bus | Train | Metro |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.95 | 0.95 | **0.99** | 0.97 | 0.98 | 0.95 | 0.89 | 0.91 |

The +17.5 pp gain over the best individual model (ResNet1D, 0.7740) comes from LightGBM routing each window to the model with the most reliable signal for that class. MOMENT embeddings are most informative for transport disambiguation (Train/Metro); ResNet1D and InceptionTime dominate for physical motion (Run/Bike).

**Script**: `scripts/train_stage16_meta_blend.py`
**Output dir**: `outputs/execution-output/meta_blend_s16_lgbm_6models/`
**Submission**: `outputs/execution-output/submissions/FeatureFlyers_blend_s16_lgbm.txt`

---

## Part 6 — Why Temporal Smoothing Was Excluded

### What We Tried (Stage 20)

Locomotion changes slowly — a person walking typically walks for several minutes, not fractions of a second. We explored two methods that exploit this temporal autocorrelation:

**HMM Viterbi**: Fits a first-order Hidden Markov Model where the transition matrix encodes how likely each activity is to follow each other. Viterbi decoding then finds the globally optimal label sequence. Val F1: **0.9298** (+0.71 pp over base average).

**BiLSTM**: Trains a bidirectional LSTM on MOMENT embeddings in temporal sequence to model longer-range dependencies. Val F1: **0.9513** (+1.23 pp over base average).

### Why Not Applied to Test

**The test rows are shuffled.** The HDF5 test data is a single flat array with no ordering information. Three independent proofs:

| Evidence | Detail |
|----------|--------|
| HDF5 structure | `test/data` is `(46363000, 9)` — no timestamps, session IDs, position splits, or ordering keys |
| Empirical autocorrelation | Lag-1 autocorrelation across 100 consecutive test window boundaries = **−0.158** (expected ~0.997 for continuous 100 Hz IMU recordings) |
| Code guard | `scripts/smooth_predictions.py` contains a shuffle detection guard that blocks smoothed output when temporal violations are detected |

Applying Viterbi or BiLSTM to shuffled windows would propagate hidden states from one activity into the next unrelated window, **corrupting predictions**. The Stage 20 results are validation diagnostics only.

---

## Part 7 — Ablation Study: Hybrid Features (Stages 22–23)

We tested whether combining frozen MOMENT embeddings with hand-crafted statistical features into a single model improves on MOMENT alone (Stage 19, F1=0.7681).

**Combined vector**: 4,096-d MOMENT + 1,416-d stat (4 pos × 354) = **5,512-d**

| Stage | Head | Val F1 | vs Stage 19 | Key finding |
|-------|------|:---:|:---:|------------|
| 22 | LightGBM (500 trees, lr=0.05) | 0.6539 | −11.4 pp | Trees cannot exploit dense embeddings (axis-aligned splits fail on jointly-encoded 1024-d vectors) |
| 23 | MLP (5512→1024→512→256→8) | 0.7493 | −1.9 pp | MLP uses embeddings correctly but stat features dilute transport signal |
| 19 | MLP (4096→1024→512→8) | **0.7681** | baseline | MOMENT alone is near-optimal for transport disambiguation |

**Positive finding from Stage 23**: Run F1 = **0.945** with hybrid features vs. ~0.71 with MOMENT alone. Stat features (acceleration variance, FFT step cadence) are highly discriminative for running and the MLP exploits this.

**Design conclusion**: Combine feature types at the **ensemble level** (separate models in Stage 16), not at the **feature-concatenation level**. MOMENT handles transport; stat-feature models handle motion physics. LightGBM stacking routes each query to the appropriate specialist.

---

## Part 8 — Complete Model Comparison

| Stage | Model | Val Macro-F1 | Category |
|-------|-------|:---:|---------|
| 5 | XGBoost — stat features only | 0.6481 | Baseline |
| 6 | MOMENT + stat → XGBoost | 0.7329 | Foundation hybrid |
| 8 | InceptionTime (CE loss) | 0.7265 | Auxiliary |
| 8 | IMUFormer (CE loss) | 0.7125 | Auxiliary |
| 12 | InceptionTime (focal + balanced) | 0.7726 | Auxiliary |
| 12 | IMUFormer (focal + balanced) | 0.7163 | Auxiliary |
| 14 | SpectrogramCNN (focal + balanced) | 0.7590 | Auxiliary |
| 15 | ResNet1D (focal + balanced) | 0.7740 | Auxiliary — best individual |
| 18 | MVPF v2 (focal + balanced + SWA) | 0.7678 | Auxiliary |
| **19** | **MOMENT-MLP** (frozen, 4096-d) | **0.7681** | **Foundation core** |
| 22 | MOMENT + stat → LightGBM (ablation) | 0.6539 | Ablation |
| 23 | MOMENT + stat → MLP (ablation) | 0.7493 | Ablation |
| **16** | **LightGBM meta-blend (6 models)** | **0.9490** | **Final submission** |

---

## Part 9 — Reproducing the Results

### Prerequisites

```bash
git clone <repo>
cd Sussex-Huawei-Locomotion-Challenge-2026
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Place shl2026.hdf5 in dataset/processed/
```

### Minimal reproduction (foundation-model path only)

```bash
# 1. Extract MOMENT embeddings (requires GPU, ~20 min total)
CUDA_VISIBLE_DEVICES=0 python scripts/extract_moment_embeddings.py \
    --hdf5 dataset/processed/shl2026.hdf5 --split train   --batch-size 64 --device cuda:0 \
    --out-dir outputs/execution-output/moment_embeddings
CUDA_VISIBLE_DEVICES=0 python scripts/extract_moment_embeddings.py \
    --hdf5 dataset/processed/shl2026.hdf5 --split validation --batch-size 64 --device cuda:0 \
    --out-dir outputs/execution-output/moment_embeddings
CUDA_VISIBLE_DEVICES=0 python scripts/extract_moment_embeddings.py \
    --hdf5 dataset/processed/shl2026.hdf5 --split test   --batch-size 64 --device cuda:0 \
    --out-dir outputs/execution-output/moment_embeddings

# 2. Train MOMENT-MLP head (~8 min)
CUDA_VISIBLE_DEVICES=0 python scripts/train_stage19_moment_mlp.py \
    --embeddings-dir outputs/execution-output/moment_embeddings \
    --epochs 60 --lr 1e-3 --batch-size 512 --seed 42 \
    --out-dir outputs/execution-output/moment_large_mlp_ep60_lr1e-03_bs512

# Expected: Val Macro-F1 = 0.7681 at epoch 8
```

### Full reproduction (Stage 16 ensemble — all 6 models)

See `docs/reproducibility_commands.md` for the complete step-by-step commands for all stages (Steps 11–18). The full pipeline requires ~4 GPU-hours across 4 GPUs.

### Verifying the submission

```bash
python scripts/verify_submission.py \
    --input outputs/execution-output/submissions/FeatureFlyers_blend_s16_lgbm.txt
# Expected: PASS — 92,726 lines, 500 values/line, labels 1–8
```

---

## Final Submission Details

| Field | Value |
|-------|-------|
| **File** | `FeatureFlyers_blend_s16_lgbm.txt` |
| **Format** | 92,726 lines × 500 comma-separated integers (1–8) |
| **Total predictions** | 46,363,000 |
| **File size** | 88.4 MB |
| **Holdout Val Macro-F1** | **0.9490** |
| **Foundation weights updated** | No — MOMENT-1-large fully frozen |
| **Temporal smoothing** | Not applied — test rows are shuffled |
| **Challenge compliance** | Foundation-enhanced ensemble; MOMENT is core; scratch models are auxiliary |

---

## One-Paragraph Summary

We build a **foundation-enhanced ensemble** for SHL 2026 locomotion recognition. The core component is **MOMENT-1-large** (341M parameters, frozen), which extracts 1,024-d patch embeddings per body position; a lightweight 3-layer MLP head trained on the concatenated 4,096-d representation achieves Val Macro-F1=0.7681 as a standalone submission. Five auxiliary scratch-trained models — InceptionTime, IMUFormer, SpectrogramCNN, ResNet1D, and MVPF v2 — are trained with focal loss and balanced sampling to handle class imbalance, and contribute complementary inductive biases (multi-scale temporal, global attention, frequency-domain, residual depth, cross-position fusion). A **LightGBM meta-learner** is trained on the stacked 48-dimensional softmax probability outputs of all six models using 80% of the labelled validation set, and evaluated on the remaining unseen 20% holdout. The meta-blend reaches holdout Val Macro-F1=**0.9490** (+17.5 pp over the best individual model). Temporal smoothing (HMM Viterbi, BiLSTM) is excluded from the test submission because the test rows are shuffled — confirmed empirically by a lag-1 autocorrelation of −0.158 across window boundaries, versus ~0.997 expected for ordered 100 Hz IMU data.
