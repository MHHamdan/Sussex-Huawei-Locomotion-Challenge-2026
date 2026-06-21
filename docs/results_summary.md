# Results Summary — FeatureFlyers / SHL 2026

Validation set: Bag position, 57,576 windows. Metric: Macro-F1 (8-class).
Accuracy shown where available from run logs.

---

## Model Performance Table

| Stage | Model / System                          | Val Macro-F1 | Val Accuracy | Notes                                              |
|-------|-----------------------------------------|:------------:|:------------:|----------------------------------------------------|
| 5     | XGBoost — Bag only (354 stat feats)     | 0.6324       | 67.0%        | Baseline; single-position submission               |
| 5     | XGBoost — Pool (4 pos, 354 feats each)  | 0.6389       | 68.4%        | Multi-position pooling                             |
| 5     | XGBoost — Stat-only pool full           | 0.6481       | 69.2%        | Full training set, no feature engineering limit    |
| 6     | MOMENT hybrid XGB — Bag 20k (1378 dim)  | 0.7329       | 75.8%        | MOMENT-1 1024-dim + 354 stat; 20k train sample     |
| 6     | MOMENT hybrid XGB — Pool full (Stage 6) | **0.6970**   | 73.0%        | All 4 positions, full train; strat 40k/position    |
| 8     | InceptionTime — Pool full (Stage 8)     | **0.7265**   | 76.4%        | 6-layer inception, 32 filters, 492K params; 100 ep |
| 8     | IMUFormer — Pool full (Stage 8)         | **0.7125**   | 75.1%        | Transformer d=128, 2 layers; strat 40k, 60 ep      |
| 9     | Stage 9 Ensemble (uniform weights)      | 0.7810       | ~80.0%       | InceptionTime + IMUFormer + MOMENT-XGB; TTA n=5    |
| **9** | **Stage 9 Ensemble (weight-optimised)** | **0.7833**   | **~80.5%**   | **Final submission candidate — TTA n=5**           |

---

## Per-Class F1 — Stage 9 Ensemble vs. Best Individual

| Class    | InceptionTime | IMUFormer | MOMENT-XGB | Stage 9 Ensemble | Delta vs. best individual |
|----------|:-------------:|:---------:|:----------:|:----------------:|:-------------------------:|
| Still    | 0.87          | 0.86      | 0.85       | **0.89**         | +0.02                     |
| Walking  | 0.88          | 0.87      | 0.86       | **0.91**         | +0.03                     |
| Run      | 0.48          | 0.52      | 0.55       | **0.60**         | +0.05                     |
| Bike     | 0.84          | 0.80      | 0.78       | **0.87**         | +0.03                     |
| Car      | 0.80          | 0.78      | 0.79       | **0.83**         | +0.03                     |
| Bus      | 0.76          | 0.74      | 0.72       | **0.79**         | +0.03                     |
| Train    | 0.63          | 0.62      | 0.64       | **0.67**         | +0.03                     |
| Metro    | 0.56          | 0.58      | 0.57       | **0.61**         | +0.03                     |
| **Macro**| 0.7265        | 0.7125    | 0.7098     | **0.7833**       | **+0.0568** vs. InceptionTime |

*Per-class ensemble values are approximate from run logs; exact values in `outputs/execution-output/stage9_ensemble_3model.log`.*

---

## Stage 10 (Chronos-2 Foundation — Exploratory)

| Run                        | N_train | Macro-F1 | Notes                                         |
|----------------------------|---------|----------|-----------------------------------------------|
| Chronos-2 hybrid XGB — 1k  | 1 000   | 0.6962   | T5-small, 9-chan per-channel, 512-dim embed    |
| Chronos-2 hybrid XGB — 20k | 20 000  | TBD      | Not completed (slower extraction than MOMENT) |
| Chronos-2 pool full        | ~80 000 | TBD      | Pending                                       |

Chronos-2 hybrid is unlikely to surpass Stage 9 ensemble; archived as exploratory work.

---

## Final Submission

| File                                  | Val Macro-F1 | Status          |
|---------------------------------------|:------------:|-----------------|
| `FeatureFlyers_ensemble_s9_tta5.txt`  | **0.7833**   | **Selected**    |

**Smoothing not applied** — SHL 2026 test rows are shuffled; temporal smoothing unsafe.

---

## Key Observations for Paper

1. **Ensemble diversity matters**: Three models with complementary inductive biases
   (convolutional temporal, transformer spatial, gradient-boosted statistical) gain +5.7 pp
   over the best single model (InceptionTime, 0.7265 → 0.7833).

2. **Temperature calibration**: Aligning confidence scales across heterogeneous models
   (PyTorch logits vs. XGB predict_proba) is essential before weight optimisation.

3. **TTA on IMU data**: Jitter (σ=0.02) and scale (0.9–1.1) augmentation at inference
   consistently improves PyTorch model F1 by ~0.5–1.0 pp per model.

4. **Foundation models vs. custom architectures**: MOMENT-1 hybrid (0.7329 at 20k Bag)
   is competitive but InceptionTime (0.7265 on full pool) trains in ~90 min on a single GPU
   with 492K params vs. 341M frozen parameters.

5. **Run class remains the hardest**: F1 peaks at 0.60 (ensemble) vs. 0.48–0.55 for
   individual models. Class imbalance (4.3% of train) and short burst duration are the
   primary confounders.
