# Reproducibility Commands — FeatureFlyers / SHL 2026

All commands run from the repository root. Local artefact paths that are
not committed are marked `<local>`.

---

## 0. Environment Setup

```bash
# Clone repository
git clone https://github.com/<org>/Sussex-Huawei-Locomotion-Challenge-2026.git
cd Sussex-Huawei-Locomotion-Challenge-2026

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 1. Sync Repo (existing clone)

```bash
git fetch origin
git checkout main
git pull --ff-only
```

---

## 2. Prepare Data

Place the original SHL 2026 zip archives in `dataset/archives/`:

```
dataset/archives/SHL-2026-Train_Bag.zip
dataset/archives/SHL-2026-Train_Hand.zip
dataset/archives/SHL-2026-Train_Hips.zip
dataset/archives/SHL-2026-Train_Torso.zip
dataset/archives/SHL-2026-Validation.zip
dataset/archives/SHL-2026-Test.zip
```

Extract and convert to HDF5:

```bash
# Extract all archives (adjust paths to your archive location)
python scripts/prepare_dataset.py --archive-dir dataset/archives --raw-dir dataset/raw

# Convert raw CSVs to single HDF5 file
python scripts/convert_to_hdf5.py \
    --raw-dir dataset/raw \
    --output dataset/processed/shl2026.hdf5

# Quick sanity check
python scripts/validate_hdf5.py --hdf5 dataset/processed/shl2026.hdf5
```

---

## 3. Validate HDF5

```bash
python scripts/validate_hdf5.py --hdf5 dataset/processed/shl2026.hdf5
```

Expected:
```
  train   : 4 positions, N_train windows
  validation: 1 position (Bag), 57,576 windows
  test    : (92,726, 500, 9) — no position labels
```

---

## 4. Run Feature Cache (Statistical + Spectral)

Pre-compute hand-crafted features for all splits and positions:

```bash
# All positions, all splits (train / validation / test)
python scripts/precompute_features.py \
    --hdf5 dataset/processed/shl2026.hdf5 \
    --out-dir dataset/processed/features \
    --splits train validation test \
    --positions Bag Hand Hips Torso

# Expected output layout:
#   dataset/processed/features/train_Bag.npz
#   dataset/processed/features/train_Hand.npz
#   ...
#   dataset/processed/features/test_Bag.npz   (no position label in HDF5 — Bag used as proxy)
```

---

## 5. Run Best XGBoost Baseline (Stage 5)

```bash
# Pool all 4 positions — full training set
python scripts/train_baseline.py \
    --hdf5 dataset/processed/shl2026.hdf5 \
    --feature-dir dataset/processed/features \
    --fusion pool \
    --seed 42 \
    --out-dir outputs/execution-output/xgb_pool_full

# Generate submission
python scripts/generate_submission.py \
    --model-path outputs/execution-output/xgb_pool_full/model.joblib \
    --output outputs/execution-output/submissions/FeatureFlyers_xgb_pool_full.txt
```

---

## 6. Cache MOMENT Embeddings (Stage 6 prerequisite)

```bash
# Val split — Bag position
python scripts/cache_moment_test_embeddings.py \
    --hdf5 dataset/processed/shl2026.hdf5 \
    --split validation --position Bag \
    --out-dir dataset/processed/embeddings \
    --device cuda:0

# Test split — Bag position
python scripts/cache_moment_test_embeddings.py \
    --hdf5 dataset/processed/shl2026.hdf5 \
    --split test --position Bag \
    --out-dir dataset/processed/embeddings \
    --device cuda:0

# Expected output:
#   dataset/processed/embeddings/validation_Bag_moment_normperwindow_mean_pool.npz
#   dataset/processed/embeddings/test_Bag_moment_normperwindow_mean_pool.npz
```

---

## 7. Train MOMENT Hybrid XGBoost Head (Stage 6)

```bash
python scripts/train_foundation_head.py \
    --fusion pool \
    --encoder moment \
    --norm per-window \
    --embed-strategy mean_pool \
    --head xgb \
    --hybrid-stat-features \
    --seed 42 \
    --strat-limit 40000 \
    --out-dir outputs/execution-output/foundation_moment_posPool_mean_pool_normperwindow_xgb_hybrid_sfull_strat40000 \
    --device cuda:0
```

---

## 8. Train InceptionTime (Stage 8)

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/train_stage8_inception.py \
    --hdf5 dataset/processed/shl2026.hdf5 \
    --position pool \
    --nb-filters 32 --depth 6 \
    --epochs 100 --batch-size 512 \
    --lr 1e-3 \
    --seed 42 \
    --out-dir outputs/execution-output/inception_posPool_nb32_d6_sfull_ep100_bs512 \
    2>&1 | tee outputs/execution-output/stage8_inception_pool_full.log
```

---

## 9. Train IMUFormer (Stage 8)

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/train_imu_former.py \
    --hdf5 dataset/processed/shl2026.hdf5 \
    --position pool \
    --d 128 --n-tf-layers 2 \
    --epochs 60 --batch-size 512 \
    --lr 3e-4 \
    --strat-limit 40000 \
    --seed 42 \
    --out-dir outputs/execution-output/imuformer_d128tf2_posPool_sfull_strat40000_ep60 \
    2>&1 | tee outputs/execution-output/stage8_imuformer_final.log
```

---

## 10. Run Stage 9 Ensemble (historical — superseded by Stage 16)

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/run_stage9_ensemble.py \
    --tta-n 5 --device cuda:0 \
    2>&1 | tee outputs/execution-output/stage9_ensemble_val.log
```

Expected val Macro-F1: **0.7833** (uniform weights); **0.7810** (weight-optimised). Superseded by Stage 16.

---

## 11. Train InceptionTime — Focal + Balanced Sampler (Stage 12)

Retrain with focal loss (γ=2) and balanced sampler to fix rare-class collapse (Run class).

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/train_stage8_inception.py \
    --hdf5 dataset/processed/shl2026.hdf5 \
    --position pool \
    --nb-filters 32 --depth 6 \
    --epochs 100 --batch-size 512 \
    --lr 1e-3 \
    --focal --focal-gamma 2.0 \
    --balanced-sampler \
    --seed 42 \
    --out-dir outputs/execution-output/inception_posPool_nb32_d6_sfull_focal_g2.0_balsampler_ep100_bs512 \
    2>&1 | tee outputs/execution-output/logs/stage12_inception_focal.log
```

Expected best val Macro-F1: **0.7726** (epoch 18).

---

## 12. Train IMUFormer — Focal + Balanced Sampler (Stage 12)

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/train_imu_former.py \
    --hdf5 dataset/processed/shl2026.hdf5 \
    --position pool \
    --d 128 --n-tf-layers 2 \
    --epochs 60 --batch-size 512 \
    --lr 3e-4 \
    --strat-limit 40000 \
    --focal --focal-gamma 2.0 \
    --balanced-sampler \
    --no-class-weights \
    --seed 42 \
    --out-dir outputs/execution-output/imuformer_d128tf2_posPool_sfull_strat40000_focal_g2.0_balsampler_nocw_ep60 \
    2>&1 | tee outputs/execution-output/logs/stage12_imuformer_focal.log
```

Expected best val Macro-F1: **0.7163** (epoch 29).

---

## 13. Train SpectrogramCNN (Stage 14)

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_spectrogram_cnn.py \
    --hdf5 dataset/processed/shl2026.hdf5 \
    --position pool \
    --nfft 64 --hop 16 \
    --epochs 80 --batch-size 256 \
    --focal --focal-gamma 2.0 \
    --balanced-sampler \
    --seed 42 \
    --out-dir outputs/execution-output/spectrogramcnn_posPool_nfft64_hop16_sfull_focal_g2.0_balsampler_ep80_bs256 \
    2>&1 | tee outputs/execution-output/logs/stage14_spectrogramcnn.log
```

Expected best val Macro-F1: **0.7590** (epoch 20).

---

## 14. Train ResNet1D (Stage 15)

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_resnet1d.py \
    --hdf5 dataset/processed/shl2026.hdf5 \
    --position pool \
    --filters 64 \
    --epochs 100 --batch-size 512 \
    --focal --focal-gamma 2.0 \
    --balanced-sampler \
    --seed 42 \
    --out-dir outputs/execution-output/resnet1d_posPool_f64_sfull_focal_g2.0_balsampler_ep100_bs512 \
    2>&1 | tee outputs/execution-output/logs/stage15_resnet1d.log
```

Expected best val Macro-F1: **0.7740** (epoch 35).

---

## 15. Train MVPF v2 (Stage 18)

4-position cross-fusion transformer with rotation augmentation and SWA.

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_mvpf.py \
    --hdf5 dataset/processed/shl2026.hdf5 \
    --fusion-dim 256 --base-filters 64 --heads 8 --tf-layers 3 \
    --epochs 80 --batch-size 256 \
    --focal --focal-gamma 2.0 \
    --balanced-sampler \
    --swa --swa-start 50 \
    --rotation-aug \
    --seed 42 \
    --out-dir outputs/execution-output/mvpf_v2_4pos_fd256_bf64_h8tf3_sfull_focal_g2.0_balsampler_swa50_rotaug_ep80_bs256 \
    2>&1 | tee outputs/execution-output/logs/stage18_mvpf_v2.log
```

Expected best val Macro-F1: **0.7678** (epoch 27).

---

## 16. Extract MOMENT-1-large Embeddings — All Splits (Stage 19 prerequisite)

Extracts `(N, 4, 1024)` float16 embeddings for train, validation, and test. One forward pass per split; test takes ~7 min on RTX 2080 Ti.

```bash
# Train split  (~392k windows → 2.8 GB)
CUDA_VISIBLE_DEVICES=0 python -u scripts/extract_moment_embeddings.py \
    --hdf5 dataset/processed/shl2026.hdf5 \
    --split train --batch-size 64 --device cuda:0 \
    --out-dir outputs/execution-output/moment_embeddings \
    2>&1 | tee outputs/execution-output/logs/moment_embed_train.log

# Validation split  (~58k windows → 418 MB)
CUDA_VISIBLE_DEVICES=0 python -u scripts/extract_moment_embeddings.py \
    --hdf5 dataset/processed/shl2026.hdf5 \
    --split validation --batch-size 64 --device cuda:0 \
    --out-dir outputs/execution-output/moment_embeddings \
    2>&1 | tee outputs/execution-output/logs/moment_embed_val.log

# Test split  (92,726 windows → 174 MB)
CUDA_VISIBLE_DEVICES=0 python -u scripts/extract_moment_embeddings.py \
    --hdf5 dataset/processed/shl2026.hdf5 \
    --split test --batch-size 64 --device cuda:0 \
    --out-dir outputs/execution-output/moment_embeddings \
    2>&1 | tee outputs/execution-output/logs/moment_embed_test.log
```

Expected output files:
```
outputs/execution-output/moment_embeddings/train_embeddings.npz      (392142, 4, 1024) float16
outputs/execution-output/moment_embeddings/validation_embeddings.npz  (57576, 4, 1024) float16
outputs/execution-output/moment_embeddings/test_embeddings.npz        (92726, 4, 1024) float16
```

---

## 17. Train MOMENT-MLP Head (Stage 19)

Trains a 3-layer MLP on the frozen 4096-d MOMENT embeddings (4 positions concatenated).

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_stage19_moment_mlp.py \
    --embeddings-dir outputs/execution-output/moment_embeddings \
    --epochs 60 --lr 1e-3 --batch-size 512 \
    --seed 42 \
    --out-dir outputs/execution-output/moment_large_mlp_ep60_lr1e-03_bs512 \
    2>&1 | tee outputs/execution-output/logs/stage19_moment_mlp.log
```

Expected best val Macro-F1: **0.7681** (epoch 8).

---

## 18. Run Stage 16 — 6-Model LightGBM Meta-Blend (Final Submission)

Requires all six base model checkpoints and MOMENT embeddings from steps 11–17.

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_stage16_meta_blend.py \
    --inception-dir   outputs/execution-output/inception_posPool_nb32_d6_sfull_focal_g2.0_balsampler_ep100_bs512 \
    --imuformer-dir   outputs/execution-output/imuformer_d128tf2_posPool_sfull_strat40000_focal_g2.0_balsampler_nocw_ep60 \
    --spectrogram-dir outputs/execution-output/spectrogramcnn_posPool_nfft64_hop16_sfull_focal_g2.0_balsampler_ep80_bs256 \
    --resnet1d-dir    outputs/execution-output/resnet1d_posPool_f64_sfull_focal_g2.0_balsampler_ep100_bs512 \
    --mvpf-dir        outputs/execution-output/mvpf_v2_4pos_fd256_bf64_h8tf3_sfull_focal_g2.0_balsampler_swa50_rotaug_ep80_bs256 \
    --moment-mlp-dir  outputs/execution-output/moment_large_mlp_ep60_lr1e-03_bs512 \
    --predict-test \
    --device cuda:0 \
    --tta-n 3 \
    2>&1 | tee outputs/execution-output/logs/stage16_6model.log
```

Expected holdout val Macro-F1: **0.9490**. Submission written to:
```
outputs/execution-output/submissions/FeatureFlyers_blend_s16_lgbm.txt
```

---

## 19. Verify Final Submission

```bash
python scripts/verify_submission.py \
    --input outputs/execution-output/submissions/FeatureFlyers_blend_s16_lgbm.txt
```

Expected:
```
  RESULT: PASS
  Lines     : 92,726  ✓
  Values/line: 500  ✓
  Label range: [1, 8]  ✓
```

---

## Notes

- All scripts default to `seed=42` (set in `configs/default.yaml`).
- GPU selection via `CUDA_VISIBLE_DEVICES=<id>` before launch; inside the script use `--device cuda:0`.
- Model checkpoints and embeddings are local-only; see `docs/final_submission_manifest.md` for paths.
- Do NOT apply temporal smoothing: test rows are shuffled. Cross-window boundary autocorrelation = -0.158 (expected ~0.997 for ordered data).
- Stage 20 (HMM, BiLSTM) val results are diagnostics only and must not be used for test submission.
