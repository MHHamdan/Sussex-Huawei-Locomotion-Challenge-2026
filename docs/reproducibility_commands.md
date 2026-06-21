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

## 10. Run Stage 9 Ensemble (Val only — calibration check)

```bash
# Requires: InceptionTime, IMUFormer checkpoints + MOMENT-XGB model + cached embeddings/features
CUDA_VISIBLE_DEVICES=2 python -u scripts/run_stage9_ensemble.py \
    --tta-n 5 --device cuda:0 \
    2>&1 | tee outputs/execution-output/stage9_ensemble_val.log
```

Expected val Macro-F1: **0.7833**

---

## 11. Generate Final Test Submission (Stage 9 + TTA 5)

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/run_stage9_ensemble.py \
    --tta-n 5 --device cuda:0 --predict-test \
    --output outputs/execution-output/submissions/FeatureFlyers_ensemble_s9_tta5.txt \
    2>&1 | tee outputs/execution-output/stage9_ensemble_final.log
```

---

## 12. Verify Submission

```bash
python scripts/verify_submission.py \
    --input outputs/execution-output/submissions/FeatureFlyers_ensemble_s9_tta5.txt
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
- Do NOT apply temporal smoothing: test rows are shuffled with no temporal order guarantee.
