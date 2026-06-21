# Final Submission Manifest — FeatureFlyers / SHL 2026

## Best Submission Candidate

| Field                  | Value                                                         |
|------------------------|---------------------------------------------------------------|
| **Team**               | FeatureFlyers                                                 |
| **Submission file**    | `FeatureFlyers_ensemble_s9_tta5.txt`                          |
| **Stage**              | 9 — 3-Model Ensemble                                          |
| **Val Macro-F1**       | **0.7833**                                                    |
| **Val Accuracy**       | ~80.5% (weight-optimised ensemble on 57,576 Bag val windows)  |
| **Smoothing applied**  | **No** — test rows are shuffled (temporal order undefined)    |

---

## Model Components

| Model           | Architecture              | Val Macro-F1 | Params  | Checkpoint path (local, not committed)                                                        |
|-----------------|---------------------------|--------------|---------|-----------------------------------------------------------------------------------------------|
| InceptionTime   | 6-layer inception, 32 nb  | 0.7265       | 492 K   | `outputs/execution-output/inception_posPool_nb32_d6_sfull_ep100_bs512/model.pt`              |
| IMUFormer       | Transformer d=128, 2 TF   | 0.7125       | ~500 K  | `outputs/execution-output/imuformer_d128tf2_posPool_sfull_strat40000_ep60/best_model.pt`     |
| MOMENT+stat XGB | MOMENT-1 + 354 stat feats | 0.7098       | XGB 300 | `outputs/execution-output/foundation_moment_posPool_mean_pool_normperwindow_xgb_hybrid_sfull_strat40000/model.joblib` |

---

## Ensemble Configuration

| Parameter                  | Value                         |
|----------------------------|-------------------------------|
| Temperature (InceptionTime)| Tuned on val (find_temperature)|
| Temperature (IMUFormer)    | Tuned on val (find_temperature)|
| Temperature (MOMENT-XGB)   | 1.000 (predict_proba, no logits) |
| Optimal weights (approx.)  | See stage9_ensemble.log       |
| Weight method              | Nelder–Mead (scipy.optimize)  |
| TTA n                      | 5 (jitter σ=0.02, scale 0.9–1.1) |
| TTA models                 | InceptionTime + IMUFormer     |
| Combination                | Weighted average of calibrated probs |

---

## Smoothing Decision

**Smoothing is NOT applied.**

The SHL 2026 test set rows are shuffled with no guaranteed temporal ordering.
Applying temporal smoothing (HMM, majority-vote window) on shuffled data would
corrupt predictions. `scripts/smooth_predictions.py` includes a shuffle guard
that warns if temporal order is violated — but for this submission, smoothing
is explicitly disabled.

---

## Generation Commands

```bash
# 1. Ensure HDF5 and model artefacts are in place (local only, not committed):
#    dataset/processed/shl2026.hdf5
#    outputs/execution-output/inception_posPool_nb32_d6_sfull_ep100_bs512/model.pt
#    outputs/execution-output/imuformer_d128tf2_posPool_sfull_strat40000_ep60/best_model.pt
#    outputs/execution-output/foundation_moment_posPool_mean_pool_normperwindow_xgb_hybrid_sfull_strat40000/model.joblib
#    dataset/processed/embeddings/test_Bag_moment_normperwindow_mean_pool.npz
#    dataset/processed/features/test_Bag.npz

# 2. Run Stage 9 ensemble with TTA n=5 and generate test submission:
CUDA_VISIBLE_DEVICES=2 python -u scripts/run_stage9_ensemble.py \
    --tta-n 5 --device cuda:0 --predict-test \
    --output outputs/execution-output/submissions/FeatureFlyers_ensemble_s9_tta5.txt \
    2>&1 | tee outputs/execution-output/stage9_ensemble_final.log
```

---

## Verification Command

```bash
python scripts/verify_submission.py \
    --input outputs/execution-output/submissions/FeatureFlyers_ensemble_s9_tta5.txt
```

Expected output:
```
  RESULT: PASS
  Lines     : 92,726  (expected 92,726) ✓
  Values/line: 500  ✓
  Label range: [1, 8]  ✓
```

---

## Files NOT Committed to Repository

The following artefacts are local-only (covered by `.gitignore`):

| Artefact                                     | Reason not committed          |
|----------------------------------------------|-------------------------------|
| `dataset/processed/shl2026.hdf5`             | Large binary, dataset terms   |
| `outputs/execution-output/inception_*/model.pt` | Large checkpoint (~50 MB)  |
| `outputs/execution-output/imuformer_*/best_model.pt` | Large checkpoint       |
| `outputs/execution-output/foundation_*/model.joblib` | XGB joblib bundle     |
| `dataset/processed/embeddings/*.npz`         | Large embedding cache         |
| `dataset/processed/features/*.npz`           | Large feature cache           |
| `outputs/execution-output/submissions/*.txt` | Submission file (large text)  |
| All `*.log` files                            | Run logs                      |

---

## Reproducibility Notes

- Random seed: `42` throughout (set in `configs/default.yaml`)
- Python: 3.11
- Key packages: see `requirements.txt`
- CUDA: GPU 2 (`CUDA_VISIBLE_DEVICES=2`) — inside script appears as `cuda:0`
- Full reproducibility commands: see `docs/reproducibility_commands.md`
