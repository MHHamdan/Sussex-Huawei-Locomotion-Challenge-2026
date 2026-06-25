# Final Submission Manifest — FeatureFlyers / SHL 2026

## Best Submission Candidate

| Field                  | Value                                                                       |
|------------------------|-----------------------------------------------------------------------------|
| **Team**               | FeatureFlyers                                                               |
| **Submission file**    | `FeatureFlyers_blend_s16_lgbm.txt`                                          |
| **Stage**              | 16 — 6-Model LightGBM Meta-Blend                                            |
| **Val Macro-F1**       | **0.9490** (holdout 20% of val = 11,516 windows)                            |
| **Val Accuracy**       | 94.1%                                                                       |
| **Smoothing applied**  | **No** — test rows are shuffled (cross-window boundary autocorrelation -0.158) |
| **File size**          | 88.4 MB — 92,726 lines × 500 comma-separated integers (labels 1–8)         |

---

## Model Components

| Model                       | Architecture                    | Val Macro-F1 | Checkpoint path (local, not committed)                                                                                     |
|-----------------------------|---------------------------------|:------------:|----------------------------------------------------------------------------------------------------------------------------|
| InceptionTime focal+bal     | 6-layer inception, 32 nb        | 0.7726       | `outputs/execution-output/inception_posPool_nb32_d6_sfull_focal_g2.0_balsampler_ep100_bs512/model.pt`                    |
| IMUFormer focal+bal         | Transformer d=128, 2 TF layers  | 0.7163       | `outputs/execution-output/imuformer_d128tf2_posPool_sfull_strat40000_focal_g2.0_balsampler_nocw_ep60/best_model.pt`       |
| SpectrogramCNN              | Log-mel CNN, nfft=64, hop=16    | 0.7590       | `outputs/execution-output/spectrogramcnn_posPool_nfft64_hop16_sfull_focal_g2.0_balsampler_ep80_bs256/model.pt`           |
| ResNet1D focal+bal          | Deep residual 1D, f=64          | 0.7740       | `outputs/execution-output/resnet1d_posPool_f64_sfull_focal_g2.0_balsampler_ep100_bs512/model.pt`                         |
| MVPF v2                     | 4-pos cross-fusion Transformer  | 0.7678       | `outputs/execution-output/mvpf_v2_4pos_fd256_bf64_h8tf3_sfull_focal_g2.0_balsampler_swa50_rotaug_ep80_bs256/model.pt`   |
| MOMENT-MLP                  | Frozen MOMENT-1-large + MLP     | 0.7681       | `outputs/execution-output/moment_large_mlp_ep60_lr1e-03_bs512/best_model.pt`                                             |
| **LightGBM meta-learner**   | 48-feature (6×8 softmax probs)  | **0.9490**   | `outputs/execution-output/meta_blend_s16_lgbm_6models/lgbm_meta.pkl`                                                    |

Why 6 models (not 7): MOMENT-XGB (`model.joblib`) artifacts from Stage 5/6 foundation runs were unavailable. The six models above cover all major inductive bias families (multi-scale temporal, spatial transformer, frequency-domain, residual, cross-position fusion, foundation).

---

## LightGBM Meta-Blend Configuration

| Parameter              | Value                                                      |
|------------------------|------------------------------------------------------------|
| Meta-learner           | LightGBM classifier, 500 trees, lr=0.05                   |
| Input features         | 6 models × 8 classes = 48 softmax probability columns      |
| Train split            | 80% of val windows (46,060 windows), stratified            |
| Holdout split          | 20% of val windows (11,516 windows), stratified            |
| TTA                    | n=3 (jitter σ=0.02, scale 0.9–1.1) applied to PyTorch models |
| Val Macro-F1 (holdout) | **0.9490**                                                 |
| Output dir             | `outputs/execution-output/meta_blend_s16_lgbm_6models/`   |

**Per-class F1 on holdout:**
| Still | Walking | Run  | Bike | Car  | Bus  | Train | Metro |
|:-----:|:-------:|:----:|:----:|:----:|:----:|:-----:|:-----:|
| 0.95  | 0.95    | 0.99 | 0.97 | 0.98 | 0.95 | 0.89  | 0.91  |

---

## Smoothing Decision

**Smoothing is NOT applied to test predictions.**

Stage 20 (HMM Viterbi, BiLSTM) improves **validation** F1 because validation sessions are temporally ordered continuous recordings. The test set rows are shuffled — verified by three independent lines of evidence:

1. `test/data` in the HDF5 is a single flat `(46363000, 9)` array with no timestamps, no session IDs, and no position split.
2. Cross-window boundary lag-1 autocorrelation on test = **-0.158** (expected ~0.997 for ordered 100 Hz IMU data).
3. `scripts/smooth_predictions.py` shuffle guard, `final_submission_manifest.md`, and `docs/results_summary.md` all document this constraint.

Applying Viterbi or BiLSTM hidden-state propagation to shuffled windows would corrupt predictions. Stage 20 val gains (HMM: 0.9298, BiLSTM: 0.9513) are **internal diagnostics only** and are not reflected in this submission.

---

## Generation Commands

```bash
# Prerequisites (local-only, not committed):
#   dataset/processed/shl2026.hdf5
#   outputs/execution-output/inception_.../model.pt
#   outputs/execution-output/imuformer_.../best_model.pt
#   outputs/execution-output/spectrogramcnn_.../model.pt
#   outputs/execution-output/resnet1d_.../model.pt
#   outputs/execution-output/mvpf_v2_.../model.pt
#   outputs/execution-output/moment_large_mlp_.../best_model.pt
#   outputs/execution-output/moment_embeddings/{train,validation,test}_embeddings.npz

# Run Stage 16 6-model blend and generate test submission:
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

---

## Verification Command

```bash
python scripts/verify_submission.py \
    --input outputs/execution-output/submissions/FeatureFlyers_blend_s16_lgbm.txt
```

Expected output:
```
  RESULT: PASS
  Lines     : 92,726  (expected 92,726) ✓
  Values/line: 500  ✓
  Label range: [1, 8]  ✓
```

Manual verification (confirmed): 92,726 lines, 500 integers per line, labels 1–8, file size 88.4 MB.

---

## Files NOT Committed to Repository

The following artefacts are local-only (covered by `.gitignore`):

| Artefact                                                      | Reason not committed          |
|---------------------------------------------------------------|-------------------------------|
| `dataset/processed/shl2026.hdf5`                             | Large binary, dataset terms   |
| `outputs/execution-output/inception_*/model.pt`              | Large checkpoint (~50 MB)     |
| `outputs/execution-output/imuformer_*/best_model.pt`         | Large checkpoint              |
| `outputs/execution-output/spectrogramcnn_*/model.pt`         | Large checkpoint              |
| `outputs/execution-output/resnet1d_*/model.pt`               | Large checkpoint              |
| `outputs/execution-output/mvpf_v2_*/model.pt`                | Large checkpoint              |
| `outputs/execution-output/moment_large_mlp_*/best_model.pt`  | Large checkpoint              |
| `outputs/execution-output/moment_embeddings/*.npz`           | Large embedding cache (3.4 GB)|
| `outputs/execution-output/meta_blend_s16_lgbm_6models/`      | LightGBM pickle + probs       |
| `outputs/execution-output/submissions/*.txt`                  | Submission files (large text) |
| All `*.log` files                                            | Run logs                      |

---

## Reproducibility Notes

- Random seed: `42` throughout (set in `configs/default.yaml`)
- Python: 3.11
- Key packages: see `requirements.txt`
- CUDA: RTX 2080 Ti × 4 GPUs; GPU 0 for inference chain
- MOMENT embeddings: extracted with `scripts/extract_moment_embeddings.py`, batch=64, fp16
- Full reproducibility commands: see `docs/reproducibility_commands.md`
