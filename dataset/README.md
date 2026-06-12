# Dataset Directory

This directory holds the SHL 2026 challenge data. **Nothing here is committed to Git.**

## Structure

```
dataset/
├── archives/      # Original .zip files (source of truth, kept locally)
├── raw/           # Extracted .txt sensor files
│   ├── train/
│   │   ├── Bag/
│   │   ├── Hand/
│   │   ├── Hips/
│   │   └── Torso/
│   ├── validation/
│   │   ├── Bag/
│   │   ├── Hand/
│   │   ├── Hips/
│   │   └── Torso/
│   └── test/      # No labels — used for final submission
└── processed/
    └── shl2026.hdf5
```

## File Format

Each sensor file (e.g. `Acc_x.txt`) is a single line of space-separated float values —
one value per sample at the sensor sampling rate.

`Label.txt` is a single line of space-separated integers (1–8).

## Generating from source

```bash
# From repo root
python scripts/prepare_dataset.py   # extract zips → dataset/raw/
python scripts/convert_to_hdf5.py   # convert → dataset/processed/shl2026.hdf5
```
