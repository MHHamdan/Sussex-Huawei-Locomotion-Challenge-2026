#!/usr/bin/env python3
"""
Validate dataset/processed/shl2026.hdf5.

Checks:
- File exists and is readable
- Expected groups: train/{Bag,Hand,Hips,Torso}, validation/{Bag,Hand,Hips,Torso}, test
- Shapes, dtypes, compression
- Labels present in train/validation, absent in test
- Sample counts match expected values
- No NaN or Inf values (sampled check)
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
HDF5_PATH = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"

SENSORS = ["Acc_x", "Acc_y", "Acc_z", "Gyr_x", "Gyr_y", "Gyr_z",
           "Mag_x", "Mag_y", "Mag_z"]

LABEL_MAP = {1: "Still", 2: "Walking", 3: "Run", 4: "Bike",
             5: "Car", 6: "Bus", 7: "Train", 8: "Metro"}

EXPECTED = {
    "train":      {"Bag": 98_036_000, "Hand": 98_036_000,
                   "Hips": 98_036_000, "Torso": 98_036_000},
    "validation": {"Bag": 14_394_500, "Hand": 14_394_500,
                   "Hips": 14_394_500, "Torso": 14_394_500},
    "test":       {"_flat": 92_726},   # windows; shape (92726, 500, 9)
}

N_TEST_WINDOWS = 92_726
WIN_SIZE = 500

POSITIONS = ["Bag", "Hand", "Hips", "Torso"]


def fmt_size(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def check(cond: bool, msg: str) -> bool:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    return cond


def main() -> None:
    try:
        import h5py
    except ImportError:
        print("ERROR: h5py not installed — run: pip install h5py")
        sys.exit(1)

    print(f"HDF5 path: {HDF5_PATH}")

    if not check(HDF5_PATH.exists(), f"File exists ({HDF5_PATH.name})"):
        sys.exit(1)

    size = HDF5_PATH.stat().st_size
    print(f"  File size: {fmt_size(size)}")

    failures = 0

    with h5py.File(HDF5_PATH, "r") as hf:
        # Top-level groups
        top = set(hf.keys())
        for grp in ("train", "validation", "test"):
            ok = check(grp in top, f"Top-level group '{grp}' present")
            if not ok:
                failures += 1

        print()
        print("=== train ===")
        for pos in POSITIONS:
            path = f"train/{pos}"
            if not check(path in hf, f"{path} exists"):
                failures += 1
                continue
            g = hf[path]
            n_exp = EXPECTED["train"][pos]

            ok = check("data" in g, f"{path}/data present")
            failures += 0 if ok else 1
            ok2 = check("labels" in g, f"{path}/labels present")
            failures += 0 if ok2 else 1

            if "data" in g:
                ds = g["data"]
                shape_ok = ds.shape == (n_exp, 9)
                check(shape_ok,
                      f"{path}/data shape {ds.shape} == ({n_exp}, 9)")
                failures += 0 if shape_ok else 1
                check(ds.dtype == np.float32,
                      f"{path}/data dtype {ds.dtype} == float32")
                # Compression
                check(ds.compression is not None,
                      f"{path}/data is compressed ({ds.compression})")
                # Sampled NaN/Inf check
                sample = ds[::200_000, :]
                finite = np.isfinite(sample).all()
                check(finite, f"{path}/data sample has no NaN/Inf")
                failures += 0 if finite else 1

            if "labels" in g:
                ls = g["labels"]
                check(ls.shape == (n_exp,),
                      f"{path}/labels shape {ls.shape} == ({n_exp},)")
                uniq = np.unique(ls[:100_000])
                valid_labels = all(int(v) in LABEL_MAP for v in uniq)
                check(valid_labels,
                      f"{path}/labels values in {{1..8}}: found {sorted(uniq.tolist())}")
                failures += 0 if valid_labels else 1

        print()
        print("=== validation ===")
        for pos in POSITIONS:
            path = f"validation/{pos}"
            if not check(path in hf, f"{path} exists"):
                failures += 1
                continue
            g = hf[path]
            n_exp = EXPECTED["validation"][pos]
            if "data" in g:
                shape_ok = g["data"].shape == (n_exp, 9)
                check(shape_ok,
                      f"{path}/data shape {g['data'].shape} == ({n_exp}, 9)")
                failures += 0 if shape_ok else 1
            if "labels" in g:
                check(g["labels"].shape == (n_exp,),
                      f"{path}/labels shape OK")

        print()
        print("=== test ===")
        if "test" in hf:
            g = hf["test"]
            check("data" in g, "test/data present")
            check("labels" not in g, "test has no labels (correct)")
            if "data" in g:
                ds = g["data"]
                expected_shape = (N_TEST_WINDOWS, WIN_SIZE, 9)
                shape_ok = ds.shape == expected_shape
                check(shape_ok,
                      f"test/data shape {ds.shape} == {expected_shape}")
                failures += 0 if shape_ok else 1
                # Sample NaN/Inf check: read a handful of windows
                sample = ds[::5000, :, :]
                finite = np.isfinite(sample).all()
                check(finite, "test/data sample has no NaN/Inf")
                failures += 0 if finite else 1

    print()
    if failures == 0:
        print("All checks PASSED.")
    else:
        print(f"{failures} check(s) FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
