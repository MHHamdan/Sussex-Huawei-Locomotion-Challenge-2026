#!/usr/bin/env python3
"""
Inspect the dataset directory: print tree, file counts, sizes, extensions,
and label distributions where available. Does NOT modify any files.
"""

import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"

LABEL_MAP = {
    1: "Still", 2: "Walking", 3: "Run", 4: "Bike",
    5: "Car", 6: "Bus", 7: "Train", 8: "Metro",
}

SENSORS = ["Acc_x", "Acc_y", "Acc_z", "Gyr_x", "Gyr_y", "Gyr_z",
           "Mag_x", "Mag_y", "Mag_z"]

POSITIONS = ["Bag", "Hand", "Hips", "Torso"]


def fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def print_tree(root: Path, prefix: str = "", max_depth: int = 4, depth: int = 0) -> None:
    if depth > max_depth:
        return
    entries = sorted(root.iterdir())
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        size_str = f"  [{fmt_size(entry.stat().st_size)}]" if entry.is_file() else ""
        print(f"{prefix}{connector}{entry.name}{size_str}")
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(entry, prefix + extension, max_depth, depth + 1)


def count_tokens(path: Path) -> int:
    """Count whitespace-separated tokens in a text file."""
    total = 0
    with open(path) as f:
        for line in f:
            total += len(line.split())
    return total


def label_distribution(path: Path) -> dict:
    counts: Counter = Counter()
    with open(path) as f:
        for line in f:
            for tok in line.split():
                counts[int(tok)] += 1
    return dict(sorted(counts.items()))


def inspect_zips(dataset_dir: Path) -> None:
    zips = sorted(dataset_dir.glob("*.zip"))
    if not zips:
        return
    print(f"\n{'='*60}")
    print(f"ZIP files in {dataset_dir.relative_to(REPO_ROOT)}")
    print(f"{'='*60}")
    for z in zips:
        size = fmt_size(z.stat().st_size)
        try:
            with zipfile.ZipFile(z) as zf:
                entries = [n for n in zf.namelist() if not n.endswith("/")]
                print(f"  {z.name} ({size}) — {len(entries)} files inside")
        except zipfile.BadZipFile:
            print(f"  {z.name} ({size}) — CORRUPT or not a zip")


def inspect_raw(raw_dir: Path) -> None:
    if not raw_dir.exists():
        print(f"\n[raw/] does not exist yet — run prepare_dataset.py first.")
        return

    print(f"\n{'='*60}")
    print(f"Raw data structure ({raw_dir.relative_to(REPO_ROOT)})")
    print(f"{'='*60}")
    print_tree(raw_dir, max_depth=2)

    for split in ("train", "validation", "test"):
        split_dir = raw_dir / split
        if not split_dir.exists():
            continue
        print(f"\n--- {split} ---")

        # Determine positions present
        if split == "test":
            positions_dirs = [split_dir]
            pos_names = ["(flat)"]
        else:
            positions_dirs = [split_dir / p for p in POSITIONS if (split_dir / p).exists()]
            pos_names = [p.name for p in positions_dirs]

        for pos_dir, pos_name in zip(positions_dirs, pos_names):
            label_file = pos_dir / "Label.txt"
            n_samples = count_tokens(label_file) if label_file.exists() else "N/A"

            sensor_files = [pos_dir / f"{s}.txt" for s in SENSORS]
            all_present = all(f.exists() for f in sensor_files)
            sensor_status = "all 9 sensors present" if all_present else (
                f"sensors present: {[s for s in SENSORS if (pos_dir/f'{s}.txt').exists()]}"
            )

            print(f"  {pos_name}: {n_samples} samples, {sensor_status}")

            if label_file.exists():
                dist = label_distribution(label_file)
                lines = [f"{LABEL_MAP.get(k, k)}:{v:,}" for k, v in dist.items()]
                print(f"    labels: {' | '.join(lines)}")


def inspect_processed(processed_dir: Path) -> None:
    if not processed_dir.exists():
        return
    hdf5_files = list(processed_dir.glob("*.hdf5")) + list(processed_dir.glob("*.h5"))
    if not hdf5_files:
        print(f"\n[processed/] exists but contains no HDF5 files yet.")
        return
    try:
        import h5py
    except ImportError:
        print("\n[processed/] HDF5 files found but h5py not installed — skipping HDF5 inspection.")
        return

    print(f"\n{'='*60}")
    print("Processed HDF5 files")
    print(f"{'='*60}")
    for hf in hdf5_files:
        size = fmt_size(hf.stat().st_size)
        print(f"\n  {hf.name} ({size})")
        with h5py.File(hf, "r") as f:
            def visitor(name, obj):
                indent = "  " * (name.count("/") + 2)
                if hasattr(obj, "shape"):
                    print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
                else:
                    print(f"{indent}{name}/")
            f.visititems(visitor)


def main() -> None:
    print(f"Dataset root: {DATASET_DIR}")
    print(f"\n{'='*60}")
    print("Top-level structure")
    print(f"{'='*60}")
    print_tree(DATASET_DIR, max_depth=1)

    inspect_zips(DATASET_DIR)
    inspect_raw(DATASET_DIR / "raw")
    inspect_processed(DATASET_DIR / "processed")

    print()


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
