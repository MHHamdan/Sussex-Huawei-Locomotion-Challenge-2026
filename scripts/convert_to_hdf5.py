#!/usr/bin/env python3
"""
Convert raw SHL 2026 .txt sensor files to a single HDF5 file.

The raw .txt files are single-line space-separated values.
Each sensor (Acc_x, etc.) and the Label file must have the same number of tokens.

Output structure (dataset/processed/shl2026.hdf5):
    /train/
        /Bag/
            data    float32 (N, 9)   columns = [Acc_x,Acc_y,Acc_z,Gyr_x,Gyr_y,Gyr_z,Mag_x,Mag_y,Mag_z]
            labels  int8    (N,)
        /Hand/  ...
        /Hips/  ...
        /Torso/ ...
    /validation/
        ...  (same sub-structure as train)
    /test/
        data    float32 (N, 9)   no labels

Usage:
    python scripts/convert_to_hdf5.py [--raw-dir dataset/raw] [--out dataset/processed/shl2026.hdf5]
    python scripts/convert_to_hdf5.py --dry-run    # just print what would be done
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

SENSORS = ["Acc_x", "Acc_y", "Acc_z", "Gyr_x", "Gyr_y", "Gyr_z",
           "Mag_x", "Mag_y", "Mag_z"]

POSITIONS = ["Bag", "Hand", "Hips", "Torso"]

CHUNK_ROWS = 500_000   # rows per HDF5 chunk (tune to memory / access pattern)


def fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def read_txt_tokens(path: Path, dtype) -> np.ndarray:
    """Read a space-separated single-row .txt file as a 1-D array."""
    raw = path.read_bytes()
    return np.frombuffer(raw.replace(b"\n", b" "), dtype="S20").astype(str)


def load_sensor_file(path: Path, dtype=np.float32) -> np.ndarray:
    """Memory-efficient load of a single sensor txt file → 1-D array."""
    tokens = path.read_text().split()
    return np.array(tokens, dtype=dtype)


def load_label_file(path: Path) -> np.ndarray:
    tokens = path.read_text().split()
    return np.array(tokens, dtype=np.int8)


def write_split(h5_group, split_dir: Path, positions: list[str], dry_run: bool) -> None:
    """Write one split (train/validation/test) into an open HDF5 group."""
    import h5py  # imported late so --dry-run works without h5py

    if split_dir.name == "test":
        _write_position(h5_group, split_dir, has_labels=False, dry_run=dry_run)
    else:
        for pos in positions:
            pos_dir = split_dir / pos
            if not pos_dir.exists():
                print(f"  [SKIP] {split_dir.name}/{pos} — not found")
                continue
            pos_group = None if dry_run else h5_group.require_group(pos)
            print(f"  {split_dir.name}/{pos}", flush=True)
            _write_position(pos_group, pos_dir, has_labels=True, dry_run=dry_run)


def _write_position(group, pos_dir: Path, has_labels: bool, dry_run: bool) -> None:
    t0 = time.time()
    # Load all sensors
    arrays = []
    for sensor in SENSORS:
        fpath = pos_dir / f"{sensor}.txt"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing sensor file: {fpath}")
        arr = load_sensor_file(fpath, dtype=np.float32)
        arrays.append(arr)
        print(f"    {sensor}: {len(arr):,} samples", flush=True)

    # Validate all sensors have same length
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"Sensor length mismatch in {pos_dir}: {dict(zip(SENSORS, lengths))}")
    n = lengths[0]

    # Stack into (N, 9)
    data = np.stack(arrays, axis=1)  # (N, 9) float32

    if has_labels:
        label_file = pos_dir / "Label.txt"
        if not label_file.exists():
            raise FileNotFoundError(f"Missing Label.txt in {pos_dir}")
        labels = load_label_file(label_file)
        if len(labels) != n:
            raise ValueError(f"Label count {len(labels)} != sensor count {n} in {pos_dir}")

    elapsed = time.time() - t0
    mem_mb = data.nbytes / 1024 / 1024
    print(f"    → {n:,} samples × 9 sensors = {mem_mb:.0f} MB float32  ({elapsed:.1f}s)")

    if dry_run:
        if has_labels:
            from collections import Counter
            dist = dict(sorted(Counter(labels.tolist()).items()))
            print(f"    label distribution: {dist}")
        return

    import h5py
    ds_data = group.create_dataset(
        "data", data=data,
        chunks=(min(CHUNK_ROWS, n), 9),
        compression="lzf",
    )
    ds_data.attrs["columns"] = SENSORS
    if has_labels:
        group.create_dataset(
            "labels", data=labels,
            chunks=(min(CHUNK_ROWS, n),),
            compression="lzf",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path,
                        default=REPO_ROOT / "dataset" / "raw",
                        help="Path to dataset/raw/")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5",
                        help="Output HDF5 path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print structure without writing HDF5")
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    out_path: Path = args.out

    if not raw_dir.exists():
        print(f"ERROR: {raw_dir} does not exist. Run scripts/prepare_dataset.py first.")
        sys.exit(1)

    if not args.dry_run:
        try:
            import h5py
        except ImportError:
            print("ERROR: h5py is not installed. Run: pip install h5py")
            sys.exit(1)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            print(f"WARNING: {out_path} already exists — overwriting.")
            out_path.unlink()

    mode = "dry-run" if args.dry_run else f"writing to {out_path.relative_to(REPO_ROOT)}"
    print(f"Converting raw SHL 2026 data ({mode})")
    print(f"Raw dir: {raw_dir.relative_to(REPO_ROOT)}\n")

    t_total = time.time()

    splits = [
        ("train",      raw_dir / "train",      POSITIONS, True),
        ("validation", raw_dir / "validation",  POSITIONS, True),
        ("test",       raw_dir / "test",        [],        False),
    ]

    if args.dry_run:
        for split_name, split_dir, positions, has_positions in splits:
            if not split_dir.exists():
                print(f"[SKIP] {split_name} — not found at {split_dir}")
                continue
            print(f"\n=== {split_name} ===")
            if has_positions:
                for pos in positions:
                    pos_dir = split_dir / pos
                    if not pos_dir.exists():
                        print(f"  [SKIP] {pos}")
                        continue
                    print(f"  {pos}")
                    _write_position(None, pos_dir, has_labels=True, dry_run=True)
            else:
                _write_position(None, split_dir, has_labels=False, dry_run=True)
    else:
        import h5py
        with h5py.File(out_path, "w") as hf:
            hf.attrs["description"] = "SHL 2026 Challenge dataset"
            hf.attrs["sensors"] = SENSORS
            hf.attrs["label_map"] = str({1:"Still",2:"Walking",3:"Run",4:"Bike",
                                          5:"Car",6:"Bus",7:"Train",8:"Metro"})
            for split_name, split_dir, positions, has_positions in splits:
                if not split_dir.exists():
                    print(f"[SKIP] {split_name} — not found")
                    continue
                print(f"\n=== {split_name} ===")
                grp = hf.require_group(split_name)
                write_split(grp, split_dir, positions, dry_run=False)

        size = out_path.stat().st_size
        print(f"\nHDF5 written: {out_path.relative_to(REPO_ROOT)} ({fmt_size(size)})")

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed:.1f}s")

    # Quick validation
    if not args.dry_run:
        print("\nValidating HDF5 …")
        import h5py
        with h5py.File(out_path, "r") as hf:
            for split_name, split_dir, positions, has_positions in splits:
                if split_name not in hf:
                    continue
                if has_positions:
                    for pos in positions:
                        if pos not in hf[split_name]:
                            continue
                        grp = hf[split_name][pos]
                        n = grp["data"].shape[0]
                        ncols = grp["data"].shape[1]
                        lab_ok = "labels" in grp and grp["labels"].shape[0] == n
                        print(f"  {split_name}/{pos}: {n:,} samples × {ncols} sensors, "
                              f"labels={'OK' if lab_ok else 'MISSING'}")
                else:
                    grp = hf[split_name]
                    if "data" in grp:
                        n = grp["data"].shape[0]
                        print(f"  {split_name}: {n:,} samples × {grp['data'].shape[1]} sensors (no labels)")
        print("Validation done.")


if __name__ == "__main__":
    main()
