#!/usr/bin/env python3
"""
Analyse label distributions in train and validation splits.

Reads from dataset/processed/shl2026.hdf5.
Saves a text summary to outputs/eda/label_analysis.txt.

Usage:
    python scripts/analyze_labels.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
HDF5_PATH = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
OUT_DIR = REPO_ROOT / "outputs" / "eda"

LABEL_MAP = {1: "Still", 2: "Walking", 3: "Run", 4: "Bike",
             5: "Car", 6: "Bus", 7: "Train", 8: "Metro"}

POSITIONS = ["Bag", "Hand", "Hips", "Torso"]


def fmt_pct(n: int, total: int) -> str:
    return f"{n/total*100:.1f}%"


def analyze_split(hf, split: str, positions: list[str], lines: list[str]) -> None:
    lines.append(f"\n{'='*60}")
    lines.append(f"  Split: {split}")
    lines.append(f"{'='*60}")

    agg: Counter = Counter()

    for pos in positions:
        path = f"{split}/{pos}"
        if path not in hf:
            lines.append(f"  [SKIP] {path} — not found")
            continue

        labels = hf[path]["labels"][:]   # int8 array
        total = len(labels)
        counts = Counter(labels.tolist())

        lines.append(f"\n  Position: {pos}  ({total:,} samples)")
        lines.append(f"  {'-'*50}")
        for lbl in sorted(LABEL_MAP):
            n = counts.get(lbl, 0)
            bar = "█" * int(n / total * 40)
            lines.append(f"  {lbl} {LABEL_MAP[lbl]:<10} {n:>12,}  {fmt_pct(n,total):>6}  {bar}")

        agg += counts

    if len(positions) > 1:
        total = sum(agg.values())
        lines.append(f"\n  All positions aggregated ({total:,} samples)")
        lines.append(f"  {'-'*50}")
        for lbl in sorted(LABEL_MAP):
            n = agg.get(lbl, 0)
            bar = "█" * int(n / total * 40)
            lines.append(f"  {lbl} {LABEL_MAP[lbl]:<10} {n:>12,}  {fmt_pct(n,total):>6}  {bar}")


def main() -> None:
    try:
        import h5py
    except ImportError:
        print("ERROR: h5py not installed — run: pip install h5py")
        sys.exit(1)

    if not HDF5_PATH.exists():
        print(f"ERROR: {HDF5_PATH} not found. Run scripts/convert_to_hdf5.py first.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("SHL 2026 — Label Distribution Analysis")
    lines.append(f"HDF5: {HDF5_PATH}")
    lines.append(f"\nLabel map: {LABEL_MAP}")

    print("Loading label distributions …", flush=True)
    with h5py.File(HDF5_PATH, "r") as hf:
        analyze_split(hf, "train",      POSITIONS, lines)
        analyze_split(hf, "validation", POSITIONS, lines)

    # Class imbalance summary
    lines.append(f"\n{'='*60}")
    lines.append("  Notes")
    lines.append(f"{'='*60}")
    lines.append("  Run class is the minority (~4% of train).")
    lines.append("  Bus is also relatively rare (~14% of train).")
    lines.append("  Consider class-weighted loss or oversampling.")
    lines.append("")

    text = "\n".join(lines)
    print(text)

    out_path = OUT_DIR / "label_analysis.txt"
    out_path.write_text(text)
    print(f"\nSaved → {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    import os
    os.chdir(REPO_ROOT)
    main()
