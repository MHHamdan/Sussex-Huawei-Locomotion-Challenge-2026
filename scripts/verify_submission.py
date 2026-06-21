#!/usr/bin/env python3
"""
Verify a SHL 2026 submission file.

Checks:
  - Line count == 92,726
  - Each line has exactly 500 comma-separated values
  - All values are integers in range [1, 8]
  - Prints class distribution and file size
  - Prints PASS / FAIL summary

Does NOT modify the file.

Usage:
    python scripts/verify_submission.py --input <path_to_submission.txt>
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

N_WINDOWS = 92_726
WIN_SIZE   = 500
LABEL_MIN  = 1
LABEL_MAX  = 8
LABEL_NAMES = {
    1: "Still",
    2: "Walking",
    3: "Run",
    4: "Bike",
    5: "Car",
    6: "Bus",
    7: "Train",
    8: "Metro",
}


def verify(path: Path) -> bool:
    print(f"\nVerifying: {path}")
    print(f"  File size : {path.stat().st_size / 1_048_576:.2f} MB")

    errors: list[str] = []
    label_counter: Counter = Counter()
    lines_ok = 0

    with open(path, "r") as fh:
        lines = fh.readlines()

    n_lines = len(lines)
    if n_lines != N_WINDOWS:
        errors.append(f"Line count {n_lines} != {N_WINDOWS}")

    for i, line in enumerate(lines):
        parts = line.rstrip("\n").split(",")

        if len(parts) != WIN_SIZE:
            errors.append(f"Line {i+1}: expected {WIN_SIZE} values, got {len(parts)}")
            if len(errors) >= 5:
                errors.append("... (further line errors suppressed)")
                break
            continue

        bad_vals: list[str] = []
        for v in parts:
            try:
                iv = int(v)
            except ValueError:
                bad_vals.append(repr(v))
                continue
            if iv < LABEL_MIN or iv > LABEL_MAX:
                bad_vals.append(str(iv))
            else:
                label_counter[iv] += 1

        if bad_vals:
            errors.append(
                f"Line {i+1}: {len(bad_vals)} out-of-range/non-integer value(s): "
                f"{bad_vals[:5]}"
            )
            if len(errors) >= 5:
                errors.append("... (further value errors suppressed)")
                break

        lines_ok += 1

    # Class distribution
    total_preds = sum(label_counter.values())
    if total_preds > 0:
        print(f"\n  Class distribution ({total_preds:,} valid predictions from {lines_ok:,} lines):")
        for cls in sorted(LABEL_NAMES):
            count = label_counter.get(cls, 0)
            pct   = 100.0 * count / total_preds if total_preds else 0.0
            bar   = "#" * int(pct / 2)
            print(f"    [{cls}] {LABEL_NAMES[cls]:<10} : {count:>10,}  ({pct:5.1f}%)  {bar}")

    print()
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        print(f"\n  RESULT: FAIL  ({len(errors)} error(s))")
        return False
    else:
        print(f"  Lines     : {n_lines:,}  (expected {N_WINDOWS:,}) ✓")
        print(f"  Values/line: {WIN_SIZE}  ✓")
        print(f"  Label range: [{LABEL_MIN}, {LABEL_MAX}]  ✓")
        print(f"\n  RESULT: PASS")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to submission .txt file")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: file not found — {args.input}", file=sys.stderr)
        sys.exit(1)

    ok = verify(args.input)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
