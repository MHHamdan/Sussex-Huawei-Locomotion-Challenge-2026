#!/usr/bin/env python3
"""
Extract SHL 2026 zip files into dataset/raw/ and move archives to dataset/archives/.

Usage:
    python scripts/prepare_dataset.py [--overwrite] [--keep-zips]

Options:
    --overwrite     Re-extract even if destination files already exist.
    --keep-zips     Do not move zip files to archives/ after extraction.
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"
ARCHIVES_DIR = DATASET_DIR / "archives"

# Mapping: zip stem → expected extraction root inside the zip
ZIP_SPECS = {
    "SHL-2026-Train_Bag":         ("train",      None),
    "SHL-2026-Train_Hand":        ("train",      None),
    "SHL-2026-Train_Hips":        ("train",      None),
    "SHL-2026-Train_Torso":       ("train",      None),
    "SHL-2026-Validation":        ("validation", None),
    "SHL-2026-Test":              ("test",       None),
    "SHL-2026-example_submission": (".",         "example_submission"),
}


def extract_zip(zip_path: Path, dest_dir: Path, overwrite: bool) -> bool:
    """Extract zip_path into dest_dir. Returns True if anything was extracted."""
    extracted_any = False
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.filename.endswith("/")]
        for member in members:
            target = dest_dir / member.filename
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_any = True
    return extracted_any


def verify_position(split_dir: Path, position: str) -> bool:
    pos_dir = split_dir / position
    if not pos_dir.exists():
        return False
    sensors = ["Acc_x", "Acc_y", "Acc_z", "Gyr_x", "Gyr_y", "Gyr_z",
               "Mag_x", "Mag_y", "Mag_z", "Label"]
    missing = [s for s in sensors if not (pos_dir / f"{s}.txt").exists()]
    if missing:
        print(f"  [WARN] {position}: missing {missing}")
        return False
    return True


def verify_test(test_dir: Path) -> bool:
    sensors = ["Acc_x", "Acc_y", "Acc_z", "Gyr_x", "Gyr_y", "Gyr_z",
               "Mag_x", "Mag_y", "Mag_z"]
    missing = [s for s in sensors if not (test_dir / f"{s}.txt").exists()]
    if missing:
        print(f"  [WARN] test: missing {missing}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if files already exist")
    parser.add_argument("--keep-zips", action="store_true",
                        help="Do not move zip files to archives/ after extraction")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(DATASET_DIR.glob("*.zip"))
    if not zip_files:
        print(f"No .zip files found in {DATASET_DIR}. Nothing to do.")
        sys.exit(0)

    print(f"Found {len(zip_files)} zip file(s) in {DATASET_DIR.relative_to(REPO_ROOT)}")

    all_ok = True
    for zip_path in zip_files:
        stem = zip_path.stem
        if stem not in ZIP_SPECS:
            print(f"  [SKIP] Unrecognised zip: {zip_path.name}")
            continue

        internal_root, alt_dest = ZIP_SPECS[stem]
        if alt_dest:
            dest = RAW_DIR / alt_dest
        else:
            dest = RAW_DIR

        print(f"\n  Extracting {zip_path.name} → {dest.relative_to(REPO_ROOT)} …", flush=True)
        extracted = extract_zip(zip_path, dest, overwrite=args.overwrite)

        if not extracted and not args.overwrite:
            print(f"    Already extracted (use --overwrite to re-extract).")
        else:
            print(f"    Done.")

        # Verify extracted files
        if internal_root in ("train", "validation"):
            split_dir = dest / internal_root
            for pos in ("Bag", "Hand", "Hips", "Torso"):
                if (split_dir / pos).exists():
                    ok = verify_position(split_dir, pos)
                    status = "OK" if ok else "INCOMPLETE"
                    print(f"    {internal_root}/{pos}: {status}")
        elif internal_root == "test":
            test_dir = dest / "test"
            ok = verify_test(test_dir)
            print(f"    test: {'OK' if ok else 'INCOMPLETE'}")
            all_ok = all_ok and ok

        if not args.keep_zips:
            target = ARCHIVES_DIR / zip_path.name
            if not target.exists():
                shutil.move(str(zip_path), str(target))
                print(f"    Moved to {target.relative_to(REPO_ROOT)}")
            else:
                print(f"    Archive already in {ARCHIVES_DIR.relative_to(REPO_ROOT)}, original left in place.")

    print("\n" + ("=" * 50))
    if all_ok:
        print("Extraction complete. Run scripts/inspect_dataset.py to verify.")
    else:
        print("Extraction finished with warnings — check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
