#!/usr/bin/env python3
"""
Scan outputs/execution-output/ for foundation experiment metrics.json files
and print a sorted comparison table.

Usage
-----
  python scripts/summarize_foundation_results.py
  python scripts/summarize_foundation_results.py --output-dir outputs/foundation
  python scripts/summarize_foundation_results.py --sort f1       # sort by Macro-F1
  python scripts/summarize_foundation_results.py --sort acc      # sort by accuracy
  python scripts/summarize_foundation_results.py --csv           # emit CSV
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTD = REPO_ROOT / "outputs" / "execution-output"
BASELINE_F1  = 0.6389


def load_metrics(output_dir: Path) -> list[dict]:
    """Collect all metrics.json under subdirs that start with 'foundation_'."""
    records = []
    for p in sorted(output_dir.glob("foundation_*/metrics.json")):
        try:
            m = json.loads(p.read_text())
            m["_run_name"] = p.parent.name
            records.append(m)
        except Exception as e:
            print(f"  [WARN] could not load {p}: {e}", file=sys.stderr)
    return records


def fmt(v, fmt_str: str = ".4f") -> str:
    if v is None:
        return "  n/a "
    return format(v, fmt_str)


def print_table(records: list[dict], sort_key: str = "f1") -> None:
    if not records:
        print("No foundation experiment results found.")
        return

    if sort_key == "f1":
        records = sorted(records, key=lambda r: r.get("best_val_macro_f1", 0.0),
                         reverse=True)
    else:
        records = sorted(records, key=lambda r: r.get("best_val_accuracy", 0.0),
                         reverse=True)

    col_w = 38
    print(f"\n{'=' * 115}")
    print(f"Foundation Model Ablation Results — baseline XGB pool F1={BASELINE_F1:.4f}")
    print(f"{'=' * 115}")
    hdr = (
        f"{'Run':<{col_w}}  "
        f"{'Encoder':<8}  {'Strategy':<16}  {'Norm':<14}  "
        f"{'Channels':>8}  {'Head':<12}  "
        f"{'Macro-F1':>8}  {'Acc':>6}  {'ΔvsBase':>7}  "
        f"{'Hybrid':>6}  {'Time(s)':>7}"
    )
    print(hdr)
    print("-" * 115)

    for r in records:
        name    = r.get("_run_name", "?")[:col_w]
        enc     = r.get("encoder", "?")[:8]
        strat   = r.get("embed_strategy", "?")[:16]
        norm    = r.get("norm", "?")[:14]
        mag     = "✓" if r.get("include_magnitude") else ""
        dlt     = "δ" if r.get("include_delta") else ""
        n_ch    = r.get("n_channels", "?")
        ch_str  = f"{n_ch}{mag}{dlt}"
        head    = r.get("head", "?")[:12]
        f1      = r.get("best_val_macro_f1", None)
        acc     = r.get("best_val_accuracy", None)
        delta   = r.get("delta_vs_baseline", None)
        hybrid  = "yes" if r.get("hybrid_stat_features") else "-"
        t_s     = r.get("total_time_s", None)
        t_str   = f"{t_s:.0f}" if t_s is not None else "?"

        delta_str = (f"+{delta:.4f}" if delta and delta >= 0
                     else f"{delta:.4f}" if delta is not None else "?")
        f1_flag = " ←BEST" if f1 and f1 == max(
            rr.get("best_val_macro_f1", 0) for rr in records) else ""

        print(
            f"{name:<{col_w}}  "
            f"{enc:<8}  {strat:<16}  {norm:<14}  "
            f"{ch_str:>8}  {head:<12}  "
            f"{fmt(f1):>8}  {fmt(acc, '.3f'):>6}  {delta_str:>7}  "
            f"{hybrid:>6}  {t_str:>7}"
            + f1_flag
        )

    print("=" * 115)
    best_f1  = max((r.get("best_val_macro_f1", 0.0) for r in records), default=0.0)
    best_acc = max((r.get("best_val_accuracy", 0.0) for r in records), default=0.0)
    print(f"\nBest Macro-F1 : {best_f1:.4f}  "
          f"(delta vs baseline: {best_f1 - BASELINE_F1:+.4f})")
    print(f"Best Accuracy : {best_acc:.4f}")
    print(f"Total runs    : {len(records)}\n")


def print_csv(records: list[dict]) -> None:
    cols = [
        "run_name", "encoder", "embed_strategy", "norm",
        "include_magnitude", "include_delta", "n_channels",
        "head", "hybrid_stat_features",
        "best_val_macro_f1", "best_val_accuracy", "delta_vs_baseline",
        "total_time_s", "epochs_trained", "early_stopped",
        "sample_limit", "position",
    ]
    print(",".join(cols))
    for r in records:
        row = [str(r.get(c if c != "run_name" else "_run_name", "")).replace(",", ";")
               for c in ["_run_name" if c == "run_name" else c for c in cols]]
        # fix: cols uses run_name but dict has _run_name
        vals = []
        for c in cols:
            k = "_run_name" if c == "run_name" else c
            vals.append(str(r.get(k, "")).replace(",", ";"))
        print(",".join(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTD)
    parser.add_argument("--sort", choices=["f1", "acc"], default="f1")
    parser.add_argument("--csv", action="store_true", help="Emit CSV instead of table")
    args = parser.parse_args()

    if not args.output_dir.exists():
        print(f"Output dir not found: {args.output_dir}")
        sys.exit(1)

    records = load_metrics(args.output_dir)
    print(f"Found {len(records)} foundation run(s) in {args.output_dir}", file=sys.stderr)

    if args.csv:
        print_csv(records)
    else:
        print_table(records, sort_key=args.sort)


if __name__ == "__main__":
    main()
