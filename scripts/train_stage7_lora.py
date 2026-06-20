#!/usr/bin/env python3
"""
Stage 7 — LoRA fine-tuning of MOMENT-1-large for SHL 2026 (paper experiment).

NOTE: This violates the competition's "frozen foundation model" rule and CANNOT
be submitted. It exists to establish an upper-bound for the research paper.

Protocol:
  - Load MOMENT-1-large in classification mode (task_name='classification')
  - Freeze all parameters
  - Add LoRA adapters (rank=8) to q,v in the top N transformer blocks (default: last 4)
  - Unfreeze the classification head
  - Train with AdamW: LoRA params at lr, head at 10x lr
  - Pool fusion: Bag+Hand+Hips+Torso stacked as independent samples
  - Validate on Bag only (57 576 windows)

Usage
-----
# Smoke test
python scripts/train_stage7_lora.py --sample-limit 2000 --epochs 3 --device cuda:1

# Full run (Stage 7)
python scripts/train_stage7_lora.py --epochs 30 --patience 7 --device cuda:1 \\
    2>&1 | tee outputs/execution-output/stage7_lora_full.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT    = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HDF5_PATH    = REPO_ROOT / "dataset" / "processed" / "shl2026.hdf5"
DEFAULT_OUTD = REPO_ROOT / "outputs" / "execution-output"
POSITIONS    = ["Bag", "Hand", "Hips", "Torso"]
N_CLASSES    = 8
LABEL_MAP    = {0:"Still",1:"Walking",2:"Run",3:"Bike",4:"Car",5:"Bus",6:"Train",7:"Metro"}
MOMENT_ID    = "AutonLab/MOMENT-1-large"


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    import torch
    model.train()
    total_loss = total_correct = total_n = 0
    for x, y in loader:
        x, y = x.to(device, dtype=torch.float32), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                out  = model(x_enc=x, input_mask=torch.ones(x.shape[0], x.shape[-1],
                             device=device, dtype=torch.bool))
                logits = out.logits if hasattr(out, "logits") else out.prediction_outputs
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(x_enc=x, input_mask=torch.ones(x.shape[0], x.shape[-1],
                         device=device, dtype=torch.bool))
            logits = out.logits if hasattr(out, "logits") else out.prediction_outputs
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        total_loss    += loss.item() * len(y)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n       += len(y)
    return total_loss / total_n, total_correct / total_n


def eval_epoch(model, loader, device):
    import torch
    from sklearn.metrics import f1_score
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            out = model(x_enc=x, input_mask=torch.ones(x.shape[0], x.shape[-1],
                        device=device, dtype=torch.bool))
            logits = out.logits if hasattr(out, "logits") else out.prediction_outputs
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(y.numpy())
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = float((preds == labels).mean())
    f1  = float(f1_score(labels, preds, average="macro", zero_division=0))
    return acc, f1, preds, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hdf5-path",      type=Path, default=HDF5_PATH)
    parser.add_argument("--sample-limit",   type=int,  default=None,
                        help="Stratified window limit per position per split (None=full)")
    parser.add_argument("--lora-rank",      type=int,  default=8)
    parser.add_argument("--lora-alpha",     type=int,  default=16)
    parser.add_argument("--lora-dropout",   type=float, default=0.05)
    parser.add_argument("--lora-blocks",    type=int,  default=4,
                        help="Number of top MOMENT transformer blocks to apply LoRA to")
    parser.add_argument("--epochs",         type=int,  default=30)
    parser.add_argument("--batch-size",     type=int,  default=64,
                        help="Smaller than InceptionTime — MOMENT is 341M params")
    parser.add_argument("--lr",             type=float, default=1e-4,
                        help="LR for LoRA adapter params")
    parser.add_argument("--head-lr",        type=float, default=1e-3,
                        help="LR for classification head (10x LoRA LR by default)")
    parser.add_argument("--patience",       type=int,  default=7)
    parser.add_argument("--seed",           type=int,  default=42)
    parser.add_argument("--device",         default="cuda:1")
    parser.add_argument("--num-workers",    type=int,  default=4)
    parser.add_argument("--output-dir",     type=Path, default=DEFAULT_OUTD)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, ConcatDataset
    from peft import LoraConfig, get_peft_model
    from sklearn.metrics import classification_report, confusion_matrix
    from momentfm import MOMENTPipeline

    from featureflyers_shl.data.dataset import SHLWindowDataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device  = torch.device(args.device)
    use_amp = device.type == "cuda"
    lim_str = str(args.sample_limit) if args.sample_limit else "full"

    run_name = (f"stage7_lora_r{args.lora_rank}_blocks{args.lora_blocks}"
                f"_posPool_s{lim_str}_ep{args.epochs}")
    run_dir  = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStage 7 — LoRA fine-tuning of MOMENT-1-large  [PAPER EXPERIMENT — NOT SUBMITTABLE]")
    print(f"  device      : {device}  (AMP={'on' if use_amp else 'off'})")
    print(f"  LoRA rank   : {args.lora_rank}  alpha={args.lora_alpha}  blocks={args.lora_blocks}")
    print(f"  positions   : {POSITIONS} (pool)")
    print(f"  epochs      : {args.epochs}  patience={args.patience}")
    print(f"  output      : {run_dir.relative_to(REPO_ROOT)}\n")

    # ---- datasets ----
    print("Loading train datasets (pool) …")
    t0 = time.time()
    train_datasets = [
        SHLWindowDataset(args.hdf5_path, split="train",
                         position=pos, sample_limit=args.sample_limit, seed=args.seed)
        for pos in POSITIONS
    ]
    ds_train = ConcatDataset(train_datasets)
    ds_val   = SHLWindowDataset(args.hdf5_path, split="validation",
                                position="Bag", sample_limit=None, seed=args.seed + 1)
    n_train  = sum(len(d) for d in train_datasets)
    print(f"  Train : {n_train:,} windows  Val : {len(ds_val):,} windows  ({time.time()-t0:.1f}s)\n")

    counts = sum(ds.class_counts().astype(np.float64) for ds in train_datasets)
    weights = torch.tensor(
        (counts.sum() / (N_CLASSES * np.where(counts > 0, counts, 1.0))).astype(np.float32),
        device=device,
    )

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=(device.type == "cuda"),
                     persistent_workers=(args.num_workers > 0))
    train_loader = DataLoader(ds_train, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(ds_val,   shuffle=False, **loader_kw)

    # ---- load MOMENT in classification mode ----
    print("Loading MOMENT-1-large (classification mode) …")
    t0 = time.time()
    moment = MOMENTPipeline.from_pretrained(
        MOMENT_ID,
        model_kwargs={
            "task_name":      "classification",
            "n_channels":     9,
            "num_class":      N_CLASSES,
            "freeze_encoder": False,   # peft will manage freezing
            "freeze_embedder": False,
        },
    )
    moment.init()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ---- freeze all params ----
    for p in moment.parameters():
        p.requires_grad = False

    # ---- apply LoRA to top `lora_blocks` encoder blocks ----
    n_blocks     = 24   # MOMENT-1-large has 24 T5 blocks
    first_block  = n_blocks - args.lora_blocks   # e.g. 20 for lora_blocks=4
    target_mods  = [
        f"encoder.block.{i}.layer.0.SelfAttention.{proj}"
        for i in range(first_block, n_blocks)
        for proj in ("q", "v")
    ]

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q", "v"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    moment = get_peft_model(moment, lora_cfg)

    # Force requires_grad on the patch embedding output so T5Stack's gradient
    # checkpointing sees a tensor that needs grad and recomputes properly.
    # peft 0.6.2 lacks enable_input_require_grads(), so we register it manually.
    def _require_grad_hook(module, input, output):
        output.requires_grad_(True)
    moment.base_model.model.patch_embedding.register_forward_hook(_require_grad_hook)

    # Only LoRA adapter weights (lora_A, lora_B) in top blocks + classification head.
    # Freezing the original W_q/W_v matrices keeps optimizer states small (~140K params).
    for name, p in moment.named_parameters():
        is_lora_adapter = ("lora_A" in name or "lora_B" in name)
        is_head         = "head" in name or "classifier" in name
        p.requires_grad = is_lora_adapter or is_head

    trainable = sum(p.numel() for p in moment.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in moment.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    moment = moment.to(device)

    # Two param groups: LoRA adapters at low LR, head at higher LR
    lora_params = [p for n, p in moment.named_parameters()
                   if p.requires_grad and "head" not in n and "classifier" not in n]
    head_params = [p for n, p in moment.named_parameters()
                   if p.requires_grad and ("head" in n or "classifier" in n)]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler    = torch.amp.GradScaler() if use_amp else None

    print(f"\n{'Ep':>4}  {'TrainLoss':>9}  {'TrainAcc':>8}  "
          f"{'ValAcc':>7}  {'MacroF1':>8}  {'LR':>8}  {'Time':>6}")
    print("-" * 68)

    best_f1    = 0.0
    best_state = None
    no_improve = 0
    history    = []
    t_train    = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        tr_loss, tr_acc = train_epoch(moment, train_loader, optimizer, criterion,
                                      device, scaler)
        val_acc, val_f1, _, _ = eval_epoch(moment, val_loader, device)
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()
        ep_time = time.time() - t_ep

        print(f"{epoch:>4}  {tr_loss:>9.4f}  {tr_acc:>7.3%}  "
              f"{val_acc:>7.3%}  {val_f1:>8.4f}  {lr_now:>8.2e}  {ep_time:>5.1f}s",
              flush=True)

        history.append(dict(epoch=epoch, train_loss=round(tr_loss,5),
                            train_acc=round(tr_acc,4), val_acc=round(val_acc,4),
                            val_macro_f1=round(val_f1,4), lr=round(lr_now,8),
                            epoch_time_s=round(ep_time,1)))

        if val_f1 >= best_f1:
            best_f1    = val_f1
            # Save only trainable (LoRA + head) weights
            best_state = {k: v.cpu().clone() for k, v in moment.state_dict().items()
                          if any(n == k for n, p in moment.named_parameters() if p.requires_grad)}
            no_improve = 0
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break

    total_time = time.time() - t_train
    print(f"\nTraining done in {total_time:.1f}s   best val macro-F1 = {best_f1:.4f}")

    # ---- final eval ----
    moment.load_state_dict(best_state, strict=False)
    final_acc, final_f1, final_preds, final_labels = eval_epoch(moment, val_loader, device)
    report = classification_report(
        final_labels, final_preds,
        labels=list(range(N_CLASSES)),
        target_names=[LABEL_MAP[i] for i in range(N_CLASSES)],
        zero_division=0,
    )
    print(f"\n{'='*60}")
    print(f"Best model — Accuracy: {final_acc:.4f}   Macro-F1: {final_f1:.4f}")
    print(f"{'='*60}")
    print(report)

    cm = confusion_matrix(final_labels, final_preds, labels=list(range(N_CLASSES)))
    header   = "Pred ->  " + "  ".join(f"{LABEL_MAP[i]:>8}" for i in range(N_CLASSES))
    cm_lines = [header]
    for i, row in enumerate(cm):
        cm_lines.append(f"{LABEL_MAP[i]:>8}  " + "  ".join(f"{v:>8}" for v in row))
    cm_str = "\n".join(cm_lines)

    # Save LoRA adapter weights + head
    torch.save(best_state, run_dir / "lora_weights.pt")
    (run_dir / "classification_report.txt").write_text(report)
    (run_dir / "confusion_matrix.txt").write_text(cm_str)

    config = dict(
        model="MOMENT-1-large+LoRA", positions=POSITIONS, fusion="pool",
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_blocks=args.lora_blocks, lora_target="q,v",
        trainable_params=trainable, total_params=total,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, head_lr=args.head_lr,
        patience=args.patience, sample_limit=args.sample_limit,
        seed=args.seed, device=str(device),
        note="PAPER EXPERIMENT ONLY — violates frozen-FM competition rule",
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    metrics = dict(
        best_val_macro_f1=round(final_f1, 4),
        best_val_accuracy=round(final_acc, 4),
        total_time_s=round(total_time, 1),
        epochs_trained=len(history),
        history=history,
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved to {run_dir.relative_to(REPO_ROOT)}/")
    print(f"  Macro-F1 : {final_f1:.4f}")
    print(f"  Accuracy : {final_acc:.4f}")


if __name__ == "__main__":
    main()
