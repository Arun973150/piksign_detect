"""Training entry point for the TopMiner image detector (Section 8.1)."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from training.data import (
    HFRealFakeDataset,
    build_transform,
    make_folder_split,
    make_parquet_split,
)
from training.model import TopMinerImageDetector


# Losses

class FocalLoss(nn.Module):
    """Paper Sec 8.1: focal loss to address class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.flatten()
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(targets == 1, p, 1 - p)
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(p, self.alpha),
            torch.full_like(p, 1 - self.alpha),
        )
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()


# Evaluation

@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5) -> dict:
    model.eval()
    crit = FocalLoss()
    correct = total = tp = fp = fn = 0
    losses: list[float] = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        losses.append(crit(logits, labels).item())
        probs = torch.sigmoid(logits.flatten())
        preds = (probs >= threshold).long()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    acc = correct / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "n": total,
    }


# Training

def parse_args():
    p = argparse.ArgumentParser(description="Train TopMiner image detector (paper Sec 8.1)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="Folder with real/ and fake/ subdirectories")
    src.add_argument("--hf-dataset", help="HuggingFace dataset ID (image+label fields)")
    src.add_argument("--parquet", help="Local OpenFake-style parquet shard for smoke tests")

    p.add_argument("--hf-config", default=None,
                   help="Optional HuggingFace dataset config, e.g. 'core' for ComplexDataLab/OpenFake")
    p.add_argument("--hf-train-split", default="train",
                   help="HF train split expression, e.g. train[:2000]")
    p.add_argument("--hf-val-split", default="validation",
                   help="HF validation split expression; falls back to test if unavailable")
    p.add_argument("--hf-image-field", default="image")
    p.add_argument("--hf-label-field", default="label")
    p.add_argument("--hf-label-real-value", default=0,
                   help="Value in --hf-label-field that means REAL. Use 'real' for OpenFake.")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Optional cap for quick HuggingFace pilot runs")
    p.add_argument("--max-val-samples", type=int, default=None,
                   help="Optional validation/test sample cap for quick HuggingFace pilot runs")
    p.add_argument("--max-parquet-samples", type=int, default=None,
                   help="Optional row cap for --parquet smoke tests")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=380)
    p.add_argument("--aug-level", type=int, default=2, choices=[0, 1, 2, 3])

    p.add_argument("--backbone", default="efficientnet_b4",
                   help="timm backbone name. efficientnet_b0/b4 etc. b0 is much faster on CPU.")
    p.add_argument("--unfreeze-backbone", action="store_true",
                   help="Train the whole EfficientNet backbone (default: frozen)")
    p.add_argument("--no-pretrained-backbone", action="store_true",
                   help="Do not download/load ImageNet weights. Useful for offline smoke tests only.")

    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--output", default="training/checkpoints/topminer.pth")
    p.add_argument("--log-csv", default="training/checkpoints/train_log.csv")
    p.add_argument("--save-every-epoch", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:    {device}")
    print(f"backbone:  {args.backbone}  (frozen={not args.unfreeze_backbone})")
    print(f"image:     {args.image_size}px,  aug level {args.aug_level}")

    train_tf = build_transform(args.image_size, args.aug_level, train=True)
    eval_tf = build_transform(args.image_size, level=0, train=False)

    if args.data:
        train_ds, val_ds = make_folder_split(
            data_root=args.data, val_split=args.val_split,
            train_transform=train_tf, eval_transform=eval_tf, seed=args.seed,
        )
        print(f"folder:    {args.data} -> {len(train_ds)} train, {len(val_ds)} val")
    elif args.parquet:
        train_ds, val_ds = make_parquet_split(
            parquet_path=args.parquet,
            val_split=args.val_split,
            train_transform=train_tf,
            eval_transform=eval_tf,
            image_field=args.hf_image_field,
            label_field=args.hf_label_field,
            label_real_value=args.hf_label_real_value,
            max_samples=args.max_parquet_samples,
            seed=args.seed,
        )
        print(f"parquet:   {args.parquet} -> {len(train_ds)} train, {len(val_ds)} val")
    else:
        train_ds = HFRealFakeDataset(
            args.hf_dataset, config=args.hf_config, split=args.hf_train_split, transform=train_tf,
            image_field=args.hf_image_field, label_field=args.hf_label_field,
            label_real_value=args.hf_label_real_value,
            max_samples=args.max_train_samples,
        )
        try:
            val_ds = HFRealFakeDataset(
                args.hf_dataset, config=args.hf_config, split=args.hf_val_split, transform=eval_tf,
                image_field=args.hf_image_field, label_field=args.hf_label_field,
                label_real_value=args.hf_label_real_value,
                max_samples=args.max_val_samples,
            )
        except Exception:
            print("no 'validation' split, trying 'test'...")
            val_ds = HFRealFakeDataset(
                args.hf_dataset, config=args.hf_config, split="test", transform=eval_tf,
                image_field=args.hf_image_field, label_field=args.hf_label_field,
                label_real_value=args.hf_label_real_value,
                max_samples=args.max_val_samples,
            )
        cfg = f"/{args.hf_config}" if args.hf_config else ""
        print(f"hf:        {args.hf_dataset}{cfg} -> {len(train_ds)} train, {len(val_ds)} val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    model = TopMinerImageDetector(
        backbone_name=args.backbone,
        freeze_backbone=not args.unfreeze_backbone,
        pretrained_backbone=not args.no_pretrained_backbone,
    ).to(device)
    trainables = model.trainable_params()
    n_train_params = sum(p.numel() for p in trainables)
    print(f"trainable: {n_train_params/1e6:.2f}M params")

    optim = AdamW(trainables, lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=max(args.epochs, 1))
    loss_fn = FocalLoss()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_fields = ["epoch", "lr", "train_loss", "val_loss", "val_acc", "val_prec", "val_rec", "val_f1", "wall_s"]
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(log_fields)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item()
            n_batches += 1
        train_loss = running / max(n_batches, 1)

        metrics = evaluate(model, val_loader, device)
        sched.step()
        wall = time.time() - t0
        lr_now = optim.param_groups[0]["lr"]

        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"lr={lr_now:.2e}  "
            f"train={train_loss:.4f}  "
            f"val_loss={metrics['loss']:.4f}  "
            f"acc={metrics['acc']:.3f}  "
            f"prec={metrics['prec']:.3f}  "
            f"rec={metrics['rec']:.3f}  "
            f"f1={metrics['f1']:.3f}  "
            f"({wall:.1f}s)"
        )

        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, lr_now, train_loss,
                metrics["loss"], metrics["acc"], metrics["prec"], metrics["rec"], metrics["f1"],
                wall,
            ])

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "metrics": metrics,
            }, out_path)
            print(f"   -> saved best to {out_path}  (val_f1={best_f1:.3f})")

        if args.save_every_epoch:
            ep_path = out_path.with_name(f"{out_path.stem}_ep{epoch}{out_path.suffix}")
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "metrics": metrics,
            }, ep_path)

    print(f"\ndone. best val_f1 = {best_f1:.3f}  ->  {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
