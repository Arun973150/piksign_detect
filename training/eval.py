"""Standalone evaluation entry point for a saved TopMiner checkpoint.

  python training/eval.py --checkpoint training/checkpoints/topminer.pth --data path/to/test_data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from training.data import (
    FolderRealFakeDataset,
    HFRealFakeDataset,
    build_transform,
    make_parquet_split,
)
from training.model import TopMinerImageDetector
from training.train import evaluate


def main():
    p = argparse.ArgumentParser(description="Evaluate a TopMiner checkpoint")
    p.add_argument("--checkpoint", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="Folder with real/ and fake/ subdirectories")
    src.add_argument("--hf-dataset", help="HuggingFace dataset ID")
    src.add_argument("--parquet", help="Local OpenFake-style parquet shard")
    p.add_argument("--hf-config", default=None,
                   help="Optional HuggingFace dataset config, e.g. 'core' or 'reddit' for OpenFake")
    p.add_argument("--hf-split", default="test")
    p.add_argument("--hf-image-field", default="image")
    p.add_argument("--hf-label-field", default="label")
    p.add_argument("--hf-label-real-value", default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--image-size", type=int, default=380)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    backbone = saved_args.get("backbone", "efficientnet_b4")
    pretrained_backbone = not saved_args.get("no_pretrained_backbone", False)

    model = TopMinerImageDetector(
        backbone_name=backbone,
        freeze_backbone=True,
        pretrained_backbone=pretrained_backbone,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tf = build_transform(args.image_size, level=0, train=False)
    if args.data:
        ds = FolderRealFakeDataset(args.data, transform=tf)
    elif args.parquet:
        train_ds, val_ds = make_parquet_split(
            parquet_path=args.parquet,
            val_split=0.5,
            train_transform=tf,
            eval_transform=tf,
            image_field=args.hf_image_field,
            label_field=args.hf_label_field,
            label_real_value=args.hf_label_real_value,
            max_samples=args.max_samples,
        )
        ds = val_ds if len(val_ds) else train_ds
    else:
        ds = HFRealFakeDataset(
            args.hf_dataset, split=args.hf_split, transform=tf,
            config=args.hf_config,
            image_field=args.hf_image_field, label_field=args.hf_label_field,
            label_real_value=args.hf_label_real_value,
            max_samples=args.max_samples,
        )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device == "cuda"))

    print(f"checkpoint: {args.checkpoint}")
    print(f"backbone:   {backbone}")
    print(f"dataset:    {len(ds)} images")
    print(f"threshold:  {args.threshold}")

    m = evaluate(model, loader, device, threshold=args.threshold)
    print()
    print(f"  loss      {m['loss']:.4f}")
    print(f"  accuracy  {m['acc']:.4f}")
    print(f"  precision {m['prec']:.4f}")
    print(f"  recall    {m['rec']:.4f}")
    print(f"  f1        {m['f1']:.4f}")
    print(f"  n         {m['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
