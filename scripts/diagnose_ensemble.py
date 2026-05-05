"""Diagnostic for the local PikSign ensemble.

Checks every enabled local model for:
  1. Cached checkpoint path and parameter count
  2. Architecture parameter count
  3. Name + shape coverage
  4. Strict load status
  5. Optional forward-pass probability on a supplied image

Usage:
  python scripts/diagnose_ensemble.py --image path/to/image.jpg
  python scripts/diagnose_ensemble.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

from backend.detectors.local_ensemble import (
    DEFAULT_CACHE_DIR,
    BaseDetectorEffNet,
    BmUCFDetector,
    LocalEnsemblePathway,
    NPRDetector,
    SPSLDetector,
    UCFDetector,
    _MODEL_SPECS,
    _TF_PIKSIGN_BASE,
    _TF_SPSL,
    _TF_UCF,
    _find_weight,
    _load_ckpt,
    _state_dict_coverage,
)


CLASSES = {
    "NPRDetector": NPRDetector,
    "UCFDetector": UCFDetector,
    "BmUCFDetector": BmUCFDetector,
    "SPSLDetector": SPSLDetector,
    "BaseDetectorEffNet": BaseDetectorEffNet,
}

TRANSFORMS = {
    "base": _TF_PIKSIGN_BASE,
    "ucf": _TF_UCF,
    "spsl": _TF_SPSL,
}


def _human_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _probability(output: torch.Tensor, mode: str) -> float:
    if mode == "softmax_ai":
        return float(torch.softmax(output, dim=1)[:, 1].item())
    return float(torch.sigmoid(output).flatten()[0].item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to a real image for inference test")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--show-keys", type=int, default=8)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Cache:  {cache_dir}")

    if args.image:
        img = Image.open(args.image).convert("RGB")
        print(f"Image:  {args.image} ({img.size[0]}x{img.size[1]})")
    else:
        rng = np.random.RandomState(42)
        img = Image.fromarray(rng.randint(0, 255, size=(384, 384, 3), dtype=np.uint8))
        print("Image:  synthetic random RGB 384x384 (pipeline check only)")

    any_bad = False

    for spec in _MODEL_SPECS:
        print("\n" + "=" * 75)
        print(f"  [{spec.name}]  {spec.filename}")
        print("=" * 75)

        wpath = _find_weight(spec.filename, cache_dir)
        if not wpath:
            print("  ! missing checkpoint")
            any_bad = True
            continue

        print(f"  ckpt path: {wpath}")
        print(f"  ckpt size: {wpath.stat().st_size / (1024 * 1024):.1f} MB")
        sd = _load_ckpt(wpath, device)
        for k in list(sd)[: args.show_keys]:
            v = sd[k]
            shape = tuple(v.shape) if hasattr(v, "shape") else "?"
            print(f"       {k:52s} shape={shape}")

        model = CLASSES[spec.cls_name]().to(device)
        audit = _state_dict_coverage(model, sd)
        coverage_pct = 100 * audit["coverage"]
        print()
        print(f"  model params: {_human_params(audit['model_params'])} ({audit['model_keys']} state keys)")
        print(f"  ckpt  params: {_human_params(audit['ckpt_params'])} ({audit['ckpt_keys']} state keys)")
        print(f"  matched keys: {audit['matched_keys']}")
        print(f"  coverage:     {coverage_pct:.1f}%")

        if audit["coverage"] < 0.999:
            print("  ! REJECTED: checkpoint does not fully match this architecture")
            any_bad = True
            continue

        model.load_state_dict(sd, strict=True)
        model.eval()
        with torch.no_grad():
            inp = TRANSFORMS[spec.transform_name](img).unsqueeze(0).to(device)
            prob = _probability(model(inp), spec.output)
        print(f"  strict load:  OK")
        print(f"  prob(AI):     {prob:.4f}")

    print("\n" + "=" * 75)
    print("  ENSEMBLE API CHECK")
    print("=" * 75)
    pathway = LocalEnsemblePathway(device=device, cache_dir=str(cache_dir))
    tmp_path = args.image
    if tmp_path:
        result = pathway.detect(tmp_path)
        print(f"  status:       {result.status}")
        print(f"  loaded:       {', '.join(result.loaded_models)}")
        print(f"  probability:  {result.probability:.4f}")
        print(f"  verdict:      {result.provider_status}")
    else:
        pathway._ensure_loaded()
        print(f"  loaded:       {', '.join(pathway._models)}")

    return 2 if any_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
