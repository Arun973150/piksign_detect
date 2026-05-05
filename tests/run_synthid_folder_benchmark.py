"""Benchmark SynthID confidence on two folders.

Default:
  AI folder:   ../detection/generated_nano
  Real folder: ../detection/real

Run from piksign_detect:
  python tests/run_synthid_folder_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.detectors.synthid import SynthIDDetector


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def images(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai-dir", default=str(REPO_ROOT / "detection" / "generated_nano"))
    parser.add_argument("--real-dir", default=str(REPO_ROOT / "detection" / "real"))
    parser.add_argument("--csv", default="synthid_nano_vs_real.csv")
    args = parser.parse_args()

    ai_dir = Path(args.ai_dir).resolve()
    real_dir = Path(args.real_dir).resolve()
    samples = [("AI", p) for p in images(ai_dir)] + [("Real", p) for p in images(real_dir)]
    if not samples:
        print("No images found.")
        return 2

    detector = SynthIDDetector()
    rows = []
    print(f"AI folder:   {ai_dir}")
    print(f"Real folder: {real_dir}")
    print(f"Images: AI={sum(1 for l, _ in samples if l == 'AI')} Real={sum(1 for l, _ in samples if l == 'Real')}")
    print("-" * 118)
    print(f"{'Actual':<6} {'Confidence':>10} {'Detected':>9} {'Phase':>8} {'Struct':>8} {'Corr':>10} File")
    print("-" * 118)

    for label, path in samples:
        r = detector.detect(str(path))
        row = {
            "Actual": label,
            "File": str(path),
            "Confidence": f"{r.confidence:.6f}",
            "Detected": r.detected,
            "Tier": r.tier,
            "Correlation": f"{r.correlation:.8f}",
            "Phase Match": f"{r.phase_match:.6f}",
            "Structure Ratio": f"{r.structure_ratio:.6f}",
            "Multi Scale Consistency": f"{r.multi_scale_consistency:.8f}",
            "Error": r.error or "",
        }
        rows.append(row)
        print(
            f"{label:<6} {r.confidence:>10.4f} {str(r.detected):>9} "
            f"{r.phase_match:>8.4f} {r.structure_ratio:>8.4f} {r.correlation:>10.6f} {path.name}"
        )

    csv_path = Path(args.csv).resolve()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 118)
    print("Threshold sweep on Robust Confidence:")
    vals = [(r["Actual"], float(r["Confidence"])) for r in rows]
    best = None
    for th in [i / 100 for i in range(30, 96, 5)]:
        tp = sum(1 for label, score in vals if label == "AI" and score >= th)
        fn = sum(1 for label, score in vals if label == "AI" and score < th)
        fp = sum(1 for label, score in vals if label == "Real" and score >= th)
        tn = sum(1 for label, score in vals if label == "Real" and score < th)
        acc = (tp + tn) / len(vals)
        line = f"th={th:.2f} acc={acc:.3f} tp={tp} fn={fn} fp={fp} tn={tn}"
        print(line)
        if best is None or acc > best[0]:
            best = (acc, th, line)

    if best:
        print(f"Best: {best[2]}")
    print(f"CSV written: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
