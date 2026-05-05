"""Compare available SynthID-style detectors on the local test set.

Runs:
  1. Current robust extractor used by the app
  2. Legacy 250-image codebook detector from detection/reverse-SynthID

Run from piksign_detect:
  python tests/run_synthid_compare.py --limit 6
  python tests/run_synthid_compare.py --all
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
TESTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from backend.detectors.synthid import SynthIDDetector
from run_test2_batch import build_samples, display_actual


EXTRACTION_DIR = REPO_ROOT / "detection" / "reverse-SynthID" / "src" / "extraction"
LEGACY_CODEBOOK = REPO_ROOT / "detection" / "reverse-SynthID" / "artifacts" / "codebook" / "synthid_codebook.pkl"


def load_legacy_detector():
    # Compatibility for pickles made with a different NumPy package layout.
    import numpy.core.numeric as numeric

    sys.modules.setdefault("numpy._core.numeric", numeric)
    if str(EXTRACTION_DIR) not in sys.path:
        sys.path.insert(0, str(EXTRACTION_DIR))
    from synthid_codebook_extractor import detect_synthid

    return detect_synthid


def verdict_from_score(score: float) -> str:
    if score >= 0.70:
        return "AI"
    if score >= 0.30:
        return "Unsure"
    return "Real"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO_ROOT / "test 2"))
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--csv", default="synthid_compare.csv")
    args = parser.parse_args()

    samples = build_samples(Path(args.root).resolve(), args.limit, args.all)
    robust = SynthIDDetector()
    legacy_detect = load_legacy_detector()

    rows = []
    print(f"Running SynthID comparison on {len(samples)} images")
    print("-" * 128)
    print(
        f"{'Actual':<7} {'Subject':<14} {'Robust':>8} {'R-Detect':>9} "
        f"{'Legacy':>8} {'L-Detect':>9} {'Prediction':<10} File"
    )
    print("-" * 128)

    for sample in samples:
        r = robust.detect(str(sample.path))
        try:
            legacy = legacy_detect(str(sample.path), str(LEGACY_CODEBOOK))
            legacy_conf = float(legacy.get("confidence", 0.0))
            legacy_detected = bool(legacy.get("is_watermarked"))
        except Exception as e:
            legacy_conf = 0.0
            legacy_detected = False
            legacy = {"error": str(e)}

        combined = max(r.confidence, legacy_conf)
        prediction = verdict_from_score(combined)

        row = {
            "Actual Label": display_actual(sample.label),
            "Subject": sample.subject,
            "File": str(sample.path),
            "Robust Confidence": f"{r.confidence:.4f}",
            "Robust Detected": r.detected,
            "Legacy Confidence": f"{legacy_conf:.4f}",
            "Legacy Detected": legacy_detected,
            "Combined Prediction": prediction,
            "Robust Correlation": f"{r.correlation:.6f}",
            "Robust Phase": f"{r.phase_match:.4f}",
            "Robust Structure": f"{r.structure_ratio:.4f}",
            "Legacy Error": legacy.get("error", ""),
        }
        rows.append(row)
        print(
            f"{row['Actual Label']:<7} {sample.subject:<14} "
            f"{r.confidence:>8.4f} {str(r.detected):>9} "
            f"{legacy_conf:>8.4f} {str(legacy_detected):>9} "
            f"{prediction:<10} {sample.path.name}"
        )

    csv_path = Path(args.csv).resolve()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 128)
    print(f"CSV written: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
