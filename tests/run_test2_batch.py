"""Batch test PikSign Detect on the local "test 2" image set.

Default behavior:
  - Finds ../test 2/test 2 fake and ../test 2/test 2 original
  - Picks 6 shared subjects, one fake and one original per subject
  - Runs the detector
  - Prints a compact table
  - Writes CSV to test2_results.csv

Run from piksign_detect:
  python tests/run_test2_batch.py

Useful variants:
  python tests/run_test2_batch.py --limit 8
  python tests/run_test2_batch.py --skip-ensemble
  python tests/run_test2_batch.py --only-ensemble
  python tests/run_test2_batch.py --only-synthid
  python tests/run_test2_batch.py --all
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.orchestrator import PikSignDetector


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class Sample:
    label: str
    subject: str
    path: Path


def subject_name(path: Path) -> str:
    name = path.stem.lower()
    name = re.sub(r"\b(fake|fakes|real|reals)\b", "", name)
    name = re.sub(r"\s+\d+$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def list_images(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def choose_best(paths: Iterable[Path], label: str) -> Path:
    paths = list(paths)

    def score(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        ext_rank = 0 if path.suffix.lower() in {".jpg", ".jpeg"} else 1
        exact_rank = 0
        if label == "fake" and re.search(r"\bfake\b", name):
            exact_rank = -1
        if label == "real" and re.search(r"\breal\b", name):
            exact_rank = -1
        return (exact_rank, ext_rank, name)

    return sorted(paths, key=score)[0]


def build_samples(root: Path, limit: int, use_all: bool) -> list[Sample]:
    fake_dir = root / "test 2 fake"
    real_dir = root / "test 2 original"
    if not fake_dir.is_dir() or not real_dir.is_dir():
        raise FileNotFoundError(f"Expected folders not found under {root}")

    fake_by_subject: dict[str, list[Path]] = {}
    real_by_subject: dict[str, list[Path]] = {}
    for p in list_images(fake_dir):
        fake_by_subject.setdefault(subject_name(p), []).append(p)
    for p in list_images(real_dir):
        real_by_subject.setdefault(subject_name(p), []).append(p)

    if use_all:
        samples = [Sample("fake", subject_name(p), p) for p in list_images(fake_dir)]
        samples += [Sample("real", subject_name(p), p) for p in list_images(real_dir)]
        return samples

    common = sorted(set(fake_by_subject) & set(real_by_subject))[:limit]
    samples: list[Sample] = []
    for subject in common:
        samples.append(Sample("fake", subject, choose_best(fake_by_subject[subject], "fake")))
        samples.append(Sample("real", subject, choose_best(real_by_subject[subject], "real")))
    return samples


def display_actual(label: str) -> str:
    return "AI" if label == "fake" else "Real"


def display_prediction(verdict: str) -> str:
    return {
        "AI_GENERATED": "AI",
        "REAL": "Real",
        "UNCERTAIN": "Unsure",
        "PROTECTED_ORIGIN": "Protected",
    }.get(verdict.upper(), verdict)


def expected_pass(label: str, verdict: str, allow_uncertain: bool = True) -> bool:
    verdict = verdict.upper()
    if label == "fake":
        expected = {"AI_GENERATED"}
        if allow_uncertain:
            expected.add("UNCERTAIN")
        return verdict in expected
    expected = {"REAL", "PROTECTED_ORIGIN"}
    if allow_uncertain:
        expected.add("UNCERTAIN")
    return verdict in expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2] / "test 2"))
    parser.add_argument("--limit", type=int, default=6, help="subjects per side")
    parser.add_argument("--all", action="store_true", help="run every image in both folders")
    parser.add_argument("--skip-ensemble", action="store_true", help="skip the local AI Check ensemble")
    parser.add_argument("--skip-text", action="store_true", help="skip OCR/text check")
    parser.add_argument("--only-ensemble", action="store_true", help="run only the local AI Check ensemble")
    parser.add_argument("--only-synthid", action="store_true", help="run only the Authenticity Signal/SynthID pathway")
    parser.add_argument("--csv", default="test2_results.csv")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    samples = build_samples(root, args.limit, args.all)
    if not samples:
        print(f"No samples found in {root}")
        return 2

    if args.only_ensemble:
        detector = PikSignDetector(
            enable_ensemble=True,
            enable_synthid=False,
            enable_ela=False,
            enable_noise_residual=False,
            enable_metadata=False,
            enable_text_analysis=False,
            enable_dimension=True,
        )
        if not detector.ensemble or not detector.ensemble.available:
            err = detector.ensemble.error if detector.ensemble else "disabled"
            print(f"AI Check ensemble is not available: {err}")
            print("Install torch + torchvision and try again.")
            return 3
    elif args.only_synthid:
        detector = PikSignDetector(
            enable_ensemble=False,
            enable_synthid=True,
            enable_ela=False,
            enable_noise_residual=False,
            enable_metadata=False,
            enable_text_analysis=False,
            enable_dimension=True,
        )
    else:
        detector = PikSignDetector(
            enable_ensemble=not args.skip_ensemble,
            enable_text_analysis=not args.skip_text,
        )

    rows = []
    allow_uncertain = not (args.only_ensemble or args.only_synthid)
    print(f"Running {len(samples)} images from {root}")
    print("-" * 112)
    print(
        f"{'Actual':<7} {'Subject':<14} {'Prediction':<10} {'Score':>6} "
        f"{'Correct':<7} {'AI Check':>9} {'Signal':>8} {'Visual':>8} {'Texture':>8} File"
    )
    print("-" * 112)

    start = time.time()
    for sample in samples:
        report = detector.detect(str(sample.path))
        fusion = report.fusion.to_dict() if report.fusion else {}
        contributions = fusion.get("contributions", {})

        ok = expected_pass(sample.label, report.verdict, allow_uncertain=allow_uncertain)
        ens_status = report.ensemble.status if report.ensemble else ""
        ens_error = report.ensemble.error if report.ensemble else ""
        row = {
            "Actual Label": display_actual(sample.label),
            "Subject": sample.subject,
            "File": str(sample.path),
            "Prediction": display_prediction(report.verdict),
            "Raw Verdict": report.verdict,
            "Final Score": f"{report.probability:.4f}",
            "Correct": ok,
            "AI Check Score": f"{report.ensemble.probability:.4f}" if report.ensemble else "",
            "AI Check Status": ens_status,
            "AI Check Error": ens_error,
            "AI Check Models": ",".join(report.ensemble.loaded_models) if report.ensemble else "",
            "Authenticity Signal Score": f"{report.synthid.probability:.4f}" if report.synthid else "",
            "Visual Consistency Score": f"{report.ela.probability:.4f}" if report.ela else "",
            "Texture Consistency Score": f"{report.noise_residual.probability:.4f}" if report.noise_residual else "",
            "File Integrity Score": f"{report.metadata.probability:.4f}" if report.metadata else "",
            "Elapsed Seconds": f"{report.elapsed_seconds:.2f}",
            "AI Check Contribution": f"{contributions.get('ensemble', 0.0):.4f}",
            "Authenticity Signal Contribution": f"{contributions.get('synthid', 0.0):.4f}",
        }
        rows.append(row)
        print(
            f"{row['Actual Label']:<7} {sample.subject:<14} {row['Prediction']:<10} "
            f"{report.probability:>6.3f} {str(ok):<7} "
            f"{row['AI Check Score']:>9} {row['Authenticity Signal Score']:>8} "
            f"{row['Visual Consistency Score']:>8} "
            f"{row['Texture Consistency Score']:>8} "
            f"{sample.path.name}"
        )

    elapsed = time.time() - start
    passed = sum(1 for r in rows if r["Correct"])
    total = len(rows)

    csv_path = Path(args.csv).resolve()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 112)
    print(f"Passed: {passed}/{total} | Accuracy-style pass rate: {passed / total:.1%} | Time: {elapsed:.1f}s")
    print(f"CSV written: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
