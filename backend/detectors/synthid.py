"""SynthID detector — wraps upstream RobustSynthIDExtractor (correct usage).

Previous version added a custom grading layer on top of upstream's output.
That caused disagreements (we'd say "not detected" when upstream said "True").

This version uses upstream's `is_watermarked` boolean **directly** as the
detection decision. The probability we expose is upstream's `confidence`
field. We map a tier label from confidence for UI display only.

Reference: backend/vendor/synthid/robust_extractor.py
"""

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np


_VENDORED = Path(__file__).resolve().parent.parent / "vendor" / "synthid"
if str(_VENDORED) not in sys.path:
    sys.path.insert(0, str(_VENDORED))

from robust_extractor import RobustSynthIDExtractor, DetectionResult


DEFAULT_CODEBOOK = Path(__file__).parent.parent / "assets" / "robust_codebook.pkl"


@dataclass
class SynthIDResult:
    available: bool
    detected: bool
    probability: float
    tier: str
    confidence: float
    correlation: float = 0.0
    phase_match: float = 0.0
    structure_ratio: float = 0.0
    multi_scale_consistency: float = 0.0
    error: Optional[str] = None
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "detected": self.detected,
            "probability": self.probability,
            "tier": self.tier,
            "confidence": self.confidence,
            "correlation": self.correlation,
            "phase_match": self.phase_match,
            "structure_ratio": self.structure_ratio,
            "multi_scale_consistency": self.multi_scale_consistency,
            "error": self.error,
        }


def _tier(conf: float, detected: bool) -> str:
    if not detected and conf < 0.45:
        return "none"
    if not detected:
        return "signal"
    if conf >= 0.80:
        return "strong"
    if conf >= 0.60:
        return "moderate"
    return "weak"


class SynthIDDetector:
    """Thin wrapper that trusts upstream's is_watermarked decision."""

    def __init__(self, codebook_path: Optional[str] = None):
        self.codebook_path = Path(codebook_path) if codebook_path else DEFAULT_CODEBOOK
        self.extractor: Optional[RobustSynthIDExtractor] = None
        self.error: Optional[str] = None

        try:
            self.extractor = RobustSynthIDExtractor()
            if self.codebook_path.exists():
                self.extractor.load_codebook(str(self.codebook_path))
            else:
                self.error = f"Codebook not found: {self.codebook_path}"
        except Exception as e:
            self.error = f"Init failed: {e}"

    @property
    def available(self) -> bool:
        return self.extractor is not None and self.extractor.codebook is not None

    def detect(self, image_path: str) -> SynthIDResult:
        if not self.available:
            return SynthIDResult(
                available=False, detected=False, probability=0.0,
                tier="none", confidence=0.0, error=self.error,
            )

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"pywt\._thresholding")
                res: DetectionResult = self.extractor.detect(image_path)
        except Exception as e:
            return SynthIDResult(
                available=False, detected=False, probability=0.0,
                tier="none", confidence=0.0, error=f"Detection failed: {e}",
            )

        detected = bool(res.is_watermarked)
        confidence = float(res.confidence)
        # Use upstream confidence as signal strength even when its strict
        # boolean gate is false. The boolean remains exposed as `detected`;
        # fusion bonuses still require a true upstream detection.
        probability = confidence
        tier = _tier(confidence, detected)

        return SynthIDResult(
            available=True,
            detected=detected,
            probability=probability,
            tier=tier,
            confidence=confidence,
            correlation=float(res.correlation),
            phase_match=float(res.phase_match),
            structure_ratio=float(res.structure_ratio),
            multi_scale_consistency=float(res.multi_scale_consistency),
            details=res.details,
        )
