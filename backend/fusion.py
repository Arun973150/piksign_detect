"""Weighted fusion - local AI-image ensemble heavy (0.60), SynthID next.

Pathways and weights:
  ensemble           0.60   Local strict-loaded PikSign ensemble (TOP weight)
  synthid            0.17   Google watermark (next highest) + bonus
  noise_residual     0.08   Wavelet noise inconsistency
  metadata           0.06   EXIF / AI-tool fingerprints
  text_analysis      0.02   OCR + glyph quality + cross-region consistency
                            (only counted when text is present)

  dimension          informational only - NOT in score

When SynthID detects strong tier, an additional bonus of up to +0.10 is added
on top of its weighted contribution (max +0.30 combined).

Verdict thresholds: REAL < 0.30 < UNCERTAIN < 0.70 < AI_GENERATED
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


BASE_WEIGHTS = {
    "ensemble": 0.60,
    "synthid": 0.17,
    "noise_residual": 0.08,
    "metadata": 0.06,
    "text_analysis": 0.00,
}

TEXT_WEIGHT_WHEN_PRESENT = 0.02
SYNTHID_STRONG_BONUS = 0.10

VERDICT_THRESHOLDS = {
    "real": 0.30,
    "ai": 0.70,
}


@dataclass
class FusionResult:
    verdict: str
    probability: float
    contributions: Dict[str, float] = field(default_factory=dict)
    weights_used: Dict[str, float] = field(default_factory=dict)
    synthid_bonus: float = 0.0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "probability": self.probability,
            "contributions": self.contributions,
            "weights_used": self.weights_used,
            "synthid_bonus": self.synthid_bonus,
            "reasons": self.reasons,
        }


def _verdict(p: float) -> str:
    if p < VERDICT_THRESHOLDS["real"]:
        return "REAL"
    if p > VERDICT_THRESHOLDS["ai"]:
        return "AI_GENERATED"
    return "UNCERTAIN"


def fuse(
    pathway_probs: Dict[str, Optional[float]],
    text_present: bool = False,
    synthid_tier: str = "none",
) -> FusionResult:
    """Combine pathway probabilities into a final verdict.

    pathway_probs values may be None when a pathway is unavailable; that
    pathway's weight is then redistributed across the active ones.
    """
    reasons: List[str] = []

    weights = {k: v for k, v in BASE_WEIGHTS.items() if k != "text_analysis"}
    if text_present:
        weights["text_analysis"] = TEXT_WEIGHT_WHEN_PRESENT
    elif "text_analysis" in pathway_probs:
        reasons.append("text_analysis skipped (no text detected)")

    active = {}
    for name, w in weights.items():
        prob = pathway_probs.get(name)
        if prob is None:
            reasons.append(f"{name} unavailable")
            continue
        active[name] = w

    total_w = sum(active.values())
    if total_w <= 0:
        reasons.append("No pathway produced a signal")
        return FusionResult(verdict="UNCERTAIN", probability=0.0, reasons=reasons)

    normalized = {k: v / total_w for k, v in active.items()}
    contributions = {k: normalized[k] * pathway_probs[k] for k in normalized}
    combined = sum(contributions.values())

    bonus = 0.0
    if synthid_tier == "strong":
        bonus = SYNTHID_STRONG_BONUS
        combined = min(1.0, combined + bonus)
        reasons.append(f"SynthID strong-tier bonus +{bonus:.2f}")
    elif synthid_tier == "moderate":
        bonus = SYNTHID_STRONG_BONUS * 0.5
        combined = min(1.0, combined + bonus)
        reasons.append(f"SynthID moderate-tier bonus +{bonus:.2f}")

    return FusionResult(
        verdict=_verdict(combined),
        probability=float(combined),
        contributions={k: float(v) for k, v in contributions.items()},
        weights_used={k: float(v) for k, v in normalized.items()},
        synthid_bonus=float(bonus),
        reasons=reasons,
    )
