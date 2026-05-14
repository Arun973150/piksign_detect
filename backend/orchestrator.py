"""PikSign Detect orchestrator.

  Step 0:  PikSign self-check (skip detection if PikSign-protected)
  Layer 1: Run all pathways
  Fusion:  Weighted combination with the local AI-image ensemble
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from backend.detectors.piksign_check import PikSignCheck, PikSignCheckResult
from backend.detectors.synthid import SynthIDDetector, SynthIDResult
from backend.detectors.bitmind import BitMindPathway, BitMindResult
from backend.detectors.local_ensemble import (
    LocalEnsemblePathway, LocalEnsembleResult,
)
from backend.detectors.ela import ELAPathway, ELAResult
from backend.detectors.noise_residual import NoiseResidualPathway, NoiseResidualResult
from backend.detectors.metadata import MetadataPathway, MetadataResult
from backend.detectors.text_analysis import TextAnalysisPathway, TextAnalysisResult
from backend.detectors.dimension import DimensionPathway, DimensionResult
from backend.fusion import fuse, FusionResult


@dataclass
class HybridAIResult:
    available: bool
    status: str
    probability: float
    provider_status: str
    source: str
    weights_used: Dict[str, float] = field(default_factory=dict)
    bitmind_probability: Optional[float] = None
    local_probability: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "status": self.status,
            "probability": self.probability,
            "provider_status": self.provider_status,
            "source": self.source,
            "weights_used": self.weights_used,
            "bitmind_probability": self.bitmind_probability,
            "local_probability": self.local_probability,
            "error": self.error,
        }


@dataclass
class DetectionReport:
    image_path: str
    verdict: str
    probability: float
    elapsed_seconds: float

    piksign_check: PikSignCheckResult
    ai_check: Optional[Any] = None
    bitmind_api: Optional[BitMindResult] = None
    synthid: Optional[SynthIDResult] = None
    ensemble: Optional[LocalEnsembleResult] = None
    ela: Optional[ELAResult] = None
    noise_residual: Optional[NoiseResidualResult] = None
    metadata: Optional[MetadataResult] = None
    text_analysis: Optional[TextAnalysisResult] = None
    dimension: Optional[DimensionResult] = None
    fusion: Optional[FusionResult] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        def _opt(x):
            return x.to_dict() if x else None

        return {
            "image_path": self.image_path,
            "verdict": self.verdict,
            "probability": self.probability,
            "elapsed_seconds": self.elapsed_seconds,
            "piksign_check": self.piksign_check.to_dict(),
            "ai_check": _opt(self.ai_check),
            "bitmind_api": _opt(self.bitmind_api),
            "synthid": _opt(self.synthid),
            "ensemble": _opt(self.ensemble),
            "ela": _opt(self.ela),
            "noise_residual": _opt(self.noise_residual),
            "metadata": _opt(self.metadata),
            "text_analysis": _opt(self.text_analysis),
            "dimension": _opt(self.dimension),
            "fusion": _opt(self.fusion),
            "notes": self.notes,
        }


class PikSignDetector:
    def __init__(
        self,
        synthid_codebook_path: Optional[str] = None,
        ensemble_device: Optional[str] = None,
        ensemble_cache_dir: Optional[str] = None,
        enable_synthid: bool = True,
        enable_bitmind_api: bool = True,
        enable_ensemble: bool = True,
        enable_ela: bool = True,
        enable_noise_residual: bool = True,
        enable_metadata: bool = True,
        enable_text_analysis: bool = True,
        enable_dimension: bool = True,
    ):
        self.piksign_check = PikSignCheck()
        self.synthid = SynthIDDetector(synthid_codebook_path) if enable_synthid else None
        self.bitmind = BitMindPathway() if enable_bitmind_api else None
        self.ensemble = (
            LocalEnsemblePathway(device=ensemble_device, cache_dir=ensemble_cache_dir)
            if enable_ensemble else None
        )
        self.ela = ELAPathway() if enable_ela else None
        self.noise_residual = NoiseResidualPathway() if enable_noise_residual else None
        self.metadata = MetadataPathway() if enable_metadata else None
        self.text_analysis = TextAnalysisPathway() if enable_text_analysis else None
        self.dimension = DimensionPathway() if enable_dimension else None

    def detect(self, image_path: str) -> DetectionReport:
        t0 = time.time()
        image_path = str(image_path)

        ps = self.piksign_check.check(image_path)
        if ps.is_protected:
            return DetectionReport(
                image_path=image_path,
                verdict="PROTECTED_ORIGIN",
                probability=0.0,
                elapsed_seconds=time.time() - t0,
                piksign_check=ps,
                dimension=self.dimension.detect(image_path) if self.dimension else None,
                notes={"short_circuit": "PikSign-protected, skipping AI detection"},
            )

        meta = self._safe(self.metadata, image_path)
        dim = self.dimension.detect(image_path) if self.dimension else None

        short = self._metadata_short_circuit(meta)
        if short:
            verdict, probability, note = short
            fusion_res = FusionResult(
                verdict=verdict,
                probability=probability,
                contributions={"metadata": probability},
                weights_used={"metadata": 1.0},
                reasons=[note],
            )
            return DetectionReport(
                image_path=image_path,
                verdict=verdict,
                probability=probability,
                elapsed_seconds=time.time() - t0,
                piksign_check=ps,
                metadata=meta,
                dimension=dim,
                fusion=fusion_res,
                notes={"short_circuit": note},
            )

        sid = self._safe(self.synthid, image_path)
        bm = self._safe(self.bitmind, image_path)
        ens = self._safe(self.ensemble, image_path)
        ai_check = self._hybrid_ai_check(bm, ens)
        ela = self._safe(self.ela, image_path)
        nr = self._safe(self.noise_residual, image_path)
        txt = self._safe(self.text_analysis, image_path)

        probs: Dict[str, Optional[float]] = {}
        if ai_check and ai_check.available and ai_check.status == "success":
            probs["ensemble"] = ai_check.probability
        strong_camera_provenance = self._strong_camera_provenance(meta)
        suppress_synthid = (
            self._suppress_synthid_when_ai_check_low(ai_check)
            or strong_camera_provenance
        )
        if sid and sid.available:
            probs["synthid"] = 0.0 if suppress_synthid else (sid.probability if sid.detected else 0.0)
        if nr and nr.available:
            probs["noise_residual"] = nr.probability
        if meta and meta.available:
            probs["metadata"] = meta.probability

        text_present = bool(txt and txt.available and txt.has_text)
        if text_present:
            probs["text_analysis"] = txt.probability

        synthid_tier = sid.tier if (sid and sid.detected and not suppress_synthid) else "none"

        fusion_res = fuse(
            pathway_probs=probs,
            text_present=text_present,
            synthid_tier=synthid_tier,
        )

        external = self._external_model_override(
            ai_check,
            strong_camera_provenance=strong_camera_provenance,
        )
        if external:
            floor, note = external
            if fusion_res.probability < floor:
                fusion_res.probability = floor
                fusion_res.verdict = "AI_GENERATED"
                fusion_res.reasons.append(note)

        if suppress_synthid and sid and sid.detected:
            if strong_camera_provenance:
                fusion_res.reasons.append("Authenticity signal limited by real-camera provenance")
            else:
                fusion_res.reasons.append("Authenticity signal suppressed because AI Check is low")

        if meta and meta.available and meta.details.get("c2pa_ai_source"):
            fusion_res.probability = max(fusion_res.probability, 0.95)
            fusion_res.verdict = "AI_GENERATED"
            fusion_res.reasons.append("File integrity identifies AI-generated source")
        elif meta and meta.available and meta.details.get("c2pa_google_ai_edit"):
            fusion_res.probability = max(fusion_res.probability, 0.85)
            fusion_res.verdict = "AI_GENERATED"
            fusion_res.reasons.append("File integrity identifies AI-edited source")

        camera_real = self._real_camera_override(meta, ai_check)
        if camera_real:
            cap, note = camera_real
            if fusion_res.probability > cap:
                fusion_res.probability = cap
            fusion_res.verdict = "REAL"
            fusion_res.reasons.append(note)

        if fusion_res.verdict == "REAL" and self._classifier_uncertainty_cap(ai_check):
            fusion_res.verdict = "UNCERTAIN"
            fusion_res.probability = max(fusion_res.probability, 0.30)
            fusion_res.reasons.append("AI Check in uncertain range — verdict capped at UNCERTAIN")

        if sid and sid.detected and sid.tier == "strong":
            fusion_res.probability = max(fusion_res.probability, 0.95)
            fusion_res.verdict = "AI_GENERATED"
            fusion_res.reasons.append("Authenticity signal strong match — verdict forced to AI_GENERATED")

        return DetectionReport(
            image_path=image_path,
            verdict=fusion_res.verdict,
            probability=fusion_res.probability,
            elapsed_seconds=time.time() - t0,
            piksign_check=ps,
            ai_check=ai_check,
            bitmind_api=bm,
            synthid=sid,
            ensemble=ens,
            ela=ela,
            noise_residual=nr,
            metadata=meta,
            text_analysis=txt,
            dimension=dim,
            fusion=fusion_res,
        )

    @staticmethod
    def _safe(pathway, image_path: str):
        if pathway is None:
            return None

    @staticmethod
    def _hybrid_ai_check(bitmind, ensemble):
        bm_ok = bool(bitmind and bitmind.available and bitmind.status == "success")
        ens_ok = bool(ensemble and ensemble.available and ensemble.status == "success")

        if bm_ok and float(bitmind.probability) >= 0.80:
            return HybridAIResult(
                available=True,
                status="success",
                probability=float(bitmind.probability),
                provider_status=bitmind.provider_status,
                source="bitmind_override",
                weights_used={"bitmind_api": 1.0},
                bitmind_probability=float(bitmind.probability),
                local_probability=float(ensemble.probability) if ens_ok else None,
            )

        if bm_ok and ens_ok:
            prob = 0.70 * float(bitmind.probability) + 0.30 * float(ensemble.probability)
            return HybridAIResult(
                available=True,
                status="success",
                probability=prob,
                provider_status="AI" if prob >= 0.5 else "AUTHENTIC",
                source="bitmind_local_weighted",
                weights_used={"bitmind_api": 0.70, "local_ensemble": 0.30},
                bitmind_probability=float(bitmind.probability),
                local_probability=float(ensemble.probability),
            )

        if bm_ok:
            return HybridAIResult(
                available=True,
                status="success",
                probability=float(bitmind.probability),
                provider_status=bitmind.provider_status,
                source="bitmind_only",
                weights_used={"bitmind_api": 1.0},
                bitmind_probability=float(bitmind.probability),
            )

        if ens_ok:
            return HybridAIResult(
                available=True,
                status="success",
                probability=float(ensemble.probability),
                provider_status=ensemble.provider_status,
                source="local_only",
                weights_used={"local_ensemble": 1.0},
                local_probability=float(ensemble.probability),
                error=(bitmind.error if bitmind else None),
            )

        return HybridAIResult(
            available=False,
            status="unavailable",
            probability=0.0,
            provider_status="unavailable",
            source="none",
            error="BitMind API and local ensemble unavailable",
        )
        try:
            return pathway.detect(image_path)
        except Exception:
            return None

    @staticmethod
    def _metadata_short_circuit(meta):
        if not (meta and meta.available):
            return None

        details = meta.details or {}
        if details.get("c2pa_ai_source"):
            return ("AI_GENERATED", 0.95, "File integrity identifies AI-generated source")
        if details.get("c2pa_google_ai_edit"):
            return ("AI_GENERATED", 0.85, "File integrity identifies AI-edited source")
        if meta.has_ai_software:
            return ("AI_GENERATED", 0.90, "File integrity identifies AI software source")
        if meta.has_ai_text_chunk:
            return ("AI_GENERATED", 0.85, "File integrity identifies prompt/workflow metadata")

        # Do not stop on weak metadata absences such as "no EXIF"; WhatsApp and
        # social apps commonly remove those fields from real photos.
        return None

    @staticmethod
    def _ensemble_prob(ensemble) -> Optional[float]:
        if ensemble and ensemble.available and ensemble.status == "success":
            return float(ensemble.probability)
        return None

    @staticmethod
    def _external_model_override(ensemble, strong_camera_provenance: bool = False):
        prob = PikSignDetector._ensemble_prob(ensemble)
        if prob is None:
            return None

        if strong_camera_provenance:
            if prob >= 0.90:
                return (0.90, "AI Check has strong AI signal despite camera provenance")
            return None

        if prob >= 0.95:
            return (0.95, "AI Check has high-confidence AI signal")
        if prob >= 0.80:
            return (0.85, "AI Check has strong AI signal")
        return None

    @staticmethod
    def _strong_camera_provenance(meta) -> bool:
        if not (meta and meta.available):
            return False
        details = meta.details or {}
        ai_c2pa = details.get("c2pa_ai_source") or details.get("c2pa_google_ai_edit")
        has_capture_time = bool(details.get("date_time_original") or details.get("create_date"))
        return (
            meta.has_exif
            and meta.has_camera_make
            and meta.has_exposure_info
            and (meta.has_lens_info or bool(details.get("lens_model")))
            and has_capture_time
            and not meta.has_ai_software
            and not meta.has_ai_text_chunk
            and not ai_c2pa
        )

    @staticmethod
    def _real_camera_override(meta, ensemble):
        if not PikSignDetector._strong_camera_provenance(meta):
            return None
        prob = PikSignDetector._ensemble_prob(ensemble)
        if prob is not None and prob >= 0.80:
            return None
        return (0.24, "Strong real-camera provenance verified")

    @staticmethod
    def _suppress_synthid_when_ai_check_low(ensemble) -> bool:
        prob = PikSignDetector._ensemble_prob(ensemble)
        return prob is not None and prob <= 0.30

    @staticmethod
    def _classifier_uncertainty_cap(ensemble) -> bool:
        prob = PikSignDetector._ensemble_prob(ensemble)
        return prob is not None and 0.30 <= prob <= 0.70
