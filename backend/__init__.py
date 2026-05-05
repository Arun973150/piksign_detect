"""PikSign Detect - production-ready AI-image and manipulation detection.

Pathways (weighted fusion):
  ensemble           0.60   Local strict-loaded PikSign classifier ensemble
  synthid            0.17   Google SynthID watermark (RobustSynthIDExtractor)
  noise_residual     0.08   Wavelet noise inconsistency (Pan & Lyu 2012)
  metadata           0.06   EXIF + AI-tool field detection
  text_analysis      0.02   OCR + glyph quality + cross-region consistency
                            (only when text is present)

Informational-only sidecar:
  dimension          aspect-ratio + crop detection (NOT in fusion score)
"""

from backend.orchestrator import DetectionReport, PikSignDetector

__all__ = ["PikSignDetector", "DetectionReport"]
