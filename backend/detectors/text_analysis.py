"""Text / glyph analysis with cross-region consistency.

When text is present in the image, AI generators leave fingerprints in the
rendered glyphs:

  Per-region signals:
    - Lower OCR confidence (mangled letterforms)
    - Stroke-width variance within a single word
    - Soft / blurry letter edges (low Laplacian variance)

  Cross-region consistency signals (NEW):
    - Multiple text regions in a real photo SHOULD share the same font height,
      stroke style, and color profile (it's the same scene captured by one
      camera). AI-generated images often have inconsistent text — different
      "fonts" in different regions, mismatched scales, different stroke
      properties. We measure inter-region variance.

If no text is detected, returns available=False so fusion redistributes.

Method:
  1. easyocr OCR
  2. Per-region: confidence, edge sharpness, stroke variance
  3. Cross-region: variance of font heights, stroke widths, mean intensity
  4. Aggregate
"""

import base64
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass
class TextAnalysisResult:
    available: bool
    has_text: bool
    probability: float
    n_regions: int
    avg_confidence: float
    avg_sharpness: float
    avg_stroke_var: float
    height_variance: float
    stroke_width_variance: float
    intensity_variance: float
    inconsistency_score: float
    error: Optional[str] = None
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "has_text": self.has_text,
            "probability": self.probability,
            "n_regions": self.n_regions,
            "avg_confidence": self.avg_confidence,
            "avg_sharpness": self.avg_sharpness,
            "avg_stroke_var": self.avg_stroke_var,
            "height_variance": self.height_variance,
            "stroke_width_variance": self.stroke_width_variance,
            "intensity_variance": self.intensity_variance,
            "inconsistency_score": self.inconsistency_score,
            "error": self.error,
            "details": self.details,
        }


_reader = None


def _get_reader(languages, gpu):
    global _reader
    if _reader is not None:
        return _reader
    try:
        from PIL import Image
        if not hasattr(Image, "ANTIALIAS"):
            Image.ANTIALIAS = Image.Resampling.LANCZOS

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.cuda")
            import easyocr
            _reader = easyocr.Reader(languages or ["en"], gpu=gpu, verbose=False)
        return _reader
    except Exception:
        return None


def _ocr_variants(img_rgb: np.ndarray):
    variants = [(img_rgb, 1.0)]
    h, w = img_rgb.shape[:2]

    if max(h, w) < 1800:
        scale = min(2.0, 1800 / max(h, w))
        variants.append((
            cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC),
            scale,
        ))

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    sharp = cv2.addWeighted(clahe, 1.6, cv2.GaussianBlur(clahe, (0, 0), 1.0), -0.6, 0)
    variants.append((cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB), 1.0))

    if max(h, w) < 1800:
        scale = min(2.0, 1800 / max(h, w))
        big = cv2.resize(sharp, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        variants.append((cv2.cvtColor(big, cv2.COLOR_GRAY2RGB), scale))

    return variants


def _readtext_best(reader, img_rgb: np.ndarray):
    best = []
    for variant, scale in _ocr_variants(img_rgb):
        try:
            results = reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                text_threshold=0.35,
                low_text=0.20,
                link_threshold=0.20,
                canvas_size=2560,
                mag_ratio=1.5,
            )
        except TypeError:
            results = reader.readtext(variant)

        clean = []
        for bbox, text, conf in results:
            if not str(text).strip() or float(conf) < 0.05:
                continue
            if scale != 1.0:
                bbox = [[float(p[0]) / scale, float(p[1]) / scale] for p in bbox]
            clean.append((bbox, text, float(conf)))
        if len(clean) > len(best):
            best = clean
    return best


class TextAnalysisPathway:
    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        self.languages = languages or ["en"]
        self.gpu = gpu

    def detect(self, image_path: str) -> TextAnalysisResult:
        img = cv2.imread(str(image_path))
        if img is None:
            return TextAnalysisResult(
                available=False, has_text=False, probability=0.0,
                n_regions=0, avg_confidence=0, avg_sharpness=0, avg_stroke_var=0,
                height_variance=0, stroke_width_variance=0, intensity_variance=0,
                inconsistency_score=0,
                error=f"Could not load: {image_path}",
            )
        return self.detect_from_array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def detect_from_array(self, img_rgb: np.ndarray) -> TextAnalysisResult:
        reader = _get_reader(self.languages, self.gpu)
        if reader is None:
            return TextAnalysisResult(
                available=False, has_text=False, probability=0.0,
                n_regions=0, avg_confidence=0, avg_sharpness=0, avg_stroke_var=0,
                height_variance=0, stroke_width_variance=0, intensity_variance=0,
                inconsistency_score=0,
                error="easyocr not installed",
            )

        try:
            results = _readtext_best(reader, img_rgb)
        except Exception as e:
            return TextAnalysisResult(
                available=False, has_text=False, probability=0.0,
                n_regions=0, avg_confidence=0, avg_sharpness=0, avg_stroke_var=0,
                height_variance=0, stroke_width_variance=0, intensity_variance=0,
                inconsistency_score=0,
                error=f"OCR failed: {e}",
            )

        if not results:
            return TextAnalysisResult(
                available=False, has_text=False, probability=0.0,
                n_regions=0, avg_confidence=0, avg_sharpness=0, avg_stroke_var=0,
                height_variance=0, stroke_width_variance=0, intensity_variance=0,
                inconsistency_score=0,
            )

        raw_confs = [float(r[2]) for r in results]
        if len(results) < 4 and float(np.mean(raw_confs)) < 0.35:
            return TextAnalysisResult(
                available=False, has_text=False, probability=0.0,
                n_regions=0, avg_confidence=float(np.mean(raw_confs)),
                avg_sharpness=0, avg_stroke_var=0,
                height_variance=0, stroke_width_variance=0, intensity_variance=0,
                inconsistency_score=0,
                details={"discarded_low_confidence_regions": len(results)},
            )

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        confs: List[float] = []
        sharps: List[float] = []
        stroke_vars: List[float] = []
        heights: List[float] = []
        stroke_widths: List[float] = []
        intensities: List[float] = []

        snippets: List[str] = []

        for (bbox, text, conf) in results:
            confs.append(float(conf))
            if len(snippets) < 8:
                snippets.append(str(text)[:40])
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            x0, x1 = max(0, min(xs)), min(gray.shape[1], max(xs))
            y0, y1 = max(0, min(ys)), min(gray.shape[0], max(ys))
            w = x1 - x0
            h = y1 - y0
            if w < 5 or h < 5:
                continue

            patch = gray[y0:y1, x0:x1]
            sharps.append(float(cv2.Laplacian(patch, cv2.CV_64F).var()))

            _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            row_widths = (binary > 128).sum(axis=1).astype(np.float32)
            row_widths_nz = row_widths[row_widths > 0]
            if len(row_widths_nz) > 1:
                cv = row_widths_nz.std() / max(row_widths_nz.mean(), 1e-6)
                stroke_vars.append(float(cv))
                stroke_widths.append(float(row_widths_nz.mean()))

            heights.append(float(h))
            intensities.append(float(patch.mean()))

        if not confs:
            return TextAnalysisResult(
                available=False, has_text=False, probability=0.0,
                n_regions=0, avg_confidence=0, avg_sharpness=0, avg_stroke_var=0,
                height_variance=0, stroke_width_variance=0, intensity_variance=0,
                inconsistency_score=0,
            )

        avg_conf = float(np.mean(confs))
        avg_sharp = float(np.mean(sharps)) if sharps else 0.0
        avg_stroke_var = float(np.mean(stroke_vars)) if stroke_vars else 0.0

        height_var = float(np.std(heights) / max(np.mean(heights), 1e-6)) if len(heights) > 1 else 0.0
        stroke_w_var = float(np.std(stroke_widths) / max(np.mean(stroke_widths), 1e-6)) if len(stroke_widths) > 1 else 0.0
        int_var = float(np.std(intensities) / max(np.mean(intensities), 1e-6)) if len(intensities) > 1 else 0.0

        conf_score = float(np.clip((0.85 - avg_conf) / 0.40, 0, 1))
        sharp_score = float(np.clip((150.0 - avg_sharp) / 120.0, 0, 1))
        stroke_score = float(np.clip((avg_stroke_var - 0.30) / 0.30, 0, 1))

        height_inc = float(np.clip((height_var - 0.30) / 0.30, 0, 1))
        stroke_inc = float(np.clip((stroke_w_var - 0.35) / 0.30, 0, 1))
        intensity_inc = float(np.clip((int_var - 0.30) / 0.30, 0, 1))
        inconsistency_score = (
            0.40 * height_inc + 0.30 * stroke_inc + 0.30 * intensity_inc
        )

        per_region = 0.40 * conf_score + 0.30 * sharp_score + 0.30 * stroke_score
        probability = float(np.clip(0.55 * per_region + 0.45 * inconsistency_score, 0, 1))

        return TextAnalysisResult(
            available=True,
            has_text=True,
            probability=probability,
            n_regions=len(confs),
            avg_confidence=avg_conf,
            avg_sharpness=avg_sharp,
            avg_stroke_var=avg_stroke_var,
            height_variance=height_var,
            stroke_width_variance=stroke_w_var,
            intensity_variance=int_var,
            inconsistency_score=inconsistency_score,
            details={
                "conf_score": conf_score,
                "sharp_score": sharp_score,
                "stroke_score": stroke_score,
                "height_inc": height_inc,
                "stroke_inc": stroke_inc,
                "intensity_inc": intensity_inc,
                "snippets": snippets,
            },
        )
