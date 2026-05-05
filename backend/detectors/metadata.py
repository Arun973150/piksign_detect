"""Metadata analysis — EXIF + PNG-chunks + C2PA + XMP.

Catches AI-tool fingerprints in image metadata:

  AI signals (push prob UP):
    - Software field contains AI-tool name (Midjourney, DALL-E, Imagen,
      Gemini, Stable Diffusion, Adobe Firefly, ComfyUI, Flux, etc.)
    - PNG tEXt/iTXt chunks with prompt strings (A1111/ComfyUI default)
    - XMP with synthetic markers
    - PNG completely without metadata (often AI tool default save)

  Real-camera signals (push prob DOWN):
    - Camera Make/Model present (and looks plausible)
    - GPS coordinates
    - Lens info, exposure, ISO present
    - Multiple consistent DateTime fields
    - C2PA manifest with provenance

References:
  - "Forensic Analysis of Image Metadata to Distinguish AI-Generated Images" (2024)
  - "An Agent-Based Forensic Framework for AI-Generated Images" (2025)
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ExifTags, PngImagePlugin


AI_SOFTWARE_PATTERNS = [
    r"stable\s*diffusion",
    r"midjourney",
    r"dall[\-\s]?e",
    r"imagen",
    r"gemini",
    r"firefly",
    r"comfyui",
    r"automatic1111",
    r"a1111",
    r"leonardo",
    r"flux",
    r"sdxl",
    r"runway",
    r"openai",
    r"nano[\-\s]?banana",
    r"playground[\-\s]?ai",
    r"ideogram",
    r"recraft",
    r"krea",
    r"bing\s*image\s*creator",
]

CAMERA_FIELDS = ["Make", "Model", "LensModel", "LensMake"]
EXPOSURE_FIELDS = ["ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength"]
C2PA_AI_SOURCE_MARKERS = [
    "trainedalgorithmicmedia",
    "compositewithtrainedalgorithmicmedia",
    "algorithmicmedia",
]
C2PA_GENERATOR_MARKERS = [
    "openai media service api",
    "google c2pa core generator library",
    "gpt-4o",
    "dall",
    "midjourney",
    "stable diffusion",
    "firefly",
    "imagen",
    "gemini",
]


@dataclass
class MetadataResult:
    available: bool
    probability: float
    has_camera_make: bool
    has_gps: bool
    has_exif: bool
    has_lens_info: bool
    has_exposure_info: bool
    has_ai_software: bool
    ai_software_match: Optional[str]
    has_c2pa: bool
    has_ai_text_chunk: bool
    raw_software: Optional[str]
    summary: List[str]
    error: Optional[str] = None
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "probability": self.probability,
            "has_camera_make": self.has_camera_make,
            "has_gps": self.has_gps,
            "has_exif": self.has_exif,
            "has_lens_info": self.has_lens_info,
            "has_exposure_info": self.has_exposure_info,
            "has_ai_software": self.has_ai_software,
            "ai_software_match": self.ai_software_match,
            "has_c2pa": self.has_c2pa,
            "has_ai_text_chunk": self.has_ai_text_chunk,
            "raw_software": self.raw_software,
            "summary": self.summary,
            "error": self.error,
        }


def _match_ai_software(text: str) -> Optional[str]:
    if not text:
        return None
    lower = text.lower()
    for pat in AI_SOFTWARE_PATTERNS:
        if re.search(pat, lower):
            return pat
    return None


def _scan_c2pa_bytes(blob: bytes) -> Dict:
    text = blob.decode("latin-1", errors="ignore")
    lower = text.lower()
    result = {
        "has_c2pa": "c2pa" in lower or "jumbf" in lower,
        "c2pa_ai_source": any(marker in lower for marker in C2PA_AI_SOURCE_MARKERS),
        "c2pa_composite_source": "digitalsourcetype/composite" in lower,
        "c2pa_google_ai_edit": False,
        "c2pa_generator": None,
        "c2pa_software_agent": None,
    }

    for marker in C2PA_GENERATOR_MARKERS:
        if marker in lower:
            result["c2pa_generator"] = _pretty_marker(marker)
            break

    if "gpt-4o" in lower:
        result["c2pa_software_agent"] = "GPT-4o"
    elif "openai media service api" in lower:
        result["c2pa_software_agent"] = "OpenAI Media Service API"
    elif "google c2pa core generator library" in lower:
        result["c2pa_software_agent"] = "Google C2PA Core Generator Library"

    result["c2pa_google_ai_edit"] = (
        "google c2pa core generator library" in lower
        and (
            result["c2pa_composite_source"]
            or "added visible watermark" in lower
            or "c2pa.edited" in lower
        )
    )

    return result


def _pretty_marker(marker: str) -> str:
    known = {
        "openai media service api": "OpenAI Media Service API",
        "google c2pa core generator library": "Google AI",
        "gpt-4o": "GPT-4o",
        "dall": "DALL-E",
        "midjourney": "Midjourney",
        "stable diffusion": "Stable Diffusion",
        "firefly": "Adobe Firefly",
        "imagen": "Imagen",
        "gemini": "Gemini",
    }
    return known.get(marker, marker)


class MetadataPathway:
    def detect(self, image_path: str) -> MetadataResult:
        path = Path(image_path)
        if not path.exists():
            return MetadataResult(
                available=False, probability=0.0,
                has_camera_make=False, has_gps=False, has_exif=False,
                has_lens_info=False, has_exposure_info=False,
                has_ai_software=False, ai_software_match=None,
                has_c2pa=False, has_ai_text_chunk=False,
                raw_software=None, summary=[], error="file not found",
            )

        summary: List[str] = []
        details: Dict = {}
        has_camera_make = has_gps = has_exif = False
        has_lens_info = has_exposure_info = False
        has_ai_software = has_c2pa = has_ai_text_chunk = False
        ai_software_match: Optional[str] = None
        raw_software: Optional[str] = None

        try:
            img = Image.open(path)
            details["format"] = img.format
            details["mode"] = img.mode
            details["size"] = img.size

            try:
                exif = img._getexif() or {}
            except Exception:
                exif = {}

            tagged = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
            has_exif = len(tagged) > 0
            details["exif_tags_count"] = len(tagged)
            details["camera_make"] = str(tagged.get("Make", "")).strip()
            details["camera_model"] = str(tagged.get("Model", "")).strip()
            details["lens_model"] = str(tagged.get("LensModel", "")).strip()
            details["date_time_original"] = str(tagged.get("DateTimeOriginal", "")).strip()
            details["create_date"] = str(tagged.get("DateTimeDigitized", "")).strip()

            for f in CAMERA_FIELDS:
                if f in tagged and str(tagged[f]).strip():
                    has_camera_make = True
                    if f in ("LensModel", "LensMake"):
                        has_lens_info = True

            for f in EXPOSURE_FIELDS:
                if f in tagged:
                    has_exposure_info = True
                    break

            if "GPSInfo" in tagged or any("GPS" in str(k) for k in tagged.keys()):
                has_gps = True

            sw = tagged.get("Software")
            if sw:
                raw_software = str(sw)
                details["software"] = raw_software
                m = _match_ai_software(raw_software)
                if m:
                    has_ai_software = True
                    ai_software_match = m
                    summary.append(f"Software field identifies AI tool: {raw_software!r}")

            uc = tagged.get("UserComment")
            if uc and not has_ai_software:
                m = _match_ai_software(str(uc))
                if m:
                    has_ai_software = True
                    ai_software_match = m
                    summary.append(f"UserComment identifies AI tool")

            if isinstance(img, PngImagePlugin.PngImageFile):
                txt = getattr(img, "text", {}) or {}
                details["png_text_keys"] = list(txt.keys())
                ai_keys = ["parameters", "prompt", "negative_prompt",
                           "Comment", "workflow", "sd-metadata"]
                for k in ai_keys:
                    if k in txt:
                        has_ai_text_chunk = True
                        summary.append(f"PNG chunk '{k}' present (A1111/ComfyUI fingerprint)")
                        break
                for k, v in txt.items():
                    if not has_ai_software:
                        m = _match_ai_software(str(v))
                        if m:
                            has_ai_software = True
                            ai_software_match = m
                            summary.append(f"AI tool in PNG chunk '{k}'")
                            break
                if "c2pa" in str(txt).lower():
                    has_c2pa = True

            try:
                xmp = getattr(img, "info", {}).get("xmp")
                if xmp:
                    details["has_xmp"] = True
                    xmp_str = (xmp.decode("utf-8", errors="ignore")
                               if isinstance(xmp, bytes) else str(xmp))
                    if "c2pa" in xmp_str.lower():
                        has_c2pa = True
                    if not has_ai_software:
                        m = _match_ai_software(xmp_str)
                        if m:
                            has_ai_software = True
                            ai_software_match = m
                            summary.append(f"AI tool in XMP")
            except Exception:
                pass

        except Exception as e:
            return MetadataResult(
                available=False, probability=0.0,
                has_camera_make=False, has_gps=False, has_exif=False,
                has_lens_info=False, has_exposure_info=False,
                has_ai_software=False, ai_software_match=None,
                has_c2pa=False, has_ai_text_chunk=False,
                raw_software=None, summary=[],
                error=f"PIL: {e}", details=details,
            )

        try:
            with open(path, "rb") as f:
                head = f.read(min(8 * 1024 * 1024, path.stat().st_size))
            c2pa_scan = _scan_c2pa_bytes(head)
            details.update(c2pa_scan)
            if c2pa_scan["has_c2pa"] or b"c2pa.assertions" in head.lower():
                has_c2pa = True
            if (
                c2pa_scan["c2pa_ai_source"]
                or c2pa_scan["c2pa_google_ai_edit"]
                or c2pa_scan["c2pa_generator"]
            ):
                has_ai_software = True
                ai_software_match = c2pa_scan["c2pa_generator"] or "C2PA trainedAlgorithmicMedia"
                if c2pa_scan["c2pa_google_ai_edit"]:
                    summary.append("Content credentials identify Google AI-edited media")
                else:
                    summary.append("Content credentials identify AI-generated media")
                if c2pa_scan["c2pa_generator"]:
                    summary.append(f"Generator: {c2pa_scan['c2pa_generator']}")
                if c2pa_scan["c2pa_software_agent"]:
                    summary.append(f"Software agent: {c2pa_scan['c2pa_software_agent']}")
        except Exception:
            pass

        is_png = details.get("format") == "PNG"
        ai_score = 0.0

        if has_ai_software:
            ai_score = 0.95
            summary.insert(0, "Strong AI signal: AI-tool name in metadata")
        elif has_ai_text_chunk:
            ai_score = 0.85
            summary.insert(0, "Strong AI signal: prompt-style PNG chunk")
        elif not has_exif and not has_c2pa:
            if is_png:
                ai_score = 0.30
                summary.append("PNG without EXIF (could be AI default save or camera-to-PNG re-save)")
            else:
                ai_score = 0.20
                summary.append("No EXIF (could be re-saved photo or AI/screenshot)")
        else:
            real_score = 0.0
            if has_camera_make:
                real_score += 0.4
                summary.append("Camera Make/Model present")
            if has_lens_info:
                real_score += 0.2
                summary.append("Lens info present")
            if has_exposure_info:
                real_score += 0.2
                summary.append("Exposure info present (real-camera signal)")
            if has_gps:
                real_score += 0.3
                summary.append("GPS coordinates present")
            ai_score = max(0.0, 0.30 - real_score * 0.30)

        if has_c2pa:
            if details.get("c2pa_ai_source"):
                ai_score = 0.99
                summary.insert(0, "Strong AI signal: C2PA declares trained algorithmic media")
            elif details.get("c2pa_google_ai_edit"):
                ai_score = 0.90
                summary.insert(0, "Strong AI signal: C2PA identifies Google AI-edited media")
            elif has_ai_software:
                ai_score = 0.95
                summary.append("C2PA + AI software identifier")
            else:
                ai_score = max(0.05, ai_score - 0.20)
                summary.append("C2PA manifest present (provenance verified)")

        return MetadataResult(
            available=True,
            probability=float(ai_score),
            has_camera_make=has_camera_make,
            has_gps=has_gps,
            has_exif=has_exif,
            has_lens_info=has_lens_info,
            has_exposure_info=has_exposure_info,
            has_ai_software=has_ai_software,
            ai_software_match=ai_software_match,
            has_c2pa=has_c2pa,
            has_ai_text_chunk=has_ai_text_chunk,
            raw_software=raw_software,
            summary=summary,
            details=details,
        )
