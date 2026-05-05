"""Dimension + aspect-ratio + crop detection (informational sidecar).

Three checks, displayed alongside the verdict but NOT in the fusion score:

  1. Exact-dimension match against known AI generator output sizes
       (1024x1024 = SDXL/SD/MJ/DALL-E, 864x1184 = Nano-Banana, etc.)

  2. Aspect-ratio match against known AI ratios (1:1, 3:4, 9:16, 16:9, etc.)
       vs typical camera sensor ratios (3:2, 4:3, 16:9)

  3. Crop detection — if the dimensions don't match either AI defaults OR
       common sensor sizes, the image was likely cropped or resized after
       capture/generation.

Used to provide context to the user. Not weighted into the AI/REAL verdict
because legitimate photos can have unusual dimensions.
"""

from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


KNOWN_AI_DIMENSIONS: Dict[Tuple[int, int], List[str]] = {
    (1024, 1024): ["SDXL", "SD 2.x", "Midjourney", "DALL-E 3", "Imagen", "Flux", "Nano-Banana"],
    (512, 512):   ["SD 1.5", "SD 2.0"],
    (768, 768):   ["SD 2.x"],
    (1152, 896):  ["SDXL"],
    (896, 1152):  ["SDXL portrait"],
    (1216, 832):  ["SDXL"],
    (832, 1216):  ["SDXL portrait"],
    (1344, 768):  ["SDXL", "Nano-Banana"],
    (768, 1344):  ["SDXL portrait", "Nano-Banana portrait"],
    (1408, 768):  ["Flux", "SDXL"],
    (768, 1408):  ["Flux portrait", "SDXL portrait"],
    (1456, 816):  ["Midjourney v6"],
    (816, 1456):  ["Midjourney v6 portrait"],
    (1024, 1792): ["DALL-E 3"],
    (1792, 1024): ["DALL-E 3"],
    (1024, 1408): ["Flux"],
    (1408, 1024): ["Flux"],
    (864, 1184):  ["Nano-Banana"],
    (1184, 864):  ["Nano-Banana"],
    (1152, 1536): ["Nano-Banana Pro"],
    (1536, 1152): ["Nano-Banana Pro"],
    (1760, 2432): ["Nano-Banana Pro"],
    (2432, 1760): ["Nano-Banana Pro"],
    (2048, 2048): ["Nano-Banana Pro", "Imagen 3"],
    (1024, 2048): ["Imagen 3"],
    (2048, 1024): ["Imagen 3"],
    (896, 1280):  ["Imagen 3"],
    (1280, 896):  ["Imagen 3"],
}


COMMON_SENSOR_DIMENSIONS = {
    (4032, 3024), (3024, 4032),
    (4000, 3000), (3000, 4000),
    (6000, 4000), (4000, 6000),
    (5472, 3648), (3648, 5472),
    (3840, 2160), (2160, 3840),
    (1920, 1080), (1080, 1920),
    (4608, 3456), (3456, 4608),
    (4096, 3072), (3072, 4096),
    (2592, 1944), (1944, 2592),
    (3264, 2448), (2448, 3264),
}


COMMON_AI_ASPECTS = {
    (1, 1):   "1:1 square",
    (3, 4):   "3:4 portrait",
    (4, 3):   "4:3 landscape",
    (9, 16):  "9:16 portrait",
    (16, 9):  "16:9 landscape",
    (2, 3):   "2:3 portrait",
    (3, 2):   "3:2 landscape",
    (7, 9):   "~3:4 (Nano-Banana)",
    (9, 7):   "~4:3 (Nano-Banana)",
}


COMMON_CAMERA_ASPECTS = {
    (3, 2):   "DSLR/mirrorless 3:2",
    (4, 3):   "phone/compact 4:3",
    (16, 9):  "video 16:9",
    (1, 1):   "square crop",
    (5, 4):   "medium-format",
}


@dataclass
class DimensionResult:
    width: int
    height: int
    aspect_ratio: float
    aspect_string: str
    matches_ai_size: bool
    matched_ai_models: List[str]
    matches_camera_size: bool
    matches_ai_aspect: bool
    matches_camera_aspect: bool
    likely_cropped: bool
    interpretation: str
    error: Optional[str] = None
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "aspect_string": self.aspect_string,
            "matches_ai_size": self.matches_ai_size,
            "matched_ai_models": self.matched_ai_models,
            "matches_camera_size": self.matches_camera_size,
            "matches_ai_aspect": self.matches_ai_aspect,
            "matches_camera_aspect": self.matches_camera_aspect,
            "likely_cropped": self.likely_cropped,
            "interpretation": self.interpretation,
            "error": self.error,
        }


def _simplify_ratio(w: int, h: int, max_denom: int = 16) -> Tuple[int, int]:
    g = gcd(w, h)
    sw, sh = w // g, h // g
    if max(sw, sh) > max_denom:
        scale = max(sw, sh) / max_denom
        return (max(1, round(sw / scale)), max(1, round(sh / scale)))
    return (sw, sh)


def _aspect_match(target: float, candidates: dict, tol: float = 0.02) -> Optional[str]:
    for (a, b), name in candidates.items():
        ratio = a / b
        if abs(target - ratio) / ratio <= tol:
            return name
    return None


class DimensionPathway:
    """Aspect + crop detection. Informational only."""

    def detect(self, image_path: str) -> DimensionResult:
        try:
            with Image.open(image_path) as img:
                w, h = img.size
        except Exception as e:
            return DimensionResult(
                width=0, height=0, aspect_ratio=0.0, aspect_string="?",
                matches_ai_size=False, matched_ai_models=[],
                matches_camera_size=False,
                matches_ai_aspect=False, matches_camera_aspect=False,
                likely_cropped=False,
                interpretation="(could not read dimensions)", error=str(e),
            )

        return self.classify(w, h)

    def classify(self, w: int, h: int) -> DimensionResult:
        aspect = w / max(h, 1)
        sw, sh = _simplify_ratio(w, h)
        aspect_str = f"{sw}:{sh}"

        ai_models = KNOWN_AI_DIMENSIONS.get((w, h), [])
        matches_ai_size = bool(ai_models)
        matches_camera_size = (w, h) in COMMON_SENSOR_DIMENSIONS

        ai_aspect_name = _aspect_match(aspect, COMMON_AI_ASPECTS)
        cam_aspect_name = _aspect_match(aspect, COMMON_CAMERA_ASPECTS)
        matches_ai_aspect = ai_aspect_name is not None
        matches_camera_aspect = cam_aspect_name is not None

        likely_cropped = (
            not matches_ai_size
            and not matches_camera_size
            and not matches_ai_aspect
            and not matches_camera_aspect
        )

        if matches_ai_size:
            interp = f"{w}x{h} matches: {', '.join(ai_models)}"
        elif matches_camera_size:
            interp = f"{w}x{h} matches typical camera sensor (real photo plausible)"
        elif likely_cropped:
            interp = f"{w}x{h} (aspect {aspect_str}) — unusual; likely cropped or resized"
        elif matches_ai_aspect and not matches_camera_aspect:
            interp = f"{w}x{h} aspect {aspect_str} ({ai_aspect_name}) — common for AI"
        elif matches_camera_aspect and not matches_ai_aspect:
            interp = f"{w}x{h} aspect {aspect_str} ({cam_aspect_name}) — typical camera"
        else:
            interp = f"{w}x{h} aspect {aspect_str} (ambiguous — common for both)"

        return DimensionResult(
            width=w,
            height=h,
            aspect_ratio=aspect,
            aspect_string=aspect_str,
            matches_ai_size=matches_ai_size,
            matched_ai_models=ai_models,
            matches_camera_size=matches_camera_size,
            matches_ai_aspect=matches_ai_aspect,
            matches_camera_aspect=matches_camera_aspect,
            likely_cropped=likely_cropped,
            interpretation=interp,
            details={
                "ai_aspect_name": ai_aspect_name,
                "camera_aspect_name": cam_aspect_name,
            },
        )
