"""ELA — Error Level Analysis.

Re-save the input as JPEG at quality 95, compute |original - recompressed|,
amplify, and analyze. Classic Krawetz (2007) method, with 2024-2025
calibration:

  - Quality 95 is the literature standard (Krawetz 2007; Warif & Idris eval 2018)
  - Amplification factor 12-15x for visualization
  - Score features:
      mean_error          mean of error map
      max_error           max single-pixel error
      std_error           spatial variance of errors
      hot_region_count    number of connected high-error regions
      hot_region_area     fraction of pixels above 90th percentile
  - Score combines mean + isolated-hot-region count. Real photos have
    spatially uniform compression error; manipulated regions show
    isolated bright spots.

Output includes a heatmap (PIL.Image) for UI overlay display.

References:
  - Krawetz, "A Picture's Worth..." (Black Hat 2007) — original method
  - Warif & Idris, "An evaluation of Error Level Analysis" (2018) — calibration study
  - "ELA-Enhanced Dual-Branch Deep Learning" (2025) — modern usage
  - "Detection of Image Tampering Using Deep Learning, Error Levels and Noise
    Residuals" (Springer NPL 2024) — combined approach
"""

import io
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image


JPEG_QUALITY = 95
AMPLIFICATION = 12.0


@dataclass
class ELAResult:
    available: bool
    probability: float
    mean_error: float
    max_error: float
    std_error: float
    hot_region_count: int
    hot_region_area_frac: float
    heatmap_png_b64: Optional[str] = None
    error: Optional[str] = None
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "probability": self.probability,
            "mean_error": self.mean_error,
            "max_error": self.max_error,
            "std_error": self.std_error,
            "hot_region_count": self.hot_region_count,
            "hot_region_area_frac": self.hot_region_area_frac,
            "heatmap_png_b64": self.heatmap_png_b64,
            "error": self.error,
        }


def _to_pil_rgb(image_path: str) -> Image.Image:
    img = Image.open(image_path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def _ela_residual(pil_img: Image.Image) -> np.ndarray:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    a = np.asarray(pil_img, dtype=np.float32)
    b = np.asarray(recompressed, dtype=np.float32)
    diff = np.abs(a - b)
    return diff


class ELAPathway:
    def __init__(self, target_size: int = 1024, amplification: float = AMPLIFICATION):
        self.target_size = target_size
        self.amplification = amplification

    def detect(self, image_path: str) -> ELAResult:
        try:
            pil_img = _to_pil_rgb(image_path)
        except Exception as e:
            return ELAResult(
                available=False, probability=0.0,
                mean_error=0, max_error=0, std_error=0,
                hot_region_count=0, hot_region_area_frac=0.0,
                error=f"Could not load: {e}",
            )

        w, h = pil_img.size
        if max(w, h) > self.target_size:
            scale = self.target_size / max(w, h)
            pil_img = pil_img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.LANCZOS,
            )

        try:
            diff = _ela_residual(pil_img)
        except Exception as e:
            return ELAResult(
                available=False, probability=0.0,
                mean_error=0, max_error=0, std_error=0,
                hot_region_count=0, hot_region_area_frac=0.0,
                error=f"ELA failed: {e}",
            )

        amplified = np.clip(diff * self.amplification, 0, 255).astype(np.uint8)

        gray_diff = np.mean(amplified, axis=2)
        mean_error = float(gray_diff.mean())
        max_error = float(gray_diff.max())
        std_error = float(gray_diff.std())

        threshold = np.percentile(gray_diff, 95)
        hot = (gray_diff > threshold).astype(np.uint8)
        hot_region_area_frac = float(hot.mean())

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hot = cv2.morphologyEx(hot * 255, cv2.MORPH_OPEN, kernel)
        n_components, _, stats, _ = cv2.connectedComponentsWithStats((hot > 0).astype(np.uint8))
        big = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] > 50]
        hot_region_count = len(big)

        mean_score = float(np.clip((mean_error - 6.0) / 8.0, 0, 1))
        std_score = float(np.clip((std_error - 12.0) / 10.0, 0, 1))
        cluster_score = float(np.clip((hot_region_count - 3) / 12.0, 0, 1))

        probability = float(np.clip(
            0.40 * mean_score + 0.30 * std_score + 0.30 * cluster_score,
            0, 1,
        ))

        heatmap_b64 = self._encode_heatmap(amplified)

        return ELAResult(
            available=True,
            probability=probability,
            mean_error=mean_error,
            max_error=max_error,
            std_error=std_error,
            hot_region_count=hot_region_count,
            hot_region_area_frac=hot_region_area_frac,
            heatmap_png_b64=heatmap_b64,
            details={
                "mean_score": mean_score,
                "std_score": std_score,
                "cluster_score": cluster_score,
            },
        )

    @staticmethod
    def _encode_heatmap(amplified: np.ndarray) -> str:
        import base64
        gray = np.mean(amplified, axis=2).astype(np.uint8)
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        ok, buf = cv2.imencode(".png", colored)
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")
