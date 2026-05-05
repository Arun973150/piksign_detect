"""Camera fingerprint pathway.

Probabilistic camera-pipeline check. Real camera photos often retain a weak
periodic sensor/demosaicing fingerprint. Images created directly in RGB,
heavily edited, or generated may have a weaker/inconsistent signal.

Probability:
  0.0 = camera pipeline signal looks present
  1.0 = camera pipeline signal looks absent or inconsistent
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np
import pywt


@dataclass
class CameraFingerprintResult:
    available: bool
    probability: float
    peak_ratio: float
    peak_strength: float
    error: Optional[str] = None
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "probability": self.probability,
            "peak_ratio": self.peak_ratio,
            "peak_strength": self.peak_strength,
            "error": self.error,
        }


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _wavelet_residual(channel: np.ndarray) -> np.ndarray:
    coeffs = pywt.wavedec2(channel, "db4", level=2)
    detail = coeffs[-1][0]
    sigma = np.median(np.abs(detail)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(channel.size + 1))

    new_coeffs = [coeffs[0]]
    for d in coeffs[1:]:
        new_coeffs.append(tuple(_soft_threshold(c, threshold) for c in d))

    denoised = pywt.waverec2(new_coeffs, "db4")[: channel.shape[0], : channel.shape[1]]
    denoised = np.nan_to_num(denoised, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(channel - denoised, nan=0.0, posinf=0.0, neginf=0.0)


class CameraFingerprintPathway:
    def __init__(self, target_size: int = 1024):
        self.target_size = target_size

    def detect(self, image_path: str) -> CameraFingerprintResult:
        img = cv2.imread(str(image_path))
        if img is None:
            return CameraFingerprintResult(
                available=False,
                probability=0.0,
                peak_ratio=1.0,
                peak_strength=0.0,
                error=f"Could not load: {image_path}",
            )
        return self.detect_from_array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def detect_from_array(self, img_rgb: np.ndarray) -> CameraFingerprintResult:
        h, w = img_rgb.shape[:2]
        if h < 128 or w < 128:
            return CameraFingerprintResult(
                available=False,
                probability=0.0,
                peak_ratio=1.0,
                peak_strength=0.0,
                error="image too small for camera fingerprint analysis",
            )

        if max(h, w) > self.target_size:
            scale = self.target_size / max(h, w)
            img_rgb = cv2.resize(
                img_rgb,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        img_f = img_rgb.astype(np.float32) / 255.0
        residual = _wavelet_residual(img_f[..., 1])

        rh, rw = residual.shape
        rh = (rh // 2) * 2
        rw = (rw // 2) * 2
        residual = residual[:rh, :rw]

        if rh < 128 or rw < 128:
            return CameraFingerprintResult(
                available=False,
                probability=0.0,
                peak_ratio=1.0,
                peak_strength=0.0,
                error="residual too small for camera fingerprint analysis",
            )

        mag = np.abs(np.fft.fftshift(np.fft.fft2(residual)))
        cy, cx = rh // 2, rw // 2
        spike_radius = 3
        locations = [
            (cy - rh // 4, cx - rw // 4),
            (cy - rh // 4, cx + rw // 4),
            (cy + rh // 4, cx - rw // 4),
            (cy + rh // 4, cx + rw // 4),
        ]

        spike_energy = 0.0
        for sy, sx in locations:
            y0, y1 = max(0, sy - spike_radius), min(rh, sy + spike_radius + 1)
            x0, x1 = max(0, sx - spike_radius), min(rw, sx + spike_radius + 1)
            if y1 > y0 and x1 > x0:
                spike_energy = max(spike_energy, float(mag[y0:y1, x0:x1].max()))

        mask = np.ones_like(mag, dtype=bool)
        for sy, sx in locations:
            y0, y1 = max(0, sy - spike_radius * 2), min(rh, sy + spike_radius * 2 + 1)
            x0, x1 = max(0, sx - spike_radius * 2), min(rw, sx + spike_radius * 2 + 1)
            mask[y0:y1, x0:x1] = False
        mask[max(0, cy - 5):min(rh, cy + 5), max(0, cx - 5):min(rw, cx + 5)] = False

        background = mag[mask]
        bg_mean = float(np.mean(background)) if background.size else 1.0
        peak_ratio = float(spike_energy / max(bg_mean, 1e-8))

        ai_score = float(np.clip((2.20 - peak_ratio) / 1.20, 0.0, 1.0))
        return CameraFingerprintResult(
            available=True,
            probability=ai_score,
            peak_ratio=peak_ratio,
            peak_strength=spike_energy,
            details={"background_mean": bg_mean, "spike_energy": spike_energy},
        )
