"""Noise residual analysis (initial noise) — block-wise inconsistency detection.

Real photos have spatially uniform sensor noise (PRNU is constant across the
sensor). When a region is spliced/inpainted from a different source — or
synthesized by an AI model — the noise statistics in that region differ from
the rest of the image.

Method (Pan & Lyu, ICCP 2012 — refined with 2024 wavelet improvements):
  1. Convert to grayscale
  2. Extract noise residual via wavelet (db8, level 3) high-pass band MAD
     soft-thresholding
  3. Split into 32x32 blocks
  4. Per-block compute:
        std       — local noise std deviation
        kurtosis  — distribution shape (heavy tails = real sensor noise)
  5. Aggregate inter-block:
        std_var      — variance of stds across blocks
        kurt_var     — variance of kurtosis across blocks
        outlier_frac — fraction of blocks > 2 sigma from median
  6. Score from how non-uniform the noise distribution is

Fully synthesized images often score HIGH because their noise is
artificially uniform (zero-variance), but in a different way than real
photos — kurtosis tends near 0 (gaussian) rather than the heavier-tailed
distribution of sensor noise.

References:
  - Pan & Lyu, "Exposing Image Splicing with Inconsistent Local Noise
    Variances" (ICCP 2012)
  - "Forgery Detection in Digital Images by Multi-Scale Noise Estimation"
    (Sensors 2021)
  - "An improved PRNU noise extraction model for highly compressed image
    blocks with low resolutions" (Multimedia Tools 2024)
"""

import base64
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np
import pywt
from scipy.stats import kurtosis as compute_kurtosis


BLOCK_SIZE = 32


@dataclass
class NoiseResidualResult:
    available: bool
    probability: float
    std_variance: float
    kurt_variance: float
    outlier_block_frac: float
    avg_block_std: float
    avg_block_kurt: float
    heatmap_png_b64: Optional[str] = None
    error: Optional[str] = None
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "probability": self.probability,
            "std_variance": self.std_variance,
            "kurt_variance": self.kurt_variance,
            "outlier_block_frac": self.outlier_block_frac,
            "avg_block_std": self.avg_block_std,
            "avg_block_kurt": self.avg_block_kurt,
            "heatmap_png_b64": self.heatmap_png_b64,
            "error": self.error,
        }


def _wavelet_residual(gray: np.ndarray) -> np.ndarray:
    coeffs = pywt.wavedec2(gray.astype(np.float32), "db8", level=3)
    detail = coeffs[-1][0]
    sigma = np.median(np.abs(detail)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(gray.size + 1))
    new_coeffs = [coeffs[0]]
    for d in coeffs[1:]:
        new_coeffs.append(tuple(_soft_threshold(c, threshold) for c in d))
    denoised = pywt.waverec2(new_coeffs, "db8")
    denoised = denoised[: gray.shape[0], : gray.shape[1]]
    denoised = np.nan_to_num(denoised, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(gray.astype(np.float32) - denoised, nan=0.0, posinf=0.0, neginf=0.0)


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


class NoiseResidualPathway:
    def __init__(self, target_size: int = 1024, block_size: int = BLOCK_SIZE):
        self.target_size = target_size
        self.block_size = block_size

    def detect(self, image_path: str) -> NoiseResidualResult:
        img = cv2.imread(str(image_path))
        if img is None:
            return NoiseResidualResult(
                available=False, probability=0.0,
                std_variance=0, kurt_variance=0, outlier_block_frac=0,
                avg_block_std=0, avg_block_kurt=0,
                error=f"Could not load: {image_path}",
            )
        return self.detect_from_array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def detect_from_array(self, img_rgb: np.ndarray) -> NoiseResidualResult:
        h, w = img_rgb.shape[:2]
        if max(h, w) > self.target_size:
            scale = self.target_size / max(h, w)
            img_rgb = cv2.resize(
                img_rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        residual = _wavelet_residual(gray)

        bs = self.block_size
        nh, nw = residual.shape[0] // bs, residual.shape[1] // bs
        if nh < 4 or nw < 4:
            return NoiseResidualResult(
                available=False, probability=0.0,
                std_variance=0, kurt_variance=0, outlier_block_frac=0,
                avg_block_std=0, avg_block_kurt=0,
                error="image too small for block analysis",
            )

        std_grid = np.zeros((nh, nw), dtype=np.float32)
        kurt_grid = np.zeros((nh, nw), dtype=np.float32)

        for i in range(nh):
            for j in range(nw):
                block = residual[i * bs:(i + 1) * bs, j * bs:(j + 1) * bs].ravel()
                std_grid[i, j] = float(block.std())
                kurt = float(compute_kurtosis(block))
                kurt_grid[i, j] = kurt if np.isfinite(kurt) else 0.0

        std_flat = std_grid.ravel()
        kurt_flat = kurt_grid.ravel()
        std_median = float(np.median(std_flat))
        std_mad = float(np.median(np.abs(std_flat - std_median)) * 1.4826)

        std_variance = float(np.nan_to_num(std_flat.std()))
        kurt_variance = float(np.nan_to_num(kurt_flat.std()))
        avg_block_std = float(np.nan_to_num(std_flat.mean()))
        avg_block_kurt = float(np.nan_to_num(kurt_flat.mean()))

        if std_mad > 1e-8:
            z = np.abs(std_flat - std_median) / std_mad
            outlier_frac = float((z > 2.0).mean())
        else:
            outlier_frac = 0.0

        std_score = float(np.clip((std_variance / max(avg_block_std, 1e-6) - 0.30) / 0.30, 0, 1))
        kurt_score = float(np.clip((kurt_variance - 1.5) / 2.0, 0, 1))
        outlier_score = float(np.clip((outlier_frac - 0.05) / 0.10, 0, 1))

        probability = float(np.clip(
            0.40 * std_score + 0.30 * kurt_score + 0.30 * outlier_score,
            0, 1,
        ))

        heatmap_b64 = self._encode_heatmap(std_grid)

        return NoiseResidualResult(
            available=True,
            probability=probability,
            std_variance=std_variance,
            kurt_variance=kurt_variance,
            outlier_block_frac=outlier_frac,
            avg_block_std=avg_block_std,
            avg_block_kurt=avg_block_kurt,
            heatmap_png_b64=heatmap_b64,
            details={
                "std_score": std_score,
                "kurt_score": kurt_score,
                "outlier_score": outlier_score,
                "block_grid": (int(nh), int(nw)),
            },
        )

    @staticmethod
    def _encode_heatmap(std_grid: np.ndarray) -> str:
        std_grid = np.nan_to_num(std_grid, nan=0.0, posinf=0.0, neginf=0.0)
        if std_grid.max() <= 0:
            return ""
        norm = ((std_grid - std_grid.min()) / max(std_grid.max() - std_grid.min(), 1e-8) * 255).astype(np.uint8)
        big = cv2.resize(norm, (norm.shape[1] * 32, norm.shape[0] * 32), interpolation=cv2.INTER_NEAREST)
        colored = cv2.applyColorMap(big, cv2.COLORMAP_VIRIDIS)
        ok, buf = cv2.imencode(".png", colored)
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")
