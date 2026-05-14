"""BitMind Oracle API image detector.

Uses the Subnet 34 image endpoint:
  POST {BITMIND_BASE_URL}{BITMIND_ENDPOINT}

The API accepts a URL or base64 image in JSON. For local uploads we normalize
to a compact JPEG and send a data URL.
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from backend.config import load_app_env

load_app_env()

try:
    from PIL import Image, ImageOps
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


MAX_IMAGE_DIM = 1600
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024
REQUEST_TIMEOUT = 75
RETRY_ATTEMPTS = 2
RETRY_BACKOFF = 1.5


def _split_keys(*values: Optional[str]) -> list[str]:
    keys: list[str] = []
    seen = set()
    for value in values:
        for key in (value or "").replace("\n", ",").split(","):
            key = key.strip()
            if key and key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def _safe_prob(value) -> float:
    try:
        return float(min(max(float(value), 0.0), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _is_transient(err: Exception) -> bool:
    msg = str(err).lower()
    return any(token in msg for token in (
        "timeout", "timed out", "rate", "throttl", "429",
        "500", "502", "503", "504", "connection", "reset", "ssl",
        "temporarily", "unavailable",
    ))


def _should_try_next_key(err: Exception) -> bool:
    msg = str(err).lower()
    return any(token in msg for token in (
        "401", "403", "unauthorized", "forbidden",
        "quota", "billing", "payment", "rate", "limit", "429",
    ))


def _image_data_url(image_path: str) -> tuple[str, bool]:
    if not HAS_PIL:
        raw = Path(image_path).read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii"), False

    with Image.open(image_path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode == "RGBA":
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")

        if max(im.size) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(im.size)
            im = im.resize(
                (max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                Image.LANCZOS,
            )

        quality = 92
        while True:
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
            raw = buf.getvalue()
            if len(raw) <= MAX_PAYLOAD_BYTES or quality <= 55:
                break
            quality -= 8

    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii"), True


@dataclass
class BitMindResult:
    available: bool
    status: str
    probability: float
    provider_status: str
    confidence: float = 0.0
    similarity: float = 0.0
    attempts: int = 0
    preprocessed: bool = False
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "status": self.status,
            "probability": self.probability,
            "provider_status": self.provider_status,
            "confidence": self.confidence,
            "similarity": self.similarity,
            "attempts": self.attempts,
            "preprocessed": self.preprocessed,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
            "details": self.details,
        }


class BitMindPathway:
    _key_index = 0
    _key_lock = threading.Lock()

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        endpoint: Optional[str] = None,
        application: Optional[str] = None,
        rich: bool = True,
        verbose: bool = False,
    ):
        if api_key:
            self.api_keys = _split_keys(api_key)
        else:
            self.api_keys = _split_keys(os.environ.get("BITMIND_API_KEY"), os.environ.get("BITMIND_API_KEYS"))
        self.base_url = (base_url or os.environ.get("BITMIND_BASE_URL") or "https://api.bitmind.ai/oracle/v1").rstrip("/")
        self.endpoint = endpoint or os.environ.get("BITMIND_ENDPOINT") or "/34/detect-image"
        self.application = application or os.environ.get("BITMIND_APPLICATION") or "oracle-api"
        self.rich = rich
        self.verbose = verbose
        self.error: Optional[str] = None

        if not self.api_keys:
            self.error = "BITMIND_API_KEY/BITMIND_API_KEYS not set"
        if not self.endpoint.startswith("/"):
            self.endpoint = "/" + self.endpoint

    @property
    def available(self) -> bool:
        return bool(self.api_keys) and self.error is None

    def detect(self, image_path: str) -> BitMindResult:
        if not self.available:
            return BitMindResult(
                available=False,
                status="unavailable",
                probability=0.0,
                provider_status="unavailable",
                error=self.error,
            )

        t0 = time.time()
        try:
            image, preprocessed = _image_data_url(image_path)
        except Exception as e:
            return BitMindResult(
                available=True,
                status="error",
                probability=0.0,
                provider_status="image_prepare_failed",
                error=str(e),
                elapsed_seconds=time.time() - t0,
            )

        last_err: Optional[Exception] = None
        attempts = 0
        for key in self._ordered_keys():
            for attempt in range(1, RETRY_ATTEMPTS + 1):
                attempts += 1
                try:
                    raw = self._call(key, image)
                    result = self._parse(raw)
                    result.attempts = attempts
                    result.preprocessed = preprocessed
                    result.elapsed_seconds = time.time() - t0
                    return result
                except Exception as e:
                    last_err = e
                    if self.verbose:
                        traceback.print_exc()
                    if _should_try_next_key(e):
                        break
                    if attempt < RETRY_ATTEMPTS and _is_transient(e):
                        time.sleep(RETRY_BACKOFF * attempt)
                        continue
                    break
            if last_err and _should_try_next_key(last_err):
                continue
            if last_err and not _is_transient(last_err):
                break

        return BitMindResult(
            available=True,
            status="error",
            probability=0.0,
            provider_status="error",
            attempts=attempts,
            preprocessed=preprocessed,
            elapsed_seconds=time.time() - t0,
            error=str(last_err) if last_err else "unknown",
        )

    def _ordered_keys(self) -> list[str]:
        with self._key_lock:
            start = self.__class__._key_index % len(self.api_keys)
            self.__class__._key_index += 1
        return self.api_keys[start:] + self.api_keys[:start]

    def _call(self, api_key: str, image: str) -> Dict[str, Any]:
        url = self.base_url + self.endpoint
        headers = {
            "Authorization": f"Bearer {api_key}",
            "x-bitmind-application": self.application,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {"image": image, "rich": self.rich, "source": "piksign_detect"}
        response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        if response.status_code >= 400:
            raise RuntimeError(f"BitMind API HTTP {response.status_code}: {response.text[:300]}")
        return response.json()

    def _parse(self, raw: Dict[str, Any]) -> BitMindResult:
        is_ai = raw.get("isAI")
        confidence = _safe_prob(raw.get("confidence"))
        similarity = _safe_prob(raw.get("similarity"))

        if isinstance(is_ai, bool):
            probability = confidence if is_ai else 1.0 - confidence
            provider_status = "AI" if is_ai else "AUTHENTIC"
        else:
            probability = confidence
            provider_status = "AI" if probability >= 0.5 else "AUTHENTIC"

        details = {
            "fqdn": raw.get("fqdn"),
            "processingTime": raw.get("processingTime"),
        }
        if "contentCredentials" in raw:
            details["contentCredentials"] = raw.get("contentCredentials")

        return BitMindResult(
            available=True,
            status="success",
            probability=_safe_prob(probability),
            provider_status=provider_status,
            confidence=confidence,
            similarity=similarity,
            details={k: v for k, v in details.items() if v is not None},
        )
