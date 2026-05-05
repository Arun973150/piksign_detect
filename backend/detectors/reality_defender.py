"""Reality Defender external classifier (top-weighted pathway, 0.60).

Production-ready wrapper:
  - Async SDK in synchronous threaded interface
  - Image preprocessing (resize >2048, re-encode to JPEG q92)
  - Retry with backoff on transient errors
  - None-safe response parsing
  - Auto-loads REALITY_DEFENDER_API_KEY from piksign_detect/.env
"""

import asyncio
import concurrent.futures
import os
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from backend.config import load_app_env

load_app_env()

try:
    from realitydefender import RealityDefender
    HAS_PROVIDER = True
except ImportError:
    HAS_PROVIDER = False

try:
    from PIL import Image, ImageOps
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


MAX_UPLOAD_DIM = 2048
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
NORMALIZE_EXTENSIONS = {".png", ".tif", ".tiff", ".webp", ".bmp", ".heic", ".heif"}
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 1.5
DETECTION_TIMEOUT = 120


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


def _should_try_next_key(err: Exception) -> bool:
    msg = str(err).lower()
    return any(m in msg for m in (
        "unauthorized", "forbidden", "401", "403",
        "rate", "quota", "limit", "usage", "payment", "billing",
    ))


@dataclass
class RealityDefenderResult:
    available: bool
    status: str
    probability: float
    provider_status: str
    models: list = field(default_factory=list)
    attempts: int = 0
    preprocessed: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "status": self.status,
            "probability": self.probability,
            "provider_status": self.provider_status,
            "models": self.models,
            "attempts": self.attempts,
            "preprocessed": self.preprocessed,
            "error": self.error,
        }


def _safe_prob(x) -> float:
    if x is None:
        return 0.0
    try:
        return float(min(max(float(x), 0.0), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _is_transient(err: Exception) -> bool:
    msg = str(err).lower()
    return any(m in msg for m in (
        "timeout", "timed out", "rate", "throttl",
        "503", "502", "504", "500", "connection", "reset", "ssl",
        "unavailable",
    ))


def _prepare_upload(image_path: str) -> Tuple[str, Optional[str]]:
    p = Path(image_path)
    if not HAS_PIL:
        return image_path, None
    try:
        size_bytes = p.stat().st_size
    except OSError:
        return image_path, None

    ext = p.suffix.lower()
    if size_bytes <= MAX_UPLOAD_BYTES and ext not in NORMALIZE_EXTENSIONS:
        try:
            with Image.open(image_path) as im:
                if max(im.size) <= MAX_UPLOAD_DIM:
                    return image_path, None
        except Exception:
            return image_path, None

    try:
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode == "RGBA":
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
            elif im.mode != "RGB":
                im = im.convert("RGB")

            w, h = im.size
            if max(w, h) > MAX_UPLOAD_DIM:
                scale = MAX_UPLOAD_DIM / max(w, h)
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", prefix="rd_", delete=False)
            tmp.close()
            im.save(tmp.name, format="JPEG", quality=92, optimize=True)
            return tmp.name, tmp.name
    except Exception:
        return image_path, None


class RealityDefenderPathway:
    _key_index = 0
    _key_lock = threading.Lock()

    def __init__(self, api_key: Optional[str] = None, verbose: bool = False):
        self.api_keys = _split_keys(
            api_key,
            os.environ.get("REALITY_DEFENDER_API_KEYS"),
            os.environ.get("REALITY_DEFENDER_API_KEY"),
        )
        self.api_key = self.api_keys[0] if self.api_keys else ""
        self.verbose = verbose
        self._client = None
        self.error: Optional[str] = None

        if not HAS_PROVIDER:
            self.error = "realitydefender SDK not installed"
            return
        if not self.api_keys:
            self.error = "REALITY_DEFENDER_API_KEY/REALITY_DEFENDER_API_KEYS not set"
            return
        try:
            self._client = RealityDefender(api_key=self.api_key)
        except Exception as e:
            self.error = f"Init failed: {e}"
            if self.verbose:
                traceback.print_exc()

    @property
    def available(self) -> bool:
        return self._client is not None

    def detect(self, image_path: str) -> RealityDefenderResult:
        if not self.available:
            return RealityDefenderResult(
                available=False, status="unavailable", probability=0.0,
                provider_status="unavailable", error=self.error,
            )

        upload_path, temp_path = _prepare_upload(image_path)
        last_err: Optional[Exception] = None
        attempts_used = 0

        try:
            for key in self._ordered_keys():
                for attempt in range(1, RETRY_ATTEMPTS + 1):
                    attempts_used += 1
                    try:
                        raw = self._run(upload_path, key)
                        result = self._parse(raw)
                        result.attempts = attempts_used
                        result.preprocessed = temp_path is not None
                        if result.status == "no_score" and attempt < RETRY_ATTEMPTS:
                            time.sleep(RETRY_BACKOFF * attempt)
                            continue
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
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

        return RealityDefenderResult(
            available=True, status="error", probability=0.0,
            provider_status="error", attempts=attempts_used,
            preprocessed=temp_path is not None,
            error=str(last_err) if last_err else "unknown",
        )

    def _ordered_keys(self) -> list[str]:
        if not self.api_keys:
            return []
        with self._key_lock:
            start = self.__class__._key_index % len(self.api_keys)
            self.__class__._key_index += 1
        return self.api_keys[start:] + self.api_keys[:start]

    def _run(self, image_path: str, api_key: str) -> Dict[str, Any]:
        def _in_thread():
            import sys as _sys
            _sys.stdout = _sys.__stdout__
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._async_call(api_key, image_path))
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_in_thread).result(timeout=DETECTION_TIMEOUT)

    @staticmethod
    async def _async_call(api_key: str, image_path: str) -> Dict[str, Any]:
        client = RealityDefender(api_key=api_key)
        upload = await client.upload(file_path=image_path)
        return await client.get_result(upload["request_id"])

    def _parse(self, raw: Dict[str, Any]) -> RealityDefenderResult:
        provider_status = raw.get("status") or "UNKNOWN"
        score = raw.get("score")
        score_missing = score is None

        models = [
            {
                "name": m.get("name", "unknown"),
                "status": m.get("status", "unknown"),
                "score": _safe_prob(m.get("score")),
            }
            for m in (raw.get("models") or [])
        ]

        prob = _safe_prob(score)
        if provider_status == "FAKE":
            prob = max(prob, 0.6)
        elif provider_status == "MANIPULATED":
            prob = max(prob, 0.7)
        elif provider_status == "SUSPICIOUS":
            prob = max(prob, 0.4)
        elif provider_status == "AUTHENTIC":
            prob = min(prob, 0.4)
        elif provider_status == "NOT_APPLICABLE":
            prob = 0.0
        elif score_missing and provider_status in ("UNKNOWN", "ANALYZING", "PROCESSING"):
            return RealityDefenderResult(
                available=True, status="no_score", probability=0.0,
                provider_status=provider_status, models=models,
            )

        return RealityDefenderResult(
            available=True, status="success", probability=prob,
            provider_status=provider_status, models=models,
        )
