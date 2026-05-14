"""FastAPI routes - upload, detection, result rendering."""

import base64
import io
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageFile, UnidentifiedImageError

from backend.config import load_app_env
from backend.orchestrator import PikSignDetector
from frontend.api.schemas import DetectionResponse, HealthResponse, PathwayScore

load_app_env()

MAX_UPLOAD_BYTES = int(os.environ.get("PIKSIGN_MAX_UPLOAD_MB", "20")) * 1024 * 1024
MAX_IMAGE_PIXELS = int(os.environ.get("PIKSIGN_MAX_IMAGE_PIXELS", "25000000"))
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = False
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
ACCESS_TOKEN = os.environ.get("PIKSIGN_ACCESS_TOKEN", "").strip()
ECHO_UPLOAD = os.environ.get("PIKSIGN_ECHO_UPLOAD", "false").lower() in {"1", "true", "yes", "on"}
DISPLAY_NAMES = {
    "ai_check": "AI Check",
    "bitmind_api": "BitMind API",
    "local_ensemble": "Local Ensemble",
    "ensemble": "AI Check",
    "synthid": "Authenticity Signal",
    "ela": "Forensic Layer A",
    "noise_residual": "Forensic Layer B",
    "metadata": "File Integrity",
    "text_analysis": "Text Quality",
}

router = APIRouter()
_templates_dir = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))

_detector: Optional[PikSignDetector] = None
_detection_lock = threading.Lock()


def get_detector() -> PikSignDetector:
    global _detector
    if _detector is None:
        _detector = PikSignDetector()
    return _detector


@router.get("/healthz", response_model=HealthResponse)
async def healthz():
    return HealthResponse()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"access_token": request.query_params.get("token", "")},
    )


@router.post("/api/detect")
async def api_detect(request: Request, file: UploadFile = File(...)):
    _require_access(
        request.headers.get("x-piksign-token")
        or request.query_params.get("token")
    )
    suffix, contents = await _validate(file)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="piksign_")
    tmp.write(contents)
    tmp.close()
    try:
        report = _detect_one_at_a_time(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return JSONResponse(_json_safe(_to_response(report).model_dump(exclude_none=True)))


@router.post("/detect", response_class=HTMLResponse)
async def html_detect(
    request: Request,
    file: UploadFile = File(...),
    access_token: str = Form(default=""),
):
    _require_access(access_token)
    suffix, contents = await _validate(file)
    image_data_url = _image_data_url(contents, suffix) if ECHO_UPLOAD else None
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="piksign_")
    tmp.write(contents)
    tmp.close()
    try:
        report = _detect_one_at_a_time(tmp.name)
        response = _to_response(report, image_data_url=image_data_url)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "filename": file.filename,
            "response": response,
        },
    )


def _detect_one_at_a_time(image_path: str):
    if not _detection_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Another image is currently being analysed. Please wait and try again.",
        )
    try:
        return get_detector().detect(image_path)
    finally:
        _detection_lock.release()


def _require_access(token: Optional[str]):
    if not ACCESS_TOKEN:
        return
    token = token or ""
    if not token:
        raise HTTPException(status_code=401, detail="Access token required")
    if token != ACCESS_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid access token")


async def _validate(file: UploadFile):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format {suffix!r}. Allowed: {sorted(ALLOWED_EXTENSIONS)}")
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES // 1024 // 1024} MB)")
    if not contents:
        raise HTTPException(400, "Empty upload")
    try:
        with Image.open(io.BytesIO(contents)) as img:
            img.verify()
            width, height = img.size
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "Image dimensions are too large")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "Uploaded file is not a valid image")
    return suffix, contents


def _image_data_url(contents: bytes, suffix: str) -> Optional[str]:
    mime_by_ext = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    mime = mime_by_ext.get(suffix)
    if not mime:
        return None
    return f"data:{mime};base64,{base64.b64encode(contents).decode('ascii')}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def _to_response(report, image_data_url: Optional[str] = None) -> DetectionResponse:
    pathways = []

    def add(name, obj, summary_fn=None):
        display_name = DISPLAY_NAMES.get(name, name)
        if obj is None:
            pathways.append(PathwayScore(name=display_name, available=False, probability=0.0))
            return
        d = obj.to_dict()
        pathways.append(PathwayScore(
            name=display_name,
            available=bool(d.get("available", True)),
            probability=float(d.get("probability", 0.0)),
            summary=summary_fn(d) if summary_fn else None,
            details=None,
        ))

    add("ai_check", report.ai_check,
        lambda d: _ai_check_summary(d))
    add("bitmind_api", report.bitmind_api,
        lambda d: _bitmind_summary(d))
    add("local_ensemble", report.ensemble,
        lambda d: _ensemble_summary(d))
    add("synthid", report.synthid,
        lambda d: _auth_signal_summary(d))
    add("ela", report.ela,
        lambda d: f"visual consistency score {d.get('probability', 0.0):.2f}")
    add("noise_residual", report.noise_residual,
        lambda d: f"texture consistency score {d.get('probability', 0.0):.2f}")
    add("metadata", report.metadata,
        lambda d: "file provenance signals checked")
    add("text_analysis", report.text_analysis,
        lambda d: _text_summary(d))

    ela_heatmap = report.ela.heatmap_png_b64 if (report.ela and report.ela.available) else None
    noise_heatmap = report.noise_residual.heatmap_png_b64 if (report.noise_residual and report.noise_residual.available) else None

    return DetectionResponse(
        verdict=report.verdict,
        probability=report.probability,
        elapsed_seconds=report.elapsed_seconds,
        pathways=pathways,
        fusion=_public_fusion(report.fusion) if report.fusion else None,
        synthid=None,
        dimension=report.dimension.to_dict() if report.dimension else None,
        metadata_summary=_public_metadata_summary(report.metadata),
        layer_a_heatmap_b64=ela_heatmap,
        layer_b_heatmap_b64=noise_heatmap,
        image_data_url=image_data_url,
        notes=report.notes,
    )


def _public_fusion(fusion) -> dict:
    raw = fusion.to_dict()
    weights = raw.get("weights_used") or {}
    contributions = raw.get("contributions") or {}
    return {
        "verdict": raw.get("verdict"),
        "probability": raw.get("probability"),
        "weights_used": {
            DISPLAY_NAMES.get(name, name): value
            for name, value in weights.items()
        },
        "contributions": {
            DISPLAY_NAMES.get(name, name): value
            for name, value in contributions.items()
        },
        "authenticity_bonus": raw.get("synthid_bonus", 0.0),
        "reasons": [
            reason
            .replace("SynthID", "Authenticity signal")
            .replace("synthid", "authenticity signal")
            .replace("Authenticity signal suppressed", "Authenticity signal limited")
            .replace("real-camera provenance", "file provenance")
            .replace("Strong real-camera provenance verified", "Strong file provenance verified")
            .replace("text_analysis", "Text Quality")
            .replace("ensemble", "AI Check")
            .replace("noise_residual", "Forensic Layer B")
            .replace("ela", "Forensic Layer A")
            .replace("metadata", "File Integrity")
            for reason in (raw.get("reasons") or [])
        ],
    }


def _public_metadata_summary(metadata) -> Optional[list[str]]:
    if not metadata:
        return None
    details = metadata.details or {}
    if details.get("c2pa_ai_source"):
        notes = ["Content credentials identify AI-generated media"]
        if details.get("c2pa_generator"):
            notes.append(f"Generator: {details['c2pa_generator']}")
        if details.get("c2pa_software_agent"):
            notes.append(f"Software agent: {details['c2pa_software_agent']}")
        return notes
    if details.get("c2pa_google_ai_edit"):
        notes = ["Content credentials identify Google AI-edited media"]
        if details.get("c2pa_generator"):
            notes.append(f"Generator: {details['c2pa_generator']}")
        if details.get("c2pa_composite_source"):
            notes.append("Source type: composite / edited")
        if details.get("c2pa_software_agent"):
            notes.append(f"Software agent: {details['c2pa_software_agent']}")
        return notes
    if not metadata.summary:
        return ["File provenance checked"]
    return [
        "File provenance signals checked",
        "Source details are hidden in the public view",
    ]


def _text_summary(d: dict) -> str:
    if not d.get("has_text"):
        return "no text regions detected"
    snippets = ((d.get("details") or {}).get("snippets") or [])[:3]
    suffix = f" | {', '.join(snippets)}" if snippets else ""
    return f"text regions={d.get('n_regions', 0)} avg confidence={d.get('avg_confidence', 0.0):.2f}{suffix}"


def _ensemble_summary(d: dict) -> str:
    error = (d.get("error") or "").strip()
    if error:
        return f"ensemble error: {error[:80]}"
    n = len(d.get("loaded_models") or [])
    return f"{d.get('provider_status', '?')} ({d.get('probability', 0.0):.2f}) | {n} models"


def _ai_check_summary(d: dict) -> str:
    error = (d.get("error") or "").strip()
    source = d.get("source", "?")
    if error and not d.get("available"):
        return f"AI Check unavailable: {error[:80]}"
    return f"{d.get('provider_status', '?')} ({d.get('probability', 0.0):.2f}) | {source}"


def _bitmind_summary(d: dict) -> str:
    error = (d.get("error") or "").strip()
    if error:
        return f"API error: {error[:80]}"
    return f"{d.get('provider_status', '?')} ({d.get('probability', 0.0):.2f}) | confidence {d.get('confidence', 0.0):.2f}"


def _auth_signal_summary(d: dict) -> str:
    confidence = d.get("confidence", 0.0)
    if d.get("detected"):
        return f"{d.get('tier', 'match')} match ({confidence:.2f})"
    if confidence >= 0.60:
        return f"suspicious signal ({confidence:.2f})"
    if confidence >= 0.45:
        return f"weak signal ({confidence:.2f})"
    return f"none ({confidence:.2f})"
