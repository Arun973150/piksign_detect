"""Pydantic schemas for API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


class PathwayScore(BaseModel):
    name: str
    available: bool
    probability: float
    summary: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class DetectionResponse(BaseModel):
    verdict: str
    probability: float
    elapsed_seconds: float
    pathways: List[PathwayScore]
    fusion: Optional[Dict[str, Any]] = None
    synthid: Optional[Dict[str, Any]] = None
    dimension: Optional[Dict[str, Any]] = None
    metadata_summary: Optional[List[str]] = None
    layer_a_heatmap_b64: Optional[str] = None
    layer_b_heatmap_b64: Optional[str] = None
    image_data_url: Optional[str] = None
    notes: Dict[str, Any] = {}
