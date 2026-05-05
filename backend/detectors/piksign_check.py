"""Step 0 self-check: detect PikSign-protected images.

Lightweight version. Looks for our protection pipeline's C2PA marker
embedded in PNG tEXt or as raw bytes in the file.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from PIL import Image, PngImagePlugin


C2PA_MARKER = b"c2pa_manifest"


@dataclass
class PikSignCheckResult:
    is_protected: bool
    has_c2pa_marker: bool
    has_piksign_assertion: bool
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "is_protected": self.is_protected,
            "has_c2pa_marker": self.has_c2pa_marker,
            "has_piksign_assertion": self.has_piksign_assertion,
        }


class PikSignCheck:
    def check(self, image_path: str) -> PikSignCheckResult:
        path = Path(image_path)
        if not path.exists():
            return PikSignCheckResult(False, False, False, {"error": "file not found"})

        has_c2pa = False
        has_assertion = False
        details = {}

        try:
            with open(path, "rb") as f:
                head = f.read(min(2 * 1024 * 1024, path.stat().st_size))
            has_c2pa = C2PA_MARKER in head
            has_assertion = b"piksign.protection" in head
        except Exception as e:
            details["read_error"] = str(e)

        try:
            img = Image.open(path)
            if isinstance(img, PngImagePlugin.PngImageFile):
                txt = getattr(img, "text", {}) or {}
                if "c2pa_manifest" in txt or "piksign_manifest" in txt:
                    has_c2pa = True
                if any("piksign.protection" in str(v) for v in txt.values()):
                    has_assertion = True
        except Exception as e:
            details["pil_error"] = str(e)

        return PikSignCheckResult(
            is_protected=has_c2pa and has_assertion,
            has_c2pa_marker=has_c2pa,
            has_piksign_assertion=has_assertion,
            details=details,
        )
