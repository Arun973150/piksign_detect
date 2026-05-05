"""FastAPI entry point."""

import os
from pathlib import Path

from fastapi import Request
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import load_app_env
from frontend.api.routes import router

load_app_env()


def create_app() -> FastAPI:
    app = FastAPI(
        title="PikSign Detect",
        description="Production AI-image detection with multi-pathway fusion (local ensemble + SynthID + forensics).",
        version="1.0.0",
    )

    cors_origins = [
        origin.strip()
        for origin in os.environ.get("PIKSIGN_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ] or ["http://127.0.0.1:8000", "http://localhost:8000"]
    allow_credentials = "*" not in cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    allowed_hosts = [
        host.strip()
        for host in os.environ.get("PIKSIGN_ALLOWED_HOSTS", "").split(",")
        if host.strip()
    ]
    if allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        return response

    static_dir = Path(__file__).resolve().parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("PIKSIGN_HOST", "0.0.0.0")
    port = int(os.environ.get("PIKSIGN_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")
