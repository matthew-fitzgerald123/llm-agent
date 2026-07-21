"""Serve the single-page web UI."""
from __future__ import annotations
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).parent / "static"


def mount_ui(app: FastAPI) -> None:
    """Register GET / to serve the web UI (kept out of the OpenAPI schema)."""

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")
