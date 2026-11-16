# backend/aegis/main.py
"""FastAPI application entry point for AegisAI AgentSwarm Backend.

Start with::

    uvicorn aegis.main:app --reload
"""

from __future__ import annotations

import faulthandler
import os
import signal
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aegis.api import router as api_router
from aegis.api.rag import router as rag_router
from aegis.api.routes import hydrate_document_registry_on_startup
from aegis.core.config import settings
from aegis.core.logging import setup_logging

# Diagnostics: `kill -USR1 <pid>` dumps all thread stacks to stderr.
if os.environ.get("AEGIS_ENABLE_FAULTHANDLER", "1") == "1":
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=True)
    except Exception:  # pragma: no cover - non-POSIX platforms
        pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle."""
    setup_logging()
    if settings.ENVIRONMENT.lower() != "test":
        hydrate_document_registry_on_startup()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
app.include_router(rag_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }
