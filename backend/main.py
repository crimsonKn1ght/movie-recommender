"""FastAPI application entry point.

Owns process-level startup: registers health probes before anything else, kicks
off background model warm-up, configures CORS, and dynamically registers the
business routers under ``app/api``. Mirrors the documented backend architecture,
scaled down to a stateless service (no database, no auth middleware).
"""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.registry import registry
from app.utils.settings import settings
from router_loader import register_routes

HEALTH_PATHS = [
    "/health", "/ready", "/healthcheck",
    "/api/health", "/api/ready", "/api/healthcheck",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Train the fast models in the background so probes are never blocked and the
    # first /recommend call does not pay the training cost.
    if settings.AUTO_TRAIN_ON_STARTUP:
        threading.Thread(target=registry.warm_up, daemon=True).start()
    yield


def _register_health(app: FastAPI) -> None:
    async def health():
        return {"status": "ok"}

    for path in HEALTH_PATHS:
        app.add_api_route(path, health, methods=["GET"], tags=["health"])


def create_app() -> FastAPI:
    app = FastAPI(
        title="Movie Recommender API",
        version="0.1.0",
        description="REST API wrapping the SVD / NMF / Neural CF recommendation engine.",
        lifespan=lifespan,
    )

    # Health endpoints first, so they are always reachable.
    _register_health(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_routes(app)
    return app


app = create_app()
