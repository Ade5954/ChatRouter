"""FastAPI application factory."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import admin_router, build_metrics_route, router
from .config.loader import (
    bootstrap_config_to_storage,
    load_config,
    load_config_from_storage,
    resolve_config_path,
)
from .config.models import AppConfig
from .core.errors import ChatRouterError
from .core.schemas import error_payload
from .observability.logging import configure_logging, get_logger
from .service import GatewayService
from .storage import Storage, build_storage

logger = get_logger(__name__)

# Optional bundled web UI (served at ``/`` when the directory is present).
STATIC_DIR = Path(__file__).resolve().parent / "static"


async def _resolve_authoritative_config(
    bootstrap: AppConfig, storage: Storage
) -> AppConfig:
    """Return the configuration this replica should run.

    The storage backend is the source of truth in multi-replica deployments:
    if it already holds a configuration document we adopt it verbatim.
    Otherwise the bootstrap configuration (loaded from the local YAML, with
    ``${ENV}`` already expanded) is persisted as the seed so every subsequent
    replica converges on it.
    """
    stored = await load_config_from_storage(storage)
    if stored is not None:
        logger.info(
            "loaded configuration from storage",
            version=await storage.get_config_version(),
        )
        return stored
    await bootstrap_config_to_storage(storage, bootstrap)
    logger.info("bootstrapped configuration into storage from local file")
    return bootstrap


def create_app(config: AppConfig | None = None, config_path: str | None = None) -> FastAPI:
    """Build the ASGI application."""
    # Bootstrap configuration: always loaded from the local YAML so the process
    # knows which storage backend to connect to. The rest of the document
    # (providers, models, tenants, routing) may be overridden by whatever is
    # already persisted in storage — see ``_resolve_authoritative_config``.
    bootstrap = config or load_config(config_path)
    effective_path = None
    if config is None:
        effective_path = str(resolve_config_path(config_path))
    configure_logging(bootstrap.observability.log_level, bootstrap.observability.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # 1. Connect the storage using the bootstrap settings.
        storage = build_storage(bootstrap.storage)
        await storage.start()
        # 2. Load the authoritative configuration (from storage if present,
        #    otherwise seed storage from the bootstrap document).
        app_config = await _resolve_authoritative_config(bootstrap, storage)
        # 3. Build the service with the resolved configuration and the
        #    already-connected storage (service.start() is idempotent on it).
        service = GatewayService(app_config, storage=storage)
        await service.start()
        app.state.service = service
        app.state.config = app_config
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(
        title="ChatRouter",
        description=(
            "Production-grade, OpenAI-compatible LLM traffic gateway with "
            "full-context-aware routing and an online feedback-adaptive policy."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.config = bootstrap
    app.state.config_path = effective_path

    if bootstrap.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=bootstrap.server.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        """Expose the gateway's own overhead so it can be monitored."""
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-chatrouter-latency-ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
        return response

    @app.exception_handler(ChatRouterError)
    async def handle_gateway_error(request: Request, exc: ChatRouterError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content=exc.to_payload(), headers=exc.headers or None
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return JSONResponse(
            status_code=400,
            content=error_payload(
                first.get("msg", "invalid request body"),
                "invalid_request_error",
                code="invalid_request",
                param=location or None,
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled gateway error", path=request.url.path)
        return JSONResponse(
            status_code=500,
            content=error_payload("internal gateway error", "server_error", code="internal_error"),
        )

    app.include_router(router)
    app.include_router(admin_router)
    if bootstrap.observability.metrics_enabled:
        app.include_router(build_metrics_route(bootstrap.observability.metrics_path))

    # Bundled web UI: the management console is served at ``/`` whenever the
    # static assets ship with the package. The API remains fully usable
    # without it.
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app
