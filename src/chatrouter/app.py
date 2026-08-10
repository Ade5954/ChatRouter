"""FastAPI application factory."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import admin_router, build_metrics_route, router
from .config.loader import load_config
from .config.models import AppConfig
from .core.errors import ChatRouterError
from .core.schemas import error_payload
from .observability.logging import configure_logging, get_logger
from .service import GatewayService

logger = get_logger(__name__)


def create_app(config: AppConfig | None = None, config_path: str | None = None) -> FastAPI:
    """Build the ASGI application."""
    app_config = config or load_config(config_path)
    configure_logging(app_config.observability.log_level, app_config.observability.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service = GatewayService(app_config)
        await service.start()
        app.state.service = service
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
    app.state.config = app_config

    if app_config.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=app_config.server.cors_origins,
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
    if app_config.observability.metrics_enabled:
        app.include_router(build_metrics_route(app_config.observability.metrics_path))

    return app
