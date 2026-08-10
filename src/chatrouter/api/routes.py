"""HTTP surface: OpenAI-compatible endpoints plus gateway extensions."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..config.models import TenantConfig
from ..core.errors import ChatRouterError
from ..core.schemas import ChatCompletionRequest, FeedbackRequest, FeedbackResponse, ModelList
from ..observability import metrics
from ..service import GatewayService
from .auth import verify_admin_key

router = APIRouter()
admin_router = APIRouter(prefix="/admin", tags=["admin"])


def get_service(request: Request) -> GatewayService:
    service: GatewayService | None = getattr(request.app.state, "service", None)
    if service is None:  # pragma: no cover - app always sets this
        raise ChatRouterError("gateway is not initialised")
    return service


async def resolve_tenant(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> TenantConfig:
    """FastAPI dependency performing authentication."""
    service = get_service(request)
    return service.tenants.resolve(authorization, x_api_key)


ServiceDep = Annotated[GatewayService, Depends(get_service)]
TenantDep = Annotated[TenantConfig, Depends(resolve_tenant)]


@router.post("/v1/chat/completions", tags=["openai"])
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    service: ServiceDep,
    tenant: TenantDep,
) -> Response:
    """OpenAI-compatible chat completions with intelligent routing."""
    client_ip = request.client.host if request.client else None
    context, projected_tokens, headers = await service.prepare(payload, tenant, client_ip)

    if payload.stream:
        stream = service.stream(context, projected_tokens)
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={
                **headers,
                "cache-control": "no-cache",
                "connection": "keep-alive",
                "x-accel-buffering": "no",
            },
        )

    body, _ = await service.complete(context, projected_tokens)
    return JSONResponse(content=body, headers=headers)


@router.get("/v1/models", tags=["openai"], response_model=ModelList)
async def list_models(service: ServiceDep, tenant: TenantDep) -> ModelList:
    """List the models this tenant may target."""
    return service.list_models(tenant)


@router.get("/v1/models/{model_id}", tags=["openai"])
async def retrieve_model(model_id: str, service: ServiceDep, tenant: TenantDep) -> Response:
    """Retrieve a single model card."""
    listing = service.list_models(tenant)
    for card in listing.data:
        if card.id == model_id:
            return JSONResponse(content=card.model_dump())
    from ..core.errors import ModelNotFoundError

    raise ModelNotFoundError(f"model '{model_id}' does not exist")


@router.post("/v1/feedback", tags=["chatrouter"], response_model=FeedbackResponse)
async def submit_feedback(
    payload: FeedbackRequest, service: ServiceDep, tenant: TenantDep
) -> FeedbackResponse:
    """Report the perceived quality of a served request.

    This closes the adaptive routing loop: ratings adjust the effective quality
    of the model that served the request, in its complexity band.
    """
    return await service.submit_feedback(payload)


@router.post("/v1/routing/explain", tags=["chatrouter"])
async def explain_routing(
    payload: ChatCompletionRequest, service: ServiceDep, tenant: TenantDep
) -> dict[str, Any]:
    """Dry-run the router and return the full decision breakdown."""
    return await service.explain(payload, tenant)


@router.get("/healthz", tags=["ops"])
async def healthz() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@router.get("/readyz", tags=["ops"])
async def readyz(service: ServiceDep) -> Response:
    """Readiness probe: at least one model must be servable."""
    usable = [m for m in service.router.models if service.breakers.allows(m.id)]
    if not usable:
        return JSONResponse(
            status_code=503, content={"status": "unavailable", "reason": "no healthy model"}
        )
    return JSONResponse(content={"status": "ready", "healthy_models": len(usable)})


@admin_router.get("/status")
async def admin_status(
    service: ServiceDep,
    request: Request,
    x_admin_key: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """Live routing, load, quota and circuit-breaker state."""
    verify_admin_key(service.config, x_admin_key)
    return await service.runtime_status()


@admin_router.get("/config")
async def admin_config(
    service: ServiceDep,
    x_admin_key: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """Effective configuration with credentials redacted."""
    verify_admin_key(service.config, x_admin_key)
    data = service.config.model_dump(mode="json")
    for provider in data.get("providers", []):
        provider.pop("api_key", None)
    for tenant in data.get("tenants", []):
        tenant["api_keys"] = [f"***{key[-4:]}" if len(key) > 4 else "***" for key in tenant.get("api_keys", [])]
    return data


def build_metrics_route(path: str) -> APIRouter:
    """Expose the Prometheus scrape endpoint at the configured path."""
    metrics_router = APIRouter()

    @metrics_router.get(path, include_in_schema=False)
    async def prometheus_metrics() -> Response:
        payload, content_type = metrics.render_metrics()
        return Response(content=payload, media_type=content_type)

    return metrics_router
