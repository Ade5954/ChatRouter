"""Gateway service: wires routing, governance, dispatch and feedback together.

This is the single orchestration point for a request:

    auth → validate → estimate → rate limit → quota → route → dispatch
         → account → learn

Every stage is independently testable; the service only sequences them and
guarantees that reservations (concurrency slots, token budget) are released
exactly once, including on failure paths.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from .api.auth import TenantRegistry
from .config.models import AppConfig, ContextOverflowStrategy, ModelTier, TenantConfig
from .core.errors import (
    ChatRouterError,
    InvalidRequestError,
    QuotaExceededError,
    RateLimitError,
)
from .core.schemas import (
    ChatCompletionRequest,
    FeedbackRequest,
    FeedbackResponse,
    ModelCard,
    ModelList,
    new_request_id,
)
from .core.tokens import count_message_tokens, estimate_request_tokens
from .core.types import DispatchResult, RequestContext, RoutingDecision, estimate_cost_usd
from .governance.circuit_breaker import BreakerState, CircuitBreakerRegistry
from .governance.load import ModelLoadTracker
from .governance.quota import QuotaManager
from .governance.rate_limit import RateLimiter
from .observability import metrics
from .observability.logging import bind_request_context, clear_request_context, get_logger
from .routing.context_fit import fits, trim_to_fit
from .routing.feedback import FeedbackStore
from .routing.router import Router
from .storage import Storage, build_storage
from .upstream.client import ProviderPool
from .upstream.dispatcher import Dispatcher

logger = get_logger(__name__)

_BREAKER_STATE_VALUE = {
    BreakerState.CLOSED: 0,
    BreakerState.HALF_OPEN: 1,
    BreakerState.OPEN: 2,
}


class GatewayService:
    """Owns every long-lived component of the gateway."""

    def __init__(self, config: AppConfig, storage: Storage | None = None) -> None:
        self.config = config
        self.storage: Storage = storage or build_storage(config.storage)
        self.tenants = TenantRegistry(config)
        self.breakers = CircuitBreakerRegistry(config.resilience.circuit_breaker)
        self.load_tracker = ModelLoadTracker(self.storage, config.resilience.overflow)
        self.feedback = FeedbackStore(self.storage, config.routing.feedback)
        self.router = Router(config, self.feedback, self.load_tracker, self.breakers)
        self.rate_limiter = RateLimiter(self.storage)
        self.quotas = QuotaManager(self.storage)
        self.providers = ProviderPool(config.providers)
        self.dispatcher = Dispatcher(config, self.providers, self.breakers, self.load_tracker)

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        await self.storage.start()
        logger.info(
            "gateway started",
            models=len(self.router.models),
            providers=len(self.config.providers),
            tenants=len(self.config.tenants),
            storage=self.config.storage.backend,
        )

    async def close(self) -> None:
        await self.providers.close()
        await self.storage.close()

    # -- chat completions -------------------------------------------------------

    async def prepare(
        self,
        request: ChatCompletionRequest,
        tenant: TenantConfig,
        client_ip: str | None = None,
    ) -> tuple[RequestContext, int, dict[str, str]]:
        """Run every pre-dispatch stage and return the ready-to-serve context."""
        self._validate(request)

        request_id = new_request_id()
        hints = request.chatrouter
        context = RequestContext(
            request_id=request_id,
            tenant=tenant,
            request=request,
            session_id=hints.session_id if hints else None,
            client_ip=client_ip,
        )
        bind_request_context(request_id=request_id, tenant=tenant.id)

        prompt_tokens, projected_tokens = estimate_request_tokens(
            request.messages, request.tools, request.requested_max_tokens
        )

        headers: dict[str, str] = {"x-chatrouter-request-id": request_id}

        # --- rate limiting ---------------------------------------------------
        verdict, rl_headers = await self.rate_limiter.check_and_consume(tenant, projected_tokens)
        headers.update(rl_headers.as_headers())
        if not verdict.allowed:
            kind = "concurrency" if "concurrency" in (verdict.reason or "") else "rate"
            metrics.RATE_LIMITED.labels(tenant=tenant.id, kind=kind).inc()
            raise RateLimitError(
                verdict.reason or "rate limit exceeded",
                retry_after=verdict.retry_after,
                headers=headers,
            )

        # From here on a concurrency slot is held; release it on every failure.
        try:
            # --- quota --------------------------------------------------------
            quota_verdict = await self.quotas.check(tenant, projected_tokens)
            headers.update(quota_verdict.as_headers())
            if not quota_verdict.allowed:
                metrics.QUOTA_EVENTS.labels(tenant=tenant.id, action="reject").inc()
                raise QuotaExceededError(
                    quota_verdict.reason or "quota exceeded", headers=headers
                )
            if quota_verdict.downgrade:
                metrics.QUOTA_EVENTS.labels(tenant=tenant.id, action="downgrade").inc()
                context.quota_downgraded = True
                headers["x-chatrouter-quota-downgraded"] = "true"
                logger.warning("tenant quota exhausted, downgrading", reason=quota_verdict.reason)

            # --- routing --------------------------------------------------------
            decision = await self.router.route(context, projected_tokens)
            context.decision = decision
            headers["x-chatrouter-model"] = decision.model.id
            headers["x-chatrouter-routing-reason"] = decision.reason.value
            if decision.assessment:
                headers["x-chatrouter-complexity"] = f"{decision.assessment.score:.3f}"
                headers["x-chatrouter-tier"] = decision.assessment.tier.value

            # --- context overflow -------------------------------------------
            # Applied after routing because the budget depends on the model
            # that was actually selected.
            self._apply_context_overflow(request, decision, headers)

            self._record_decision_metrics(decision)
            logger.info(
                "routed request",
                model=decision.model.id,
                reason=decision.reason.value,
                complexity=(
                    round(decision.assessment.score, 3) if decision.assessment else None
                ),
                fallbacks=[m.id for m in decision.fallback_chain],
                prompt_tokens=prompt_tokens,
            )
            return context, projected_tokens, headers
        except Exception:
            await self.rate_limiter.release(tenant)
            raise

    async def complete(
        self, context: RequestContext, projected_tokens: int
    ) -> tuple[dict[str, Any], DispatchResult]:
        """Serve a non-streaming completion."""
        tenant = context.tenant
        try:
            result = await self.dispatcher.dispatch(context, projected_tokens)
        except ChatRouterError as exc:
            await self._finalise_failure(context, projected_tokens, exc)
            raise
        finally:
            await self.rate_limiter.release(tenant)

        await self._finalise_success(context, result, projected_tokens)
        return result.payload, result

    async def stream(
        self, context: RequestContext, projected_tokens: int
    ) -> AsyncIterator[bytes]:
        """Serve a streaming completion, accounting once the stream ends."""
        tenant = context.tenant
        succeeded = False
        try:
            async for chunk in self.dispatcher.dispatch_stream(context, projected_tokens):
                yield chunk
            succeeded = True
        except ChatRouterError as exc:
            await self._finalise_failure(context, projected_tokens, exc)
            raise
        finally:
            await self.rate_limiter.release(tenant)
            if succeeded:
                await self._finalise_stream(context, projected_tokens)

    # -- accounting and learning -------------------------------------------------

    async def _finalise_success(
        self, context: RequestContext, result: DispatchResult, projected_tokens: int
    ) -> None:
        tenant = context.tenant
        model = result.model
        total_tokens = result.total_tokens
        cost = estimate_cost_usd(model, result.prompt_tokens, result.completion_tokens)

        await self.rate_limiter.reconcile_tokens(tenant, projected_tokens, total_tokens)
        await self.quotas.record(tenant, total_tokens, cost)

        tier = context.decision.assessment.tier if context.decision and context.decision.assessment else None
        implicit = self.feedback.implicit_score(
            success=True,
            attempts=result.attempts,
            truncated=result.truncated,
            latency_ms=result.latency_ms,
            latency_prior_ms=model.latency_prior_ms,
        )
        await self.feedback.record_outcome(
            model.id, tier, success=True, latency_ms=result.latency_ms, implicit_score=implicit
        )
        await self.feedback.remember_request(
            context.request_id,
            model.id,
            tenant.id,
            tier,
            context.decision.assessment.score if context.decision and context.decision.assessment else None,
        )

        self._record_usage_metrics(context, model.id, result.prompt_tokens, result.completion_tokens, cost)
        metrics.REQUESTS_TOTAL.labels(tenant=tenant.id, model=model.id, status="success").inc()
        metrics.REQUEST_DURATION.labels(tenant=tenant.id, model=model.id).observe(
            context.elapsed_ms / 1000
        )
        if result.attempts > 1:
            first = context.attempts[0].model_id
            if first != model.id:
                metrics.FAILOVER_EVENTS.labels(from_model=first, to_model=model.id).inc()

        logger.info(
            "request completed",
            model=model.id,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=round(cost, 6),
            attempts=result.attempts,
            latency_ms=round(context.elapsed_ms, 1),
        )
        clear_request_context()

    async def _finalise_stream(self, context: RequestContext, projected_tokens: int) -> None:
        """Account a finished stream using the terminal usage chunk."""
        last = next((a for a in reversed(context.attempts) if a.success), None)
        if last is None:
            clear_request_context()
            return

        model = self.config.model_by_id(last.model_id)
        if model is None:
            clear_request_context()
            return

        prompt_tokens = last.prompt_tokens or projected_tokens
        completion_tokens = last.completion_tokens
        total = prompt_tokens + completion_tokens
        cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)

        await self.rate_limiter.reconcile_tokens(context.tenant, projected_tokens, total)
        await self.quotas.record(context.tenant, total, cost)

        tier = (
            context.decision.assessment.tier
            if context.decision and context.decision.assessment
            else None
        )
        implicit = self.feedback.implicit_score(
            success=True,
            attempts=len(context.attempts),
            truncated=last.truncated,
            latency_ms=last.latency_ms,
            latency_prior_ms=model.latency_prior_ms,
        )
        await self.feedback.record_outcome(
            model.id, tier, success=True, latency_ms=last.latency_ms, implicit_score=implicit
        )
        await self.feedback.remember_request(
            context.request_id,
            model.id,
            context.tenant.id,
            tier,
            context.decision.assessment.score
            if context.decision and context.decision.assessment
            else None,
        )

        if last.first_token_ms is not None:
            metrics.TIME_TO_FIRST_TOKEN.labels(model=model.id).observe(last.first_token_ms / 1000)
        self._record_usage_metrics(context, model.id, prompt_tokens, completion_tokens, cost)
        metrics.REQUESTS_TOTAL.labels(
            tenant=context.tenant.id, model=model.id, status="success"
        ).inc()
        metrics.REQUEST_DURATION.labels(tenant=context.tenant.id, model=model.id).observe(
            context.elapsed_ms / 1000
        )
        logger.info(
            "stream completed",
            model=model.id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=round(cost, 6),
            ttft_ms=round(last.first_token_ms, 1) if last.first_token_ms else None,
        )
        clear_request_context()

    async def _finalise_failure(
        self, context: RequestContext, projected_tokens: int, error: ChatRouterError
    ) -> None:
        """Release the token reservation and record the failure as evidence."""
        await self.rate_limiter.reconcile_tokens(context.tenant, projected_tokens, 0)

        tier = (
            context.decision.assessment.tier
            if context.decision and context.decision.assessment
            else None
        )
        for attempt in context.attempts:
            if attempt.success:
                continue
            await self.feedback.record_outcome(
                attempt.model_id,
                tier,
                success=False,
                latency_ms=attempt.latency_ms,
                implicit_score=0.0,
            )
            metrics.UPSTREAM_ERRORS.labels(
                model=attempt.model_id, status=str(attempt.status_code or "error")
            ).inc()

        model_id = context.decision.model.id if context.decision else "none"
        metrics.REQUESTS_TOTAL.labels(
            tenant=context.tenant.id, model=model_id, status="error"
        ).inc()
        logger.error(
            "request failed",
            model=model_id,
            error=error.message,
            code=error.code,
            attempts=len(context.attempts),
        )
        clear_request_context()

    def _record_usage_metrics(
        self,
        context: RequestContext,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
    ) -> None:
        tenant_id = context.tenant.id
        metrics.TOKENS_TOTAL.labels(tenant=tenant_id, model=model_id, direction="prompt").inc(
            prompt_tokens
        )
        metrics.TOKENS_TOTAL.labels(tenant=tenant_id, model=model_id, direction="completion").inc(
            completion_tokens
        )
        metrics.PROMPT_TOKENS.labels(model=model_id).observe(prompt_tokens)
        metrics.COST_TOTAL.labels(tenant=tenant_id, model=model_id).inc(cost)

    def _record_decision_metrics(self, decision) -> None:
        assessment = decision.assessment
        tier = assessment.tier.value if assessment else "explicit"
        metrics.ROUTING_DECISIONS.labels(
            reason=decision.reason.value, tier=tier, model=decision.model.id
        ).inc()
        if assessment:
            metrics.COMPLEXITY_SCORE.labels(tier=tier).observe(assessment.score)
            if assessment.latent_escalation > 0.05:
                metrics.CONTEXT_ESCALATIONS.labels(tier=tier).inc()

    # -- feedback API ------------------------------------------------------------

    def _apply_context_overflow(
        self,
        request: ChatCompletionRequest,
        decision: RoutingDecision,
        headers: dict[str, str],
    ) -> None:
        """Trim the conversation if it cannot fit the selected model.

        Only runs under the ``trim_history`` strategy. The other strategies
        either reject earlier in routing or accept the risk of routing to the
        widest model as-is.
        """
        cfg = self.config.routing.context_overflow
        if cfg.strategy is not ContextOverflowStrategy.TRIM_HISTORY:
            return

        model = decision.model
        reserve = request.requested_max_tokens or 0
        prompt_tokens = count_message_tokens(request.messages, model.id)
        if fits(model, prompt_tokens, reserve):
            return

        result = trim_to_fit(request.messages, model, cfg, reserve)
        if not result.trimmed:
            return

        request.messages = result.messages
        decision.notes.extend(result.notes)
        headers["x-chatrouter-context-trimmed"] = str(result.removed_messages)
        metrics.CONTEXT_TRIMMED.labels(model=model.id).inc()
        logger.warning(
            "conversation trimmed to fit context window",
            model=model.id,
            removed=result.removed_messages,
            before=result.original_tokens,
            after=result.final_tokens,
        )

    async def submit_feedback(self, payload: FeedbackRequest) -> FeedbackResponse:
        """Apply explicit user feedback to the adaptive routing statistics."""
        score = payload.normalised_score()
        if score is None:
            raise InvalidRequestError(
                "feedback must include one of: score, rating, thumb, accepted, "
                "regenerated or edited"
            )

        # Claiming consumes the record, so a request_id can only be scored
        # once. Without this an attacker holding a request_id (it is returned
        # in a response header) could replay negative feedback until the model
        # is evicted from routing.
        record = await self.feedback.claim_request(payload.request_id)
        if record is None:
            return FeedbackResponse(
                accepted=False,
                request_id=payload.request_id,
                detail=(
                    "request_id is unknown, expired, or has already received "
                    "feedback; this submission was discarded"
                ),
            )

        model_id = str(record.get("model_id"))
        band = str(record.get("band", "all"))
        await self.feedback.record_feedback(model_id, band, score)

        metrics.FEEDBACK_TOTAL.labels(
            model=model_id, polarity="positive" if score >= 0.5 else "negative"
        ).inc()
        logger.info(
            "feedback recorded", model=model_id, band=band, score=round(score, 3)
        )
        return FeedbackResponse(
            accepted=True, request_id=payload.request_id, model=model_id, applied_score=score
        )

    # -- introspection --------------------------------------------------------------

    def list_models(self, tenant: TenantConfig) -> ModelList:
        """OpenAI-compatible model listing, filtered by tenant permission."""
        cards: list[ModelCard] = []
        for alias in self.config.routing.auto_model_aliases:
            cards.append(
                ModelCard(
                    id=alias,
                    owned_by="chatrouter",
                    metadata={"virtual": True, "description": "context-aware automatic routing"},
                )
            )
        for model in self.router.models:
            if model.id in tenant.denied_models:
                continue
            if tenant.allowed_models and model.id not in tenant.allowed_models:
                continue
            if tenant.max_tier and model.tier.rank > tenant.max_tier.rank:
                continue
            cards.append(
                ModelCard(
                    id=model.id,
                    owned_by=model.provider,
                    metadata={
                        "tier": model.tier.value,
                        "context_window": model.context_window,
                        "supports_tools": model.supports_tools,
                        "supports_vision": model.supports_vision,
                    },
                )
            )
        return ModelList(data=cards)

    async def explain(
        self, request: ChatCompletionRequest, tenant: TenantConfig
    ) -> dict[str, Any]:
        """Dry-run the router and expose the full reasoning, without dispatching."""
        self._validate(request)
        context = RequestContext(
            request_id=new_request_id(), tenant=tenant, request=request
        )
        _, projected = estimate_request_tokens(
            request.messages, request.tools, request.requested_max_tokens
        )
        decision = await self.router.route(context, projected)
        return {
            "request_id": context.request_id,
            "tenant": tenant.id,
            "projected_tokens": projected,
            "decision": decision.as_dict(),
        }

    async def runtime_status(self) -> dict[str, Any]:
        """Aggregate health, load and learned quality for the admin API."""
        stats = await self.feedback.get_all_stats()
        loads = await self.load_tracker.snapshot_many(self.router.models)
        breaker_states = self.breakers.snapshot()

        models: list[dict[str, Any]] = []
        for model in self.router.models:
            model_stats = stats.get(model.id)
            effective = (
                self.feedback.effective_quality(model.quality_prior, model_stats)
                if model_stats
                else model.quality_prior
            )
            metrics.MODEL_QUALITY.labels(model=model.id).set(effective)
            snapshot = loads.get(model.id)
            if snapshot:
                metrics.MODEL_INFLIGHT.labels(model=model.id).set(snapshot.inflight)
            metrics.CIRCUIT_STATE.labels(model=model.id).set(
                _BREAKER_STATE_VALUE[self.breakers.state(model.id)]
            )
            models.append(
                {
                    "id": model.id,
                    "provider": model.provider,
                    "tier": model.tier.value,
                    "quality_prior": model.quality_prior,
                    "effective_quality": round(effective, 4),
                    "stats": model_stats.as_dict() if model_stats else None,
                    "load": snapshot.as_dict() if snapshot else None,
                    "circuit": breaker_states.get(model.id, {"state": "closed"}),
                }
            )

        tenants = []
        for tenant in self.tenants.all():
            tenants.append(
                {
                    "rate_limit": await self.rate_limiter.snapshot(tenant),
                    "quota": await self.quotas.snapshot(tenant),
                }
            )

        return {"models": models, "tenants": tenants, "timestamp": time.time()}

    # -- validation -----------------------------------------------------------------

    @staticmethod
    def _validate(request: ChatCompletionRequest) -> None:
        if not request.messages:
            raise InvalidRequestError("'messages' must contain at least one message", param="messages")
        if request.n is not None and request.n > 1:
            # Multi-candidate sampling breaks per-request cost attribution.
            raise InvalidRequestError("'n' greater than 1 is not supported by the gateway", param="n")
        if not request.model:
            raise InvalidRequestError("'model' is required", param="model")
