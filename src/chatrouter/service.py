"""Gateway service: wires routing, governance, dispatch and feedback together.

This is the single orchestration point for a request:

    auth → validate → estimate → rate limit → quota → route → dispatch
         → account → learn

Every stage is independently testable; the service only sequences them and
guarantees that reservations (concurrency slots, token budget) are released
exactly once, including on failure paths.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from .api.auth import TenantRegistry
from .cache.response_cache import ResponseCache
from .config.models import (
    AppConfig,
    ContextOverflowStrategy,
    ModelConfig,
    TenantConfig,
)
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
from .core.types import (
    DispatchResult,
    RequestContext,
    RoutingDecision,
    RoutingDecisionReason,
    estimate_cost_usd,
)
from .governance.circuit_breaker import BreakerState, CircuitBreakerRegistry
from .governance.load import ModelLoadTracker
from .governance.quota import QuotaManager
from .governance.rate_limit import RateLimiter
from .observability import metrics
from .observability.logging import bind_request_context, clear_request_context, get_logger
from .routing.context_fit import fits, trim_to_fit
from .routing.feedback import FeedbackStore
from .routing.feedback_normalizer import FeedbackNormalizer
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
        self.feedback_normalizer = FeedbackNormalizer.from_feedback_config(config.routing.feedback)
        self.router = Router(
            config, self.feedback, self.load_tracker, self.breakers, self.storage
        )
        self.rate_limiter = RateLimiter(self.storage)
        self.quotas = QuotaManager(self.storage)
        self.providers = ProviderPool(config.providers)
        self.dispatcher = Dispatcher(config, self.providers, self.breakers, self.load_tracker)
        self.response_cache = ResponseCache(config.routing.response_cache, self.storage)
        # Provider pools replaced by :meth:`reload`; closed on shutdown.
        self._retired_pools: list[ProviderPool] = []
        # Configuration version this replica has applied. The subscription
        # task compares incoming notifications against this value and skips
        # reloads for versions it has already seen (including the one it just
        # published itself — see ``apply_config_update``).
        self._applied_config_version: int = 0
        self._reload_watcher_task: asyncio.Task[None] | None = None
        # Belt-and-suspenders: even if the notification stream loses a message
        # (a bug, a Redis flush, an XAUTOCLAIM edge case), this periodic poll
        # of the canonical version catches up within the interval. It is the
        # last line of defence for eventual consistency.
        self._reconciler_task: asyncio.Task[None] | None = None
        self._reconcile_interval_seconds: float = 30.0

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        await self.storage.start()
        # Record the version of the configuration we are starting with so the
        # subscription task ignores the reload notification this replica may
        # publish for itself later (and any stale notification that arrives
        # during startup).
        self._applied_config_version = await self.storage.get_config_version()
        # Signal from the watcher task once it has registered its
        # subscription. We wait for it before returning so a publish() that
        # happens immediately after start() — e.g. another replica applying
        # a config update in the same event loop tick — cannot race past
        # the subscriber and be lost.
        self._watcher_subscribed = asyncio.Event()
        self._reload_watcher_task = asyncio.get_running_loop().create_task(
            self._watch_config_reloads()
        )
        await self._watcher_subscribed.wait()
        # Start the reconciler after the watcher is live so the two never
        # race: the watcher handles the common path, the reconciler only
        # fires when it detects drift.
        self._reconciler_task = asyncio.get_running_loop().create_task(
            self._reconcile_config_version()
        )
        logger.info(
            "gateway started",
            models=len(self.router.models),
            providers=len(self.config.providers),
            tenants=len(self.config.tenants),
            storage=self.config.storage.backend,
            config_version=self._applied_config_version,
        )

    async def close(self) -> None:
        # Cancel the watcher and reconciler first so neither can observe a
        # half-shutdown service (storage closed, providers gone) and try to
        # reload.
        for task in (self._reload_watcher_task, self._reconciler_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._reload_watcher_task = None
        self._reconciler_task = None
        await self.providers.close()
        for pool in self._retired_pools:
            await pool.close()
        await self.storage.close()

    async def reload(self, new_config: AppConfig) -> None:
        """Hot-swap the configuration without dropping the process.

        Every component is rebuilt from ``new_config``; the storage backend is
        kept (its counters/learned statistics survive the reload). In-flight
        requests hold references to the old dispatcher and provider pool, so
        the old pool is closed after a grace window instead of immediately
        (and on shutdown as well).
        """
        old_providers = self.providers
        self.config = new_config
        self.tenants = TenantRegistry(new_config)
        self.breakers = CircuitBreakerRegistry(new_config.resilience.circuit_breaker)
        self.load_tracker = ModelLoadTracker(self.storage, new_config.resilience.overflow)
        self.feedback = FeedbackStore(self.storage, new_config.routing.feedback)
        self.feedback_normalizer = FeedbackNormalizer.from_feedback_config(
            new_config.routing.feedback
        )
        self.router = Router(
            new_config, self.feedback, self.load_tracker, self.breakers, self.storage
        )
        self.rate_limiter = RateLimiter(self.storage)
        self.quotas = QuotaManager(self.storage)
        self.providers = ProviderPool(new_config.providers)
        self.dispatcher = Dispatcher(
            new_config, self.providers, self.breakers, self.load_tracker
        )
        self.response_cache = ResponseCache(new_config.routing.response_cache, self.storage)

        self._retired_pools.append(old_providers)
        asyncio.get_running_loop().create_task(self._close_old_providers(old_providers))
        logger.info(
            "configuration reloaded",
            models=len(new_config.models),
            providers=len(new_config.providers),
            tenants=len(new_config.tenants),
        )

    async def reload_from_storage(self) -> bool:
        """Pull the authoritative configuration from storage and reload.

        Returns ``True`` if a configuration was found and applied, ``False``
        when storage holds no configuration (so callers can skip silently).
        Used by the cross-replica reload subscription (Phase 2) and by the
        admin endpoint after it persists an update.
        """
        from .config.loader import load_config_from_storage

        new_config = await load_config_from_storage(self.storage)
        if new_config is None:
            return False
        await self.reload(new_config)
        self._applied_config_version = await self.storage.get_config_version()
        return True

    async def apply_config_update(self, config_dict: dict[str, Any]) -> AppConfig:
        """Persist, apply, and announce a configuration update.

        This is the single entry point for admin-driven changes: it writes the
        new document to the shared store, hot-applies it to *this* replica,
        and publishes a reload notification so every other replica converges
        on the same configuration. Callers should validate ``config_dict``
        into an ``AppConfig`` first (and surface validation errors to the
        client) before calling this — but we re-validate here as a defence
        in depth so the store can never hold an invalid document.
        """
        from .config.models import AppConfig

        new_config = AppConfig.model_validate(config_dict)
        await self.storage.set_config(config_dict)
        await self.reload(new_config)
        # Bump our own applied version *before* publishing: the pub/sub
        # message will be delivered back to this same replica (Redis fans
        # out to every subscriber, including the publisher), and the watcher
        # must treat it as already-applied rather than re-reloading.
        self._applied_config_version = await self.storage.get_config_version()
        await self.storage.publish_config_reload(self._applied_config_version)
        return new_config

    async def _watch_config_reloads(self) -> None:
        """Background task: react to cross-replica config reload notifications.

        Subscribes to the storage's reload channel and pulls the fresh
        configuration from storage whenever a newer version arrives. The
        task runs for the lifetime of the service and is cancelled on
        shutdown. Errors are logged and swallowed so a transient storage
        hiccup cannot kill the subscription permanently.
        """
        try:
            # Establish the subscription first and signal readiness before
            # entering the consume loop: start() awaits this signal so a
            # publish() in the same tick cannot race past the subscriber.
            events = self.storage.config_reload_events()
            self._watcher_subscribed.set()
            async for version in events:
                if version <= self._applied_config_version:
                    # Stale or self-published notification: we have already
                    # applied this version (or a newer one) via the local
                    # reload path. Skipping avoids a redundant rebuild and
                    # the double-close of the provider pool that a second
                    # reload would trigger.
                    continue
                try:
                    applied = await self.reload_from_storage()
                    if not applied:
                        logger.warning(
                            "config reload notification received but storage holds no config",
                            version=version,
                        )
                    else:
                        logger.info(
                            "config reloaded from cross-replica notification",
                            version=version,
                        )
                except Exception:
                    # The reload failed; leave the previous configuration
                    # in place and keep listening. The next notification (or
                    # the next admin write) will retry.
                    logger.exception(
                        "config reload from storage failed", version=version
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            # The subscription itself crashed (storage connection lost, etc.).
            # Without it this replica will keep serving with its last-known
            # configuration until restarted; the next admin write will
            # republish and any live replicas will catch up.
            logger.exception("config reload subscription crashed")

    async def _reconcile_config_version(self) -> None:
        """Periodically check the canonical config version against our own.

        This is the belt-and-suspenders backstop for the notification stream:
        if a notification is lost for any reason (stream eviction by MAXLEN,
        an XAUTOCLAIM edge case, a bug), this poll catches the version drift
        within ``_reconcile_interval_seconds`` and reloads from storage. It is
        strictly a safety net — the notification stream is the primary path
        and reacts in milliseconds, while this runs every 30s.
        """
        try:
            while True:
                await asyncio.sleep(self._reconcile_interval_seconds)
                try:
                    current = await self.storage.get_config_version()
                    if current > self._applied_config_version:
                        logger.warning(
                            "config version drift detected by reconciler",
                            applied=self._applied_config_version,
                            storage=current,
                        )
                        await self.reload_from_storage()
                except Exception:
                    # Storage may be temporarily unreachable; keep the
                    # reconciler alive so it retries on the next tick.
                    logger.exception("config reconciliation check failed")
        except asyncio.CancelledError:
            raise

    async def _close_old_providers(self, pool: ProviderPool, grace_seconds: float = 60.0) -> None:
        """Close a replaced provider pool after in-flight requests drain."""
        await asyncio.sleep(grace_seconds)
        await pool.close()
        if pool in self._retired_pools:
            self._retired_pools.remove(pool)

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

        # Token estimation is CPU-heavy (tiktoken + per-message heuristics);
        # run it off the event loop so it cannot block concurrent requests.
        prompt_tokens, projected_tokens = await asyncio.to_thread(
            estimate_request_tokens,
            request.messages,
            request.tools,
            request.requested_max_tokens,
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
            decision = await self.router.route(
                context, projected_tokens, prompt_tokens=prompt_tokens
            )
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
        """Serve a non-streaming completion, short-circuiting on a cache hit.

        Accounting (quota, feedback learning, session affinity, cache
        write-through) is *not* awaited here: the response body is returned to
        the caller as soon as the upstream completes and the concurrency slot
        is released, and the accounting runs as a background task afterwards.
        Failure accounting stays synchronous so error responses carry complete
        state.
        """
        tenant = context.tenant
        cache = self.response_cache
        decision = context.decision
        model = decision.model if decision else None

        if (
            model is not None
            and cache.should_participate(context.request, tenant)
        ):
            key = cache.key_for(context.request, model)
            entry = await cache.get(key)
            if entry is not None:
                payload = entry["payload"]
                cached_model_id = entry.get("model_id") or model.id
                cached_model = self.config.model_by_id(cached_model_id) or model
                result = self._result_from_cache(context, cached_model, payload, projected_tokens)
                metrics.CACHE_HITS.labels(tenant=tenant.id, model=cached_model.id).inc()
                context.cache_hit = True
                await self.rate_limiter.release(tenant)
                return result.payload, result

            metrics.CACHE_MISSES.labels(tenant=tenant.id, model=model.id, reason="miss").inc()
        elif model is not None:
            reason = (
                "streaming"
                if context.request.stream
                else "disabled"
                if not self.config.routing.response_cache.enabled
                else "excluded"
            )
            metrics.CACHE_MISSES.labels(tenant=tenant.id, model=model.id, reason=reason).inc()

        try:
            result = await self.dispatcher.dispatch(context, projected_tokens)
        except ChatRouterError as exc:
            await self._finalise_failure(context, projected_tokens, exc)
            await self.rate_limiter.release(tenant)
            raise

        # The slot is released synchronously so concurrency limits stay exact;
        # the rest of the accounting happens in the response's background task.
        await self.rate_limiter.release(tenant)
        return result.payload, result

    def _result_from_cache(
        self,
        context: RequestContext,
        model: ModelConfig,
        payload: dict[str, Any],
        projected_tokens: int,
    ) -> DispatchResult:
        """Reconstruct a DispatchResult from a cached payload.

        Token counts come from the cached ``usage`` block when present,
        otherwise from the request estimate — accounting is approximate but
        never zero, so quota and cost stay meaningful.
        """
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0) or projected_tokens
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)

        # A cached response belongs to the original request id, not a fresh one.
        served = dict(payload)
        served["model"] = model.id
        served.setdefault("id", context.request_id)

        return DispatchResult(
            model=model,
            payload=served,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=0.0,
            attempts=0,
            truncated=any(
                c.get("finish_reason") == "length"
                for c in served.get("choices", []) or []
                if isinstance(c, dict)
            ),
        )

    async def stream(
        self, context: RequestContext, projected_tokens: int
    ) -> AsyncIterator[bytes]:
        """Serve a streaming completion, accounting once the stream ends.

        The accounting itself runs as a background task attached to the
        ``StreamingResponse``: ``_finalise_stream`` already no-ops when the
        stream failed before any successful attempt, so the background task
        only needs to inspect the attempts recorded by the dispatcher.
        """
        tenant = context.tenant
        try:
            async for chunk in self.dispatcher.dispatch_stream(context, projected_tokens):
                yield chunk
        except ChatRouterError as exc:
            await self._finalise_failure(context, projected_tokens, exc)
            # The response has already started (HTTP 200 + SSE headers), so a
            # raise would leave the client with a silent empty stream. Emit the
            # failure in-band instead; the terminal [DONE] lets SSE parsers
            # finish cleanly.
            yield self.dispatcher._error_event(exc)
            yield b"data: [DONE]\n\n"
        finally:
            await self.rate_limiter.release(tenant)

    # -- accounting and learning -------------------------------------------------

    async def _finalise_success_bg(
        self, context: RequestContext, result: DispatchResult, projected_tokens: int
    ) -> None:
        """Run post-response accounting in the background.

        Invoked from the response's ``BackgroundTask`` after the body has been
        sent, so the client is never kept waiting for the quota/feedback
        bookkeeping. Failures here must not surface to the client — the
        response is already committed — so they are logged and swallowed.
        """
        bind_request_context(request_id=context.request_id, tenant=context.tenant.id)
        try:
            await self._finalise_success(context, result, projected_tokens)
            await self._maybe_cache_result(context, result, projected_tokens)
        except Exception:
            logger.exception("background accounting failed", request_id=context.request_id)
        finally:
            clear_request_context()

    async def _finalise_stream_bg(
        self, context: RequestContext, projected_tokens: int
    ) -> None:
        """Background counterpart of :meth:`_finalise_stream`.

        ``_finalise_stream`` itself already no-ops when no attempt succeeded
        (the stream failed before any bytes, or the client disconnected), so
        the background task never needs to know how the stream ended.
        """
        bind_request_context(request_id=context.request_id, tenant=context.tenant.id)
        try:
            await self._finalise_stream(context, projected_tokens)
        except Exception:
            logger.exception("background stream accounting failed", request_id=context.request_id)
        finally:
            clear_request_context()

    async def _maybe_cache_result(
        self, context: RequestContext, result: DispatchResult, projected_tokens: int
    ) -> None:
        """Write-through a clean success into the exact-match response cache."""
        cache = self.response_cache
        decision = context.decision
        if (
            context.cache_hit
            or decision is None
            or result.completion_tokens <= 0
            or not cache.should_participate(context.request, context.tenant)
        ):
            return

        affinity_ttl: int | None = None
        if (
            self.config.routing.session_affinity.enabled
            and context.session_id
            and cache._config.affinity_aware
        ):
            # Cap the cache entry at the remaining affinity binding so a
            # cached answer can never outlive the model the session is pinned
            # to (which would otherwise serve a stale model's response).
            affinity_ttl = await self.storage.session_affinity_ttl(context.session_id)
        await cache.put(
            cache.key_for(context.request, decision.model),
            result.payload,
            decision.model.id,
            affinity_ttl_seconds=affinity_ttl,
        )
        metrics.CACHE_STORED.labels(tenant=context.tenant.id, model=decision.model.id).inc()

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
        await self._update_session_affinity(context, model.id, result.prompt_tokens)
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
        await self._update_session_affinity(context, model.id, prompt_tokens)
        clear_request_context()

    async def _update_session_affinity(
        self, context: RequestContext, model_id: str, prompt_tokens: int
    ) -> None:
        """Persist the model a session routed to, accumulating its cached prefix.

        This is the single source of truth for session affinity: each turn's
        prompt tokens are added to the session's historical prefix so the router
        can price the real cost of switching away (prefix-cache loss,
        ``prefix_tokens * (c_in - c_cache)``). Pinned/explicit-model sessions are
        stable by construction and are skipped.
        """
        if self.storage is None or not context.session_id:
            return
        affinity_cfg = self.config.routing.session_affinity
        if not affinity_cfg.enabled:
            return
        decision = context.decision
        if decision is not None and decision.reason in (
            RoutingDecisionReason.PINNED,
            RoutingDecisionReason.EXPLICIT_MODEL,
        ):
            return
        # The router already read this session's binding for the routing
        # decision; reuse its prefix instead of re-reading storage (one fewer
        # round-trip per request).
        prev = context.affinity_prefix_tokens
        model = self.config.model_by_id(model_id)
        cap = model.context_window if model is not None else 0
        new_prefix = prev + max(0, int(prompt_tokens))
        if cap > 0:
            new_prefix = min(new_prefix, cap)
        await self.storage.set_session_affinity(
            context.session_id, model_id, new_prefix, affinity_cfg.ttl_seconds
        )

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
        normalized = self.feedback_normalizer.normalize(payload)
        if normalized is None:
            raise InvalidRequestError(
                "feedback must include one of: score, rating, thumb, accepted, "
                "regenerated or edited"
            )
        score = normalized.score

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
            "feedback recorded",
            model=model_id,
            band=band,
            score=round(score, 3),
            source=normalized.source,
        )
        return FeedbackResponse(
            accepted=True,
            request_id=payload.request_id,
            model=model_id,
            applied_score=score,
            source=normalized.source,
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
        prompt, projected = await asyncio.to_thread(
            estimate_request_tokens, request.messages, request.tools, request.requested_max_tokens
        )
        decision = await self.router.route(context, projected, prompt_tokens=prompt)
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
