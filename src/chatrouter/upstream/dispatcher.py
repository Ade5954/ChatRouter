"""Upstream dispatch with retries, model failover and graceful degradation.

The dispatcher walks the fallback chain produced by the router. Within one
model it retries transient errors with exponential backoff; when a model is
exhausted (or its breaker trips) it degrades to the next model in the chain.
Streaming is retryable only until the first byte reaches the client.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator

from ..config.models import AppConfig, ModelConfig
from ..core.errors import (
    AllUpstreamsFailedError,
    ChatRouterError,
    NoCapacityError,
    UpstreamError,
)
from ..core.types import AttemptRecord, DispatchResult, RequestContext
from ..governance.circuit_breaker import CircuitBreakerRegistry
from ..governance.load import ModelLoadTracker
from ..observability.logging import get_logger
from .client import ProviderPool

logger = get_logger(__name__)

_SSE_DONE = b"data: [DONE]"


class Dispatcher:
    """Executes a routing decision against the upstream providers."""

    def __init__(
        self,
        config: AppConfig,
        pool: ProviderPool,
        breakers: CircuitBreakerRegistry,
        load_tracker: ModelLoadTracker,
    ) -> None:
        self._config = config
        self._pool = pool
        self._breakers = breakers
        self._load = load_tracker
        self._retry = config.resilience.retry
        self._overflow = config.resilience.overflow

    # -- non-streaming --------------------------------------------------------

    async def dispatch(self, context: RequestContext, projected_tokens: int) -> DispatchResult:
        """Run the request through the fallback chain until one succeeds."""
        decision = context.decision
        assert decision is not None, "dispatch requires a routing decision"

        chain = decision.chain()
        last_error: ChatRouterError | None = None
        attempts_used = 0
        started = time.monotonic()

        for model in chain:
            if attempts_used >= self._retry.max_attempts:
                break
            if not self._breakers.allows(model.id):
                logger.warning("skipping model with open circuit", model=model.id)
                continue
            if not await self._ensure_capacity(model, projected_tokens):
                logger.warning("skipping saturated model", model=model.id)
                continue

            # Retry budget is shared across the chain so a failing primary
            # cannot consume the whole budget before failover happens.
            per_model_attempts = 1 if self._retry.enable_model_failover else self._retry.max_attempts
            for local_attempt in range(per_model_attempts):
                if attempts_used >= self._retry.max_attempts:
                    break
                attempts_used += 1
                record = AttemptRecord(model_id=model.id, started_at=time.monotonic())
                context.attempts.append(record)

                await self._load.reserve(model, projected_tokens)
                try:
                    result = await self._call_once(context, model, projected_tokens, record)
                except UpstreamError as exc:
                    record.finished_at = time.monotonic()
                    record.error = exc.message
                    record.status_code = exc.upstream_status
                    self._breakers.record_failure(model.id)
                    await self._load.release(model, 0, projected_tokens)
                    last_error = exc
                    if not self._is_retryable(exc):
                        # A genuine client error will fail on every model.
                        raise exc
                    logger.warning(
                        "upstream attempt failed",
                        model=model.id,
                        status=exc.upstream_status,
                        error=exc.message,
                        attempt=attempts_used,
                    )
                    await self._backoff(attempts_used)
                    continue
                except Exception as exc:  # defensive: never leak a raw error
                    record.finished_at = time.monotonic()
                    record.error = str(exc)
                    self._breakers.record_failure(model.id)
                    await self._load.release(model, 0, projected_tokens)
                    last_error = UpstreamError(str(exc), model_id=model.id)
                    await self._backoff(attempts_used)
                    continue

                self._breakers.record_success(model.id)
                await self._load.release(model, result.total_tokens, projected_tokens)
                result.attempts = attempts_used
                result.latency_ms = (time.monotonic() - started) * 1000
                return result

        if last_error is not None:
            raise AllUpstreamsFailedError(
                f"all {attempts_used} attempt(s) across {len(chain)} model(s) failed: "
                f"{last_error.message}"
            )
        raise NoCapacityError("no model in the fallback chain had capacity")

    async def _call_once(
        self,
        context: RequestContext,
        model: ModelConfig,
        projected_tokens: int,
        record: AttemptRecord,
    ) -> DispatchResult:
        """One upstream call, with usage extraction."""
        client = self._pool.get(model.provider)
        payload = self._build_payload(context, model)

        response = await client.chat_completion(payload)
        record.finished_at = time.monotonic()
        record.success = True
        record.status_code = response.status_code

        usage = response.payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        if not prompt_tokens and not completion_tokens:
            # Some providers omit usage; fall back to the estimate so quota and
            # TPM accounting stay meaningful.
            prompt_tokens = projected_tokens
        record.prompt_tokens = prompt_tokens
        record.completion_tokens = completion_tokens

        truncated = any(
            choice.get("finish_reason") == "length"
            for choice in response.payload.get("choices", [])
            if isinstance(choice, dict)
        )
        record.truncated = truncated

        # Present the gateway's model id to the client, not the upstream name.
        response.payload["model"] = model.id
        response.payload.setdefault("id", context.request_id)

        return DispatchResult(
            model=model,
            payload=response.payload,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=record.latency_ms or 0.0,
            attempts=1,
            truncated=truncated,
        )

    # -- streaming -------------------------------------------------------------

    async def dispatch_stream(
        self, context: RequestContext, projected_tokens: int
    ) -> AsyncIterator[bytes]:
        """Stream the response, failing over only before the first byte."""
        decision = context.decision
        assert decision is not None, "dispatch requires a routing decision"

        chain = decision.chain()
        last_error: ChatRouterError | None = None
        attempts_used = 0

        for model in chain:
            if attempts_used >= self._retry.max_attempts:
                break
            if not self._breakers.allows(model.id):
                continue
            if not await self._ensure_capacity(model, projected_tokens):
                continue

            attempts_used += 1
            record = AttemptRecord(model_id=model.id, started_at=time.monotonic())
            context.attempts.append(record)
            client = self._pool.get(model.provider)
            payload = self._build_payload(context, model, stream=True)

            await self._load.reserve(model, projected_tokens)
            released = False
            try:
                async with client.chat_completion_stream(payload) as lines:
                    first_chunk = True
                    async for chunk in self._relay(lines, context, model, record):
                        if first_chunk:
                            record.first_token_ms = (time.monotonic() - record.started_at) * 1000
                            first_chunk = False
                        yield chunk

                record.finished_at = time.monotonic()
                record.success = True
                self._breakers.record_success(model.id)
                await self._load.release(
                    model, record.prompt_tokens + record.completion_tokens, projected_tokens
                )
                released = True
                return
            except UpstreamError as exc:
                record.finished_at = time.monotonic()
                record.error = exc.message
                record.status_code = exc.upstream_status
                self._breakers.record_failure(model.id)
                if not released:
                    await self._load.release(model, 0, projected_tokens)
                    released = True
                last_error = exc
                if record.first_token_ms is not None:
                    # Bytes already reached the client: emit an in-band error
                    # instead of silently switching models mid-stream.
                    yield self._error_event(exc)
                    yield b"data: [DONE]\n\n"
                    return
                if not self._is_retryable(exc) or not self._retry.retry_streaming_before_first_chunk:
                    raise
                logger.warning("stream attempt failed before first chunk", model=model.id)
                await self._backoff(attempts_used)
                continue
            finally:
                if not released:
                    await self._load.release(model, 0, projected_tokens)

        if last_error is not None:
            raise AllUpstreamsFailedError(f"streaming failed on all candidates: {last_error.message}")
        raise NoCapacityError("no model in the fallback chain had capacity")

    async def _relay(
        self,
        lines: AsyncIterator[str],
        context: RequestContext,
        model: ModelConfig,
        record: AttemptRecord,
    ) -> AsyncIterator[bytes]:
        """Forward SSE lines, rewriting the model name and capturing usage."""
        async for line in lines:
            if not line:
                continue
            if not line.startswith("data:"):
                # Comments/keep-alives are forwarded verbatim.
                yield f"{line}\n\n".encode()
                continue

            data = line[5:].strip()
            if data == "[DONE]":
                yield b"data: [DONE]\n\n"
                continue

            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                yield f"{line}\n\n".encode()
                continue

            event["model"] = model.id
            event.setdefault("id", context.request_id)

            usage = event.get("usage")
            if isinstance(usage, dict):
                record.prompt_tokens = int(usage.get("prompt_tokens", record.prompt_tokens) or 0)
                record.completion_tokens = int(
                    usage.get("completion_tokens", record.completion_tokens) or 0
                )
            for choice in event.get("choices", []) or []:
                if isinstance(choice, dict) and choice.get("finish_reason") == "length":
                    record.truncated = True

            payload = json.dumps(event, ensure_ascii=False)
            yield f"data: {payload}\n\n".encode()

    @staticmethod
    def _error_event(error: ChatRouterError) -> bytes:
        body = json.dumps(error.to_payload(), ensure_ascii=False)
        return f"data: {body}\n\n".encode()

    # -- helpers ----------------------------------------------------------------

    def _build_payload(
        self, context: RequestContext, model: ModelConfig, stream: bool | None = None
    ) -> dict:
        """Serialise the request for a specific upstream model."""
        payload = context.request.upstream_payload(model.upstream_model)
        if stream is not None:
            payload["stream"] = stream
        if stream:
            # Ask for usage in the terminal chunk so accounting stays accurate.
            options = payload.get("stream_options") or {}
            options["include_usage"] = True
            payload["stream_options"] = options

        # Respect the target model's own output ceiling.
        if model.max_output_tokens:
            for field in ("max_tokens", "max_completion_tokens"):
                if payload.get(field):
                    payload[field] = min(int(payload[field]), model.max_output_tokens)

        if not model.supports_tools:
            payload.pop("tools", None)
            payload.pop("tool_choice", None)
            payload.pop("parallel_tool_calls", None)
        return payload

    def _is_retryable(self, error: UpstreamError) -> bool:
        if not error.retryable:
            return False
        if error.upstream_status is None:
            return True
        return error.upstream_status in self._retry.retry_on_status

    async def _backoff(self, attempt: int) -> None:
        """Exponential backoff with jitter to avoid synchronised retries."""
        base = self._retry.backoff_base_seconds
        if base <= 0:
            return
        delay = min(base * (2 ** (attempt - 1)), self._retry.backoff_max_seconds)
        await asyncio.sleep(delay * (0.5 + random.random() * 0.5))

    async def _ensure_capacity(self, model: ModelConfig, projected_tokens: int) -> bool:
        """Check headroom, optionally queueing briefly for it."""
        snapshot = await self._load.snapshot(model)
        if snapshot.has_headroom(projected_tokens):
            return True
        if not self._overflow.enabled or not self._overflow.queue_enabled:
            return False
        return await self._load.wait_for_capacity(
            model, projected_tokens, self._overflow.queue_max_wait_seconds
        )
