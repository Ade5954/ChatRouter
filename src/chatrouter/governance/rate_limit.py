"""Tenant rate limiting: RPM, TPM and concurrency.

Limits are enforced *before* the upstream call. Token limits work on an
estimate, which is reconciled against real usage once the response is known —
this keeps TPM accurate without blocking on tokenisation of the completion.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config.models import TenantConfig
from ..storage.base import RateLimitVerdict, Storage

_WINDOW_SECONDS = 60


@dataclass(slots=True)
class RateLimitHeaders:
    """Values surfaced to clients as ``x-ratelimit-*`` response headers."""

    limit_requests: int | None = None
    remaining_requests: int | None = None
    limit_tokens: int | None = None
    remaining_tokens: int | None = None
    reset_seconds: float | None = None

    def as_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.limit_requests is not None:
            headers["x-ratelimit-limit-requests"] = str(self.limit_requests)
        if self.remaining_requests is not None:
            headers["x-ratelimit-remaining-requests"] = str(max(0, self.remaining_requests))
        if self.limit_tokens is not None:
            headers["x-ratelimit-limit-tokens"] = str(self.limit_tokens)
        if self.remaining_tokens is not None:
            headers["x-ratelimit-remaining-tokens"] = str(max(0, self.remaining_tokens))
        if self.reset_seconds is not None:
            headers["x-ratelimit-reset-requests"] = f"{self.reset_seconds:.0f}s"
        return headers


class RateLimiter:
    """Enforces per-tenant request, token and concurrency limits."""

    def __init__(self, storage: Storage) -> None:
        self._storage = storage

    @staticmethod
    def _rpm_key(tenant_id: str) -> str:
        return f"tenant:rpm:{tenant_id}"

    @staticmethod
    def _tpm_key(tenant_id: str) -> str:
        return f"tenant:tpm:{tenant_id}"

    @staticmethod
    def _inflight_key(tenant_id: str) -> str:
        return f"tenant:inflight:{tenant_id}"

    async def check_and_consume(
        self, tenant: TenantConfig, projected_tokens: int
    ) -> tuple[RateLimitVerdict, RateLimitHeaders]:
        """Atomically verify and reserve the tenant's budget.

        Counters are incremented first and rolled back on rejection: this keeps
        the check race-free across replicas at the cost of a brief overshoot
        that is immediately corrected.
        """
        limits = tenant.rate_limit
        headers = RateLimitHeaders(
            limit_requests=limits.rpm,
            limit_tokens=limits.tpm,
            reset_seconds=await self._storage.window_ttl(self._rpm_key(tenant.id), _WINDOW_SECONDS),
        )

        # --- concurrency -------------------------------------------------
        if limits.max_concurrency:
            inflight = await self._storage.incr_gauge(self._inflight_key(tenant.id), 1)
            if inflight > limits.max_concurrency:
                await self._storage.incr_gauge(self._inflight_key(tenant.id), -1)
                return (
                    RateLimitVerdict(
                        allowed=False,
                        limit=limits.max_concurrency,
                        remaining=0,
                        retry_after=1.0,
                        reason=(
                            f"concurrency limit of {limits.max_concurrency} "
                            f"reached for tenant '{tenant.id}'"
                        ),
                    ),
                    headers,
                )

        # --- requests per minute --------------------------------------------
        if limits.rpm:
            used = await self._storage.incr_window(self._rpm_key(tenant.id), 1, _WINDOW_SECONDS)
            headers.remaining_requests = max(0, limits.rpm - used)
            if used > limits.rpm:
                await self._storage.incr_window(self._rpm_key(tenant.id), -1, _WINDOW_SECONDS)
                await self._release_concurrency(tenant)
                retry_after = await self._storage.window_ttl(self._rpm_key(tenant.id), _WINDOW_SECONDS)
                return (
                    RateLimitVerdict(
                        allowed=False,
                        limit=limits.rpm,
                        remaining=0,
                        retry_after=retry_after,
                        reason=f"request rate limit of {limits.rpm} RPM exceeded",
                    ),
                    headers,
                )

        # --- tokens per minute ------------------------------------------------
        if limits.tpm and projected_tokens:
            used = await self._storage.incr_window(
                self._tpm_key(tenant.id), projected_tokens, _WINDOW_SECONDS
            )
            headers.remaining_tokens = max(0, limits.tpm - used)
            if used > limits.tpm:
                await self._storage.incr_window(
                    self._tpm_key(tenant.id), -projected_tokens, _WINDOW_SECONDS
                )
                if limits.rpm:
                    await self._storage.incr_window(self._rpm_key(tenant.id), -1, _WINDOW_SECONDS)
                await self._release_concurrency(tenant)
                retry_after = await self._storage.window_ttl(self._tpm_key(tenant.id), _WINDOW_SECONDS)
                return (
                    RateLimitVerdict(
                        allowed=False,
                        limit=limits.tpm,
                        remaining=max(0, limits.tpm - (used - projected_tokens)),
                        retry_after=retry_after,
                        reason=f"token rate limit of {limits.tpm} TPM exceeded",
                    ),
                    headers,
                )

        return RateLimitVerdict(allowed=True, limit=limits.rpm, remaining=headers.remaining_requests), headers

    async def reconcile_tokens(
        self, tenant: TenantConfig, projected_tokens: int, actual_tokens: int
    ) -> None:
        """Correct the TPM window once real usage is known."""
        if not tenant.rate_limit.tpm:
            return
        delta = actual_tokens - projected_tokens
        if delta:
            await self._storage.incr_window(self._tpm_key(tenant.id), delta, _WINDOW_SECONDS)

    async def release(self, tenant: TenantConfig) -> None:
        """Release the concurrency slot held by a finished request."""
        await self._release_concurrency(tenant)

    async def _release_concurrency(self, tenant: TenantConfig) -> None:
        if tenant.rate_limit.max_concurrency:
            await self._storage.incr_gauge(self._inflight_key(tenant.id), -1)

    async def snapshot(self, tenant: TenantConfig) -> dict[str, object]:
        """Current consumption, for the admin endpoint."""
        return {
            "tenant": tenant.id,
            "rpm_used": await self._storage.get_window(self._rpm_key(tenant.id), _WINDOW_SECONDS),
            "rpm_limit": tenant.rate_limit.rpm,
            "tpm_used": await self._storage.get_window(self._tpm_key(tenant.id), _WINDOW_SECONDS),
            "tpm_limit": tenant.rate_limit.tpm,
            "inflight": await self._storage.get_gauge(self._inflight_key(tenant.id)),
            "concurrency_limit": tenant.rate_limit.max_concurrency,
        }
