"""Tenant quota accounting.

Quotas are longer-horizon budgets (hourly/daily/monthly) covering request
count, token volume and monetary spend. When a quota is exhausted the tenant is
either rejected or transparently downgraded to the cheapest tier, depending on
policy — downgrading keeps the product usable while protecting the budget.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config.models import TenantConfig
from ..storage.base import QuotaUsage, Storage

_PERIOD_SECONDS = {
    "hour": 3600,
    "day": 86_400,
    "month": 30 * 86_400,
}


@dataclass(slots=True)
class QuotaVerdict:
    """Outcome of a quota check."""

    allowed: bool
    downgrade: bool = False
    reason: str | None = None
    usage: QuotaUsage | None = None
    reset_seconds: float = 0.0

    def as_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.usage is not None:
            headers["x-chatrouter-quota-requests"] = str(self.usage.requests)
            headers["x-chatrouter-quota-tokens"] = str(self.usage.tokens)
            headers["x-chatrouter-quota-cost-usd"] = f"{self.usage.cost_usd:.6f}"
        if self.reset_seconds:
            headers["x-chatrouter-quota-reset"] = f"{self.reset_seconds:.0f}s"
        return headers


class QuotaManager:
    """Checks and records tenant consumption against configured quotas."""

    def __init__(self, storage: Storage) -> None:
        self._storage = storage

    @staticmethod
    def _key(tenant_id: str, period: str) -> str:
        return f"tenant:quota:{tenant_id}:{period}"

    @staticmethod
    def _window(tenant: TenantConfig) -> int:
        return _PERIOD_SECONDS[tenant.quota.period]

    async def check(self, tenant: TenantConfig, projected_tokens: int) -> QuotaVerdict:
        """Verify the tenant still has budget for this request."""
        quota = tenant.quota
        if not (quota.max_requests or quota.max_tokens or quota.max_cost_usd):
            return QuotaVerdict(allowed=True)

        window = self._window(tenant)
        usage = await self._storage.get_usage(self._key(tenant.id, quota.period), window)

        exceeded: str | None = None
        if quota.max_requests and usage.requests >= quota.max_requests:
            exceeded = f"request quota of {quota.max_requests} per {quota.period} exhausted"
        elif quota.max_tokens and usage.tokens + projected_tokens > quota.max_tokens:
            exceeded = f"token quota of {quota.max_tokens} per {quota.period} exhausted"
        elif quota.max_cost_usd and usage.cost_usd >= quota.max_cost_usd:
            exceeded = f"spend quota of ${quota.max_cost_usd} per {quota.period} exhausted"

        if exceeded is None:
            return QuotaVerdict(
                allowed=True, usage=usage, reset_seconds=usage.window_reset_seconds
            )

        if quota.on_exceed == "downgrade":
            # Keep serving, but only from the cheapest tier available.
            return QuotaVerdict(
                allowed=True,
                downgrade=True,
                reason=exceeded,
                usage=usage,
                reset_seconds=usage.window_reset_seconds,
            )
        return QuotaVerdict(
            allowed=False, reason=exceeded, usage=usage, reset_seconds=usage.window_reset_seconds
        )

    async def record(
        self, tenant: TenantConfig, tokens: int, cost_usd: float, requests: int = 1
    ) -> QuotaUsage:
        """Book actual consumption against the tenant's quota window."""
        return await self._storage.add_usage(
            self._key(tenant.id, tenant.quota.period),
            requests,
            tokens,
            cost_usd,
            self._window(tenant),
        )

    async def snapshot(self, tenant: TenantConfig) -> dict[str, object]:
        """Current quota consumption, for the admin endpoint."""
        usage = await self._storage.get_usage(
            self._key(tenant.id, tenant.quota.period), self._window(tenant)
        )
        return {
            "tenant": tenant.id,
            "period": tenant.quota.period,
            "requests": usage.requests,
            "max_requests": tenant.quota.max_requests,
            "tokens": usage.tokens,
            "max_tokens": tenant.quota.max_tokens,
            "cost_usd": round(usage.cost_usd, 6),
            "max_cost_usd": tenant.quota.max_cost_usd,
            "resets_in_seconds": round(usage.window_reset_seconds, 1),
            "on_exceed": tenant.quota.on_exceed,
        }
