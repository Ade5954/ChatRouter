"""Tests for rate limiting, quotas, circuit breaking and overflow."""

from __future__ import annotations

import asyncio

from chatrouter.config.models import (
    CircuitBreakerConfig,
    OverflowConfig,
    QuotaConfig,
    RateLimitConfig,
    TenantConfig,
)
from chatrouter.governance.circuit_breaker import BreakerState, CircuitBreakerRegistry
from chatrouter.governance.load import ModelLoadTracker
from chatrouter.governance.quota import QuotaManager
from chatrouter.governance.rate_limit import RateLimiter
from chatrouter.storage.memory import MemoryStorage


async def fresh_storage() -> MemoryStorage:
    storage = MemoryStorage("test")
    await storage.start()
    return storage


class TestRateLimiter:
    async def test_requests_allowed_under_limit(self):
        storage = await fresh_storage()
        limiter = RateLimiter(storage)
        tenant = TenantConfig(id="t", rate_limit=RateLimitConfig(rpm=5))
        for _ in range(5):
            verdict, _ = await limiter.check_and_consume(tenant, 0)
            assert verdict.allowed

    async def test_rpm_limit_enforced(self):
        storage = await fresh_storage()
        limiter = RateLimiter(storage)
        tenant = TenantConfig(id="t", rate_limit=RateLimitConfig(rpm=3))
        for _ in range(3):
            verdict, _ = await limiter.check_and_consume(tenant, 0)
            assert verdict.allowed
        verdict, _ = await limiter.check_and_consume(tenant, 0)
        assert not verdict.allowed
        assert "RPM" in (verdict.reason or "")

    async def test_tpm_limit_enforced(self):
        storage = await fresh_storage()
        limiter = RateLimiter(storage)
        tenant = TenantConfig(id="t", rate_limit=RateLimitConfig(tpm=1000))
        verdict, _ = await limiter.check_and_consume(tenant, 900)
        assert verdict.allowed
        verdict, _ = await limiter.check_and_consume(tenant, 500)
        assert not verdict.allowed

    async def test_rejected_request_does_not_consume_budget(self):
        """A rejected call must roll back its own reservation."""
        storage = await fresh_storage()
        limiter = RateLimiter(storage)
        tenant = TenantConfig(id="t", rate_limit=RateLimitConfig(tpm=1000))
        await limiter.check_and_consume(tenant, 900)
        await limiter.check_and_consume(tenant, 500)  # rejected
        # The rejected 500 must not be counted, leaving room for 100.
        verdict, _ = await limiter.check_and_consume(tenant, 100)
        assert verdict.allowed

    async def test_concurrency_limit_enforced(self):
        storage = await fresh_storage()
        limiter = RateLimiter(storage)
        tenant = TenantConfig(id="t", rate_limit=RateLimitConfig(max_concurrency=2))
        assert (await limiter.check_and_consume(tenant, 0))[0].allowed
        assert (await limiter.check_and_consume(tenant, 0))[0].allowed
        assert not (await limiter.check_and_consume(tenant, 0))[0].allowed
        await limiter.release(tenant)
        assert (await limiter.check_and_consume(tenant, 0))[0].allowed

    async def test_token_reconciliation(self):
        storage = await fresh_storage()
        limiter = RateLimiter(storage)
        tenant = TenantConfig(id="t", rate_limit=RateLimitConfig(tpm=1000))
        await limiter.check_and_consume(tenant, 800)
        # Actual usage was much lower; the budget must be returned.
        await limiter.reconcile_tokens(tenant, 800, 100)
        verdict, _ = await limiter.check_and_consume(tenant, 800)
        assert verdict.allowed

    async def test_no_limits_always_allows(self):
        storage = await fresh_storage()
        limiter = RateLimiter(storage)
        tenant = TenantConfig(id="t")
        for _ in range(100):
            assert (await limiter.check_and_consume(tenant, 10_000))[0].allowed

    async def test_concurrent_requests_respect_limit(self):
        """Parallel callers must not exceed the configured ceiling."""
        storage = await fresh_storage()
        limiter = RateLimiter(storage)
        tenant = TenantConfig(id="t", rate_limit=RateLimitConfig(rpm=10))
        results = await asyncio.gather(
            *(limiter.check_and_consume(tenant, 0) for _ in range(25))
        )
        allowed = sum(1 for verdict, _ in results if verdict.allowed)
        assert allowed == 10


class TestQuota:
    async def test_within_quota_allowed(self):
        storage = await fresh_storage()
        quotas = QuotaManager(storage)
        tenant = TenantConfig(id="t", quota=QuotaConfig(max_requests=10))
        assert (await quotas.check(tenant, 100)).allowed

    async def test_request_quota_exhausted_rejects(self):
        storage = await fresh_storage()
        quotas = QuotaManager(storage)
        tenant = TenantConfig(
            id="t", quota=QuotaConfig(max_requests=2, on_exceed="reject")
        )
        await quotas.record(tenant, 10, 0.0)
        await quotas.record(tenant, 10, 0.0)
        verdict = await quotas.check(tenant, 10)
        assert not verdict.allowed

    async def test_cost_quota_downgrade(self):
        storage = await fresh_storage()
        quotas = QuotaManager(storage)
        tenant = TenantConfig(
            id="t", quota=QuotaConfig(max_cost_usd=1.0, on_exceed="downgrade")
        )
        await quotas.record(tenant, 1000, 1.5)
        verdict = await quotas.check(tenant, 10)
        assert verdict.allowed
        assert verdict.downgrade

    async def test_token_quota_considers_projection(self):
        storage = await fresh_storage()
        quotas = QuotaManager(storage)
        tenant = TenantConfig(id="t", quota=QuotaConfig(max_tokens=1000))
        await quotas.record(tenant, 900, 0.0)
        assert not (await quotas.check(tenant, 500)).allowed
        assert (await quotas.check(tenant, 50)).allowed

    async def test_no_quota_configured_allows(self):
        storage = await fresh_storage()
        quotas = QuotaManager(storage)
        tenant = TenantConfig(id="t")
        assert (await quotas.check(tenant, 10**9)).allowed


class TestCircuitBreaker:
    def test_starts_closed(self):
        breakers = CircuitBreakerRegistry(CircuitBreakerConfig())
        assert breakers.allows("m")
        assert breakers.state("m") is BreakerState.CLOSED

    def test_opens_after_consecutive_failures(self):
        breakers = CircuitBreakerRegistry(CircuitBreakerConfig(failure_threshold=3))
        for _ in range(3):
            breakers.record_failure("m")
        assert breakers.state("m") is BreakerState.OPEN
        assert not breakers.allows("m")

    def test_success_resets_consecutive_counter(self):
        breakers = CircuitBreakerRegistry(CircuitBreakerConfig(failure_threshold=3))
        breakers.record_failure("m")
        breakers.record_failure("m")
        breakers.record_success("m")
        breakers.record_failure("m")
        assert breakers.state("m") is BreakerState.CLOSED

    def test_half_open_after_cooldown(self):
        breakers = CircuitBreakerRegistry(
            CircuitBreakerConfig(failure_threshold=2, open_seconds=0.01)
        )
        breakers.record_failure("m")
        breakers.record_failure("m")
        assert not breakers.allows("m")
        import time

        time.sleep(0.02)
        assert breakers.allows("m")
        assert breakers.state("m") is BreakerState.HALF_OPEN

    def test_half_open_failure_reopens(self):
        import time

        breakers = CircuitBreakerRegistry(
            CircuitBreakerConfig(failure_threshold=2, open_seconds=0.01)
        )
        breakers.record_failure("m")
        breakers.record_failure("m")
        time.sleep(0.02)
        breakers.allows("m")
        breakers.record_failure("m")
        assert breakers.state("m") is BreakerState.OPEN

    def test_half_open_success_closes(self):
        import time

        breakers = CircuitBreakerRegistry(
            CircuitBreakerConfig(failure_threshold=2, open_seconds=0.01, half_open_max_calls=2)
        )
        breakers.record_failure("m")
        breakers.record_failure("m")
        time.sleep(0.02)
        breakers.allows("m")
        breakers.record_success("m")
        breakers.allows("m")
        breakers.record_success("m")
        assert breakers.state("m") is BreakerState.CLOSED

    def test_disabled_breaker_always_allows(self):
        breakers = CircuitBreakerRegistry(CircuitBreakerConfig(enabled=False))
        for _ in range(100):
            breakers.record_failure("m")
        assert breakers.allows("m")

    def test_failure_rate_threshold(self):
        breakers = CircuitBreakerRegistry(
            CircuitBreakerConfig(
                failure_threshold=100, min_requests=10, failure_rate_threshold=0.5
            )
        )
        for _ in range(6):
            breakers.record_failure("m")
            breakers.record_success("m")
        assert breakers.state("m") is BreakerState.OPEN


class TestLoadTracking:
    async def test_utilisation_reflects_reservations(self):
        from chatrouter.config.models import ModelConfig

        storage = await fresh_storage()
        tracker = ModelLoadTracker(storage, OverflowConfig())
        model = ModelConfig(id="m", provider="p", upstream_model="u", max_rpm=10)
        for _ in range(5):
            await tracker.reserve(model, 100)
        snapshot = await tracker.snapshot(model)
        assert snapshot.rpm_used == 5
        assert 0.4 < snapshot.utilisation < 0.6

    async def test_headroom_detection(self):
        from chatrouter.config.models import ModelConfig

        storage = await fresh_storage()
        tracker = ModelLoadTracker(storage, OverflowConfig())
        model = ModelConfig(id="m", provider="p", upstream_model="u", max_rpm=2)
        await tracker.reserve(model, 0)
        await tracker.reserve(model, 0)
        snapshot = await tracker.snapshot(model)
        assert not snapshot.has_headroom()

    async def test_release_frees_concurrency(self):
        from chatrouter.config.models import ModelConfig

        storage = await fresh_storage()
        tracker = ModelLoadTracker(storage, OverflowConfig())
        model = ModelConfig(id="m", provider="p", upstream_model="u", max_concurrency=1)
        await tracker.reserve(model, 100)
        assert not (await tracker.snapshot(model)).has_headroom()
        await tracker.release(model, 100, 100)
        assert (await tracker.snapshot(model)).has_headroom()

    async def test_saturation_threshold(self):
        from chatrouter.config.models import ModelConfig

        storage = await fresh_storage()
        tracker = ModelLoadTracker(storage, OverflowConfig(saturation_threshold=0.5))
        model = ModelConfig(id="m", provider="p", upstream_model="u", max_rpm=10)
        for _ in range(6):
            await tracker.reserve(model, 0)
        assert tracker.is_saturated(await tracker.snapshot(model))

    async def test_unlimited_model_never_saturates(self):
        from chatrouter.config.models import ModelConfig

        storage = await fresh_storage()
        tracker = ModelLoadTracker(storage, OverflowConfig())
        model = ModelConfig(id="m", provider="p", upstream_model="u")
        for _ in range(1000):
            await tracker.reserve(model, 1000)
        assert not tracker.is_saturated(await tracker.snapshot(model))
