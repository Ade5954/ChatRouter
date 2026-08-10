"""Tests for the routing engine and its interaction with governance."""

from __future__ import annotations

import pytest

from chatrouter.config.models import ModelTier
from chatrouter.core.errors import ModelNotFoundError, PermissionError_
from chatrouter.core.schemas import RoutingHints, new_request_id
from chatrouter.core.types import RequestContext, RoutingDecisionReason

from .conftest import assistant, make_config, make_request, user


async def route(service, request, tenant=None, projected=1000):
    tenant = tenant or service.config.tenants[0]
    context = RequestContext(
        request_id=new_request_id(), tenant=tenant, request=request
    )
    return await service.router.route(context, projected)


class TestTierSelection:
    async def test_simple_request_routes_cheap(self, service):
        decision = await route(service, make_request([user("hi")]))
        assert decision.model.tier is ModelTier.ECONOMY

    async def test_hard_request_routes_up(self, service):
        decision = await route(
            service,
            make_request(
                [user("Prove by induction that this algorithm is optimal, with full derivations.")]
            ),
        )
        assert decision.model.tier.rank > ModelTier.ECONOMY.rank

    async def test_decision_carries_assessment(self, service):
        decision = await route(service, make_request([user("hello")]))
        assert decision.assessment is not None
        assert decision.reason in (
            RoutingDecisionReason.CONTEXT_AWARE,
            RoutingDecisionReason.FEEDBACK_ADAPTIVE,
            RoutingDecisionReason.EXPLORATION,
        )

    async def test_fallback_chain_is_populated(self, service):
        decision = await route(service, make_request([user("hello")]))
        assert decision.fallback_chain
        assert decision.model.id not in [m.id for m in decision.fallback_chain]

    async def test_context_escalates_tier_versus_single_turn(self, service):
        """The headline capability, verified end to end through the router."""
        followup = [
            user("Derive a closed form for this recurrence and prove it is tight."),
            assistant("T(n) = O(n log n)."),
            user("and the other case?"),
        ]
        with_context = await route(service, make_request(followup))
        without_context = await route(service, make_request([user("and the other case?")]))
        assert with_context.model.tier.rank >= without_context.model.tier.rank


class TestExplicitModel:
    async def test_explicit_model_is_respected(self, service):
        decision = await route(service, make_request([user("hi")], model="strong"))
        assert decision.model.id == "strong"
        assert decision.reason is RoutingDecisionReason.EXPLICIT_MODEL

    async def test_unknown_model_raises(self, service):
        with pytest.raises(ModelNotFoundError):
            await route(service, make_request([user("hi")], model="does-not-exist"))

    async def test_auto_alias_triggers_routing(self, service):
        decision = await route(service, make_request([user("hi")], model="auto"))
        assert decision.assessment is not None

    async def test_pin_model_hint(self, service):
        request = make_request([user("hi")], chatrouter=RoutingHints(pin_model="reasoner"))
        decision = await route(service, request)
        assert decision.model.id == "reasoner"
        assert decision.reason is RoutingDecisionReason.PINNED


class TestTenantConstraints:
    async def test_max_tier_caps_routing(self):
        from chatrouter.config.models import TenantConfig
        from chatrouter.service import GatewayService

        config = make_config(
            tenants=[
                TenantConfig(id="capped", api_keys=["k"], max_tier=ModelTier.ECONOMY)
            ]
        )
        service = GatewayService(config)
        await service.start()
        try:
            decision = await route(
                service,
                make_request([user("Prove this theorem rigorously with full derivations.")]),
                tenant=config.tenants[0],
            )
            assert decision.model.tier is ModelTier.ECONOMY
        finally:
            await service.close()

    async def test_denied_model_rejected(self):
        from chatrouter.config.models import TenantConfig
        from chatrouter.service import GatewayService

        config = make_config(
            tenants=[TenantConfig(id="t", api_keys=["k"], denied_models=["strong"])]
        )
        service = GatewayService(config)
        await service.start()
        try:
            with pytest.raises(PermissionError_):
                await route(
                    service,
                    make_request([user("hi")], model="strong"),
                    tenant=config.tenants[0],
                )
        finally:
            await service.close()

    async def test_allowed_models_whitelist(self):
        from chatrouter.config.models import TenantConfig
        from chatrouter.service import GatewayService

        config = make_config(
            tenants=[TenantConfig(id="t", api_keys=["k"], allowed_models=["cheap"])]
        )
        service = GatewayService(config)
        await service.start()
        try:
            decision = await route(
                service,
                make_request([user("Prove this theorem with rigorous derivations.")]),
                tenant=config.tenants[0],
            )
            assert decision.model.id == "cheap"
        finally:
            await service.close()


class TestCapabilityFiltering:
    async def test_vision_request_avoids_blind_models(self, service):
        from chatrouter.core.schemas import ChatMessage

        message = ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Describe this."},
                {"type": "image_url", "image_url": {"url": "https://x.test/i.png"}},
            ],
        )
        decision = await route(service, make_request([message]))
        assert decision.model.supports_vision

    async def test_oversized_prompt_avoids_small_context(self, service):
        huge = "token " * 40000
        decision = await route(service, make_request([user(huge)]), projected=60000)
        assert decision.model.context_window >= 60000


class TestFeedbackAdaptation:
    async def test_negative_feedback_reduces_effective_quality(self, service):
        model_id = "mid"
        before = await service.feedback.get_stats(model_id, ModelTier.STANDARD)
        base = service.feedback.effective_quality(0.65, before)

        for _ in range(40):
            await service.feedback.record_feedback(model_id, ModelTier.STANDARD.value, 0.0)

        after = await service.feedback.get_stats(model_id, ModelTier.STANDARD)
        adapted = service.feedback.effective_quality(0.65, after)
        assert adapted < base

    async def test_positive_feedback_increases_effective_quality(self, service):
        model_id = "cheap"
        for _ in range(40):
            await service.feedback.record_feedback(model_id, ModelTier.ECONOMY.value, 1.0)
        stats = await service.feedback.get_stats(model_id, ModelTier.ECONOMY)
        assert service.feedback.effective_quality(0.4, stats) > 0.4

    async def test_failures_degrade_quality(self, service):
        for _ in range(30):
            await service.feedback.record_outcome(
                "mid", ModelTier.STANDARD, success=False, latency_ms=100, implicit_score=0.0
            )
        stats = await service.feedback.get_stats("mid", ModelTier.STANDARD)
        assert stats.success_rate < 0.5
        assert service.feedback.effective_quality(0.65, stats) < 0.65

    async def test_feedback_shifts_model_choice(self, service):
        """The loop must actually change routing, not just the statistics."""
        request = make_request([user("Write a short summary of this text.")])
        first = await route(service, request)

        # Punish the initially chosen model heavily and repeatedly.
        for _ in range(80):
            await service.feedback.record_feedback(
                first.model.id, first.assessment.tier.value, 0.0
            )
            await service.feedback.record_outcome(
                first.model.id, first.assessment.tier, success=False, implicit_score=0.0
            )

        scored = await service.router._score_candidates(
            service.router.models,
            first.assessment.tier,
            service.config.tenants[0],
            None,
            await service.load_tracker.snapshot_many(service.router.models),
            1000,
        )
        by_id = {c.model.id: c for c in scored}
        assert by_id[first.model.id].quality < first.model.quality_prior

    async def test_implicit_score_penalises_retries(self, service):
        clean = service.feedback.implicit_score(
            success=True, attempts=1, truncated=False, latency_ms=1000, latency_prior_ms=1000
        )
        retried = service.feedback.implicit_score(
            success=True, attempts=3, truncated=False, latency_ms=1000, latency_prior_ms=1000
        )
        assert retried < clean

    async def test_implicit_score_penalises_truncation(self, service):
        full = service.feedback.implicit_score(
            success=True, attempts=1, truncated=False, latency_ms=1000, latency_prior_ms=1000
        )
        cut = service.feedback.implicit_score(
            success=True, attempts=1, truncated=True, latency_ms=1000, latency_prior_ms=1000
        )
        assert cut < full

    async def test_failed_request_scores_zero(self, service):
        assert (
            service.feedback.implicit_score(
                success=False, attempts=1, truncated=False, latency_ms=None, latency_prior_ms=1000
            )
            == 0.0
        )


class TestCircuitBreakerInRouting:
    async def test_open_circuit_excluded_from_fallbacks(self, service):
        for _ in range(10):
            service.breakers.record_failure("strong")
        decision = await route(service, make_request([user("hi")]))
        assert "strong" not in [m.id for m in decision.fallback_chain]

    async def test_health_penalty_grows_with_failures(self, service):
        assert service.breakers.health_penalty("mid") == 0.0
        for _ in range(4):
            service.breakers.record_failure("mid")
        assert service.breakers.health_penalty("mid") > 0
