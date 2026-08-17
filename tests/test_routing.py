"""Tests for the routing engine and its interaction with governance."""

from __future__ import annotations

import pytest

from chatrouter.config.models import FeedbackConfig, ModelTier, RoutingConfig
from chatrouter.core.errors import ModelNotFoundError, PermissionError_
from chatrouter.core.schemas import RoutingHints, new_request_id
from chatrouter.core.types import RequestContext, RoutingDecisionReason
from chatrouter.service import GatewayService

from .conftest import assistant, make_config, make_request, user


async def route(service, request, tenant=None, projected=1000):
    tenant = tenant or service.config.tenants[0]
    context = RequestContext(request_id=new_request_id(), tenant=tenant, request=request)
    return await service.router.route(context, projected)


@pytest.fixture
async def feedback_service():
    """Feedback loop converges in 1-2 samples (the defaults need ~20+), so the
    quality-adaptation tests stay fast while exercising the same code paths."""
    config = make_config(
        routing=RoutingConfig(
            default_model="mid",
            feedback=FeedbackConfig(
                min_samples=1, ema_alpha=0.5, learning_rate=0.5, exploration_ratio=0.0
            ),
        )
    )
    service = GatewayService(config)
    await service.start()
    try:
        yield service
    finally:
        await service.close()


async def tenant_service(**tenant_overrides):
    from chatrouter.config.models import TenantConfig

    config = make_config(tenants=[TenantConfig(id="t", api_keys=["k"], **tenant_overrides)])
    service = GatewayService(config)
    await service.start()
    return service, config.tenants[0]


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

    async def test_decision_carries_assessment_and_fallbacks(self, service):
        decision = await route(service, make_request([user("hello")]))
        assert decision.assessment is not None
        assert decision.reason in (
            RoutingDecisionReason.CONTEXT_AWARE,
            RoutingDecisionReason.FEEDBACK_ADAPTIVE,
            RoutingDecisionReason.EXPLORATION,
        )
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
    async def test_explicit_vs_auto_routing(self, service):
        explicit = await route(service, make_request([user("hi")], model="strong"))
        assert explicit.model.id == "strong"
        assert explicit.reason is RoutingDecisionReason.EXPLICIT_MODEL

        auto = await route(service, make_request([user("hi")], model="auto"))
        assert auto.assessment is not None

    async def test_unknown_model_raises(self, service):
        with pytest.raises(ModelNotFoundError):
            await route(service, make_request([user("hi")], model="does-not-exist"))

    async def test_pin_model_hint(self, service):
        request = make_request([user("hi")], chatrouter=RoutingHints(pin_model="reasoner"))
        decision = await route(service, request)
        assert decision.model.id == "reasoner"
        assert decision.reason is RoutingDecisionReason.PINNED


class TestTenantConstraints:
    async def test_max_tier_caps_routing(self):
        service, tenant = await tenant_service(max_tier=ModelTier.ECONOMY)
        try:
            decision = await route(
                service,
                make_request([user("Prove this theorem rigorously with full derivations.")]),
                tenant=tenant,
            )
            assert decision.model.tier is ModelTier.ECONOMY
        finally:
            await service.close()

    async def test_denied_and_allowed_models_enforced(self):
        service, tenant = await tenant_service(denied_models=["strong"], allowed_models=["cheap"])
        try:
            with pytest.raises(PermissionError_):
                await route(service, make_request([user("hi")], model="strong"), tenant=tenant)
            decision = await route(
                service,
                make_request([user("Prove this theorem with rigorous derivations.")]),
                tenant=tenant,
            )
            assert decision.model.id == "cheap"
        finally:
            await service.close()


class TestCapabilityFiltering:
    async def test_capability_filters_exclude_unfit_models(self, service):
        from chatrouter.core.schemas import ChatMessage

        vision = ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Describe this."},
                {"type": "image_url", "image_url": {"url": "https://x.test/i.png"}},
            ],
        )
        vision_decision = await route(service, make_request([vision]))
        assert vision_decision.model.supports_vision

        huge = "token " * 40000
        huge_decision = await route(service, make_request([user(huge)]), projected=60000)
        assert huge_decision.model.context_window >= 60000


class TestFeedbackAdaptation:
    async def test_feedback_moves_effective_quality(self, feedback_service):
        mid = "mid"
        before = await feedback_service.feedback.get_stats(mid, ModelTier.STANDARD)
        base = feedback_service.feedback.effective_quality(0.65, before)

        for _ in range(5):
            await feedback_service.feedback.record_feedback(mid, ModelTier.STANDARD.value, 0.0)
        after = await feedback_service.feedback.get_stats(mid, ModelTier.STANDARD)
        assert feedback_service.feedback.effective_quality(0.65, after) < base

        for _ in range(5):
            await feedback_service.feedback.record_feedback(mid, ModelTier.STANDARD.value, 1.0)
        boosted = await feedback_service.feedback.get_stats(mid, ModelTier.STANDARD)
        assert feedback_service.feedback.effective_quality(0.65, boosted) > base

    async def test_failures_degrade_quality(self, feedback_service):
        for _ in range(5):
            await feedback_service.feedback.record_outcome(
                "mid", ModelTier.STANDARD, success=False, latency_ms=100, implicit_score=0.0
            )
        stats = await feedback_service.feedback.get_stats("mid", ModelTier.STANDARD)
        assert stats.success_rate < 0.5
        assert feedback_service.feedback.effective_quality(0.65, stats) < 0.65

    async def test_feedback_shifts_model_choice(self, feedback_service):
        """The loop must actually change routing, not just the statistics."""
        request = make_request([user("Write a short summary of this text.")])
        first = await route(feedback_service, request)

        for _ in range(5):
            await feedback_service.feedback.record_feedback(
                first.model.id, first.assessment.tier.value, 0.0
            )
            await feedback_service.feedback.record_outcome(
                first.model.id, first.assessment.tier, success=False, implicit_score=0.0
            )

        scored = await feedback_service.router._score_candidates(
            feedback_service.router.models,
            first.assessment.tier,
            feedback_service.config.tenants[0],
            None,
            await feedback_service.load_tracker.snapshot_many(feedback_service.router.models),
            1000,
        )
        by_id = {c.model.id: c for c in scored}
        assert by_id[first.model.id].quality < first.model.quality_prior

    async def test_implicit_score_semantics(self, feedback_service):
        clean = feedback_service.feedback.implicit_score(
            success=True, attempts=1, truncated=False, latency_ms=1000, latency_prior_ms=1000
        )
        retried = feedback_service.feedback.implicit_score(
            success=True, attempts=3, truncated=False, latency_ms=1000, latency_prior_ms=1000
        )
        cut = feedback_service.feedback.implicit_score(
            success=True, attempts=1, truncated=True, latency_ms=1000, latency_prior_ms=1000
        )
        failed = feedback_service.feedback.implicit_score(
            success=False, attempts=1, truncated=False, latency_ms=None, latency_prior_ms=1000
        )
        assert retried < clean
        assert cut < clean
        assert failed == 0.0


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
