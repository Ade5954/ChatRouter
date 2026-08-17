"""Tests for session-level cache affinity.

Frequent model switching across a multi-turn conversation shatters the upstream
prefix cache. Affinity keeps a session on one model unless complexity drifts
past ``max_drift_tiers``, and encodes the cache-saving into the utility score.

All tests use an isolated GatewayService with exploration disabled so the
behaviour is deterministic and does not depend on the global random sequence.
"""

from __future__ import annotations

from chatrouter.config.models import RoutingConfig, SessionAffinityConfig
from chatrouter.core.schemas import new_request_id
from chatrouter.core.types import RequestContext
from chatrouter.routing.router import Router
from chatrouter.service import GatewayService
from chatrouter.storage.base import Storage
from chatrouter.storage.memory import MemoryStorage

from .conftest import make_config, make_request, user

# Exploration is disabled in these tests so routing is fully deterministic.
_AFFINITY_CONFIG = dict(
    routing=RoutingConfig(
        default_model="mid",
        session_affinity=SessionAffinityConfig(enabled=True, stickiness=0.4),
    )
)


def _ctx(service, request, session_id=None):
    return RequestContext(
        request_id=new_request_id(),
        tenant=service.config.tenants[0],
        request=request,
        session_id=session_id,
    )


async def _route(service, request, session_id=None):
    return await service.router.route(_ctx(service, request, session_id), 1000)


class TestSessionAffinity:
    async def test_affinity_sticks_and_persists(self):
        """A simpler follow-up sticks to the session's existing model, and the
        binding is actually persisted for the next turn to reuse."""
        config = make_config(**_AFFINITY_CONFIG)
        svc = GatewayService(config)
        await svc.start()
        try:
            first = await _route(svc, make_request([user("Explain recursion.")]), "sess-a")
            second = await _route(svc, make_request([user("ok thanks")]), "sess-a")
            assert second.model.id == first.model.id
            assert second.reason.value == "session_affinity"

            # Read straight from storage to prove the write happened.
            assert await svc.storage.get_session_model("sess-a") == first.model.id
        finally:
            await svc.close()

    async def test_drift_beyond_max_tier_overrides_affinity(self):
        """A sudden hard task upgrades even if it breaks affinity."""
        config = make_config(**_AFFINITY_CONFIG)
        svc = GatewayService(config)
        await svc.start()
        try:
            await _route(svc, make_request([user("Hello")]), "sess-c")
            hard = await _route(
                svc,
                make_request(
                    [user("Prove by induction this algorithm is optimal, with full derivations.")]
                ),
                "sess-c",
            )
            # A trivial turn maps to economy/cheap; a proof maps to a higher tier,
            # so affinity must yield to the real complexity.
            assert hard.model.tier.rank > 0
        finally:
            await svc.close()

    async def test_disabled_affinity_does_not_stick(self):
        config = make_config(
            routing=RoutingConfig(
                default_model="mid",
                session_affinity=SessionAffinityConfig(enabled=False),
            )
        )
        svc = GatewayService(config)
        await svc.start()
        try:
            first = await _route(svc, make_request([user("Hello")]), "sess-off")
            second = await _route(svc, make_request([user("ok")]), "sess-off")
            # Without affinity, nothing forces the two turns together.
            assert second.reason.value != "session_affinity"
            # Storage must not be written when disabled.
            assert await svc.storage.get_session_model("sess-off") is None
            _ = first
        finally:
            await svc.close()

    async def test_stickiness_penalises_switching(self):
        """Non-sticky models score lower than the sticky one under stickiness."""
        first_cfg = make_config(**_AFFINITY_CONFIG)
        svc1 = GatewayService(first_cfg)
        await svc1.start()
        try:
            first = await _route(
                svc1, make_request([user("Summarise this paragraph.")]), "sess-pen"
            )
        finally:
            await svc1.close()

        # Re-run scoring with max stickiness to amplify the cache penalty effect.
        config = make_config(
            routing=RoutingConfig(
                default_model="mid",
                session_affinity=SessionAffinityConfig(enabled=True, stickiness=1.0),
            )
        )
        svc = GatewayService(config)
        await svc.start()
        try:
            await _route(svc, make_request([user("Summarise this paragraph.")]), "sess-pen2")
            assess = svc.router.analyse(make_request([user("Summarise this paragraph.")]))
            # Price the cache loss so the affinity penalty is strong and
            # deterministic: declare the session's model has a 0 cached-input
            # price (the full input-cost gap), over a long historical prefix.
            aff = svc.config.model_by_id(first.model.id)
            aff.cached_input_cost_per_1k = 0.0
            candidates = await svc.router._score_candidates(
                svc.router.models,
                assess.tier,
                svc.config.tenants[0],
                None,
                {},
                1000,
                affinity_model_id=first.model.id,
                affinity_prefix_tokens=1_000_000,
                stickiness=1.0,
            )
            sticky = next(c for c in candidates if c.model.id == first.model.id)
            others = [c for c in candidates if c.model.id != first.model.id]
            assert others
            assert all(sticky.utility > c.utility for c in others)
        finally:
            await svc.close()


class TestSessionModelStorage:
    """Storage contract for session affinity must be implemented everywhere."""

    async def test_memory_backend_roundtrip(self):
        store: Storage = MemoryStorage("chatrouter")
        await store.start()
        try:
            assert await store.get_session_model("x") is None
            await store.set_session_affinity("x", "gpt-4o", 1234, 60)
            # Backwards-compatible view still returns just the model id.
            assert await store.get_session_model("x") == "gpt-4o"
            binding = await store.get_session_affinity("x")
            assert binding.model_id == "gpt-4o"
            assert binding.prefix_tokens == 1234
            assert binding.ttl_remaining > 0
        finally:
            await store.close()

    async def test_base_declares_affinity_methods(self):
        # Guard against accidental interface drift.
        assert hasattr(Storage, "get_session_model")
        assert hasattr(Storage, "get_session_affinity")
        assert hasattr(Storage, "set_session_affinity")
        # Router must accept a storage handle (for affinity read/write).
        import inspect

        params = inspect.signature(Router.__init__).parameters
        assert "storage" in params
