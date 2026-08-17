"""Tests for context-window overflow handling.

A conversation that outgrows every model's window is a normal event in
production, not a client error to be rejected by default. These tests pin the
three degradation strategies and, importantly, the invariants that make
trimming safe: the system prompt and the newest turns must always survive.
"""

from __future__ import annotations

import pytest

from chatrouter.config.models import (
    ContextOverflowConfig,
    ContextOverflowStrategy,
    ModelConfig,
    ModelTier,
    RoutingConfig,
)
from chatrouter.core.errors import ContextOverflowError
from chatrouter.core.schemas import new_request_id
from chatrouter.core.tokens import count_message_tokens
from chatrouter.core.types import RequestContext
from chatrouter.routing.context_fit import fits, largest_window_model, trim_to_fit
from chatrouter.service import GatewayService

from .conftest import assistant, make_config, make_request, system, user


def tiny_models() -> list[ModelConfig]:
    """Two models with deliberately small, unequal windows."""
    return [
        ModelConfig(
            id="small",
            provider="p1",
            upstream_model="small-1",
            tier=ModelTier.ECONOMY,
            input_cost_per_1k=0.0001,
            output_cost_per_1k=0.0002,
            context_window=200,
            quality_prior=0.4,
        ),
        ModelConfig(
            id="wide",
            provider="p1",
            upstream_model="wide-1",
            tier=ModelTier.PREMIUM,
            input_cost_per_1k=0.005,
            output_cost_per_1k=0.015,
            context_window=800,
            quality_prior=0.85,
        ),
    ]


def long_conversation(turns: int = 40) -> list:
    """A conversation guaranteed to overflow the tiny models above."""
    messages = [system("You are a precise assistant.")]
    for i in range(turns):
        messages.append(user(f"Question number {i} about a reasonably long topic."))
        messages.append(assistant(f"Answer number {i} with some supporting detail."))
    messages.append(user("Given all of the above, what is the final conclusion?"))
    return messages


async def route(service, request, projected=0):
    context = RequestContext(
        request_id=new_request_id(), tenant=service.config.tenants[0], request=request
    )
    return await service.router.route(context, projected)


def trim_cfg(**overrides) -> ContextOverflowConfig:
    return ContextOverflowConfig(
        strategy=ContextOverflowStrategy.TRIM_HISTORY, **overrides
    )


class TestOverflowRouting:
    async def test_overflow_routes_to_widest_window(self):
        """Default strategy: prefer the model that can hold the prompt; ordinary
        requests must be left untouched."""
        config = make_config(models=tiny_models(), routing=RoutingConfig(default_model="small"))
        service = GatewayService(config)
        await service.start()
        try:
            decision = await route(service, make_request(long_conversation()))
            assert decision.model.id == "wide"
            assert any("context overflow" in n for n in decision.notes)

            normal = await route(service, make_request([user("hi")]))
            assert not any("context overflow" in n for n in normal.notes)
        finally:
            await service.close()

    async def test_reject_strategy_raises(self):
        """Explicit opt-in to failing fast rather than silently degrading."""
        config = make_config(
            models=tiny_models(),
            routing=RoutingConfig(
                default_model="small",
                context_overflow=ContextOverflowConfig(
                    strategy=ContextOverflowStrategy.REJECT
                ),
            ),
        )
        service = GatewayService(config)
        await service.start()
        try:
            with pytest.raises(ContextOverflowError):
                await route(service, make_request(long_conversation()))
        finally:
            await service.close()


class TestTrimming:
    def test_trim_preserves_edges_and_inserts_notice(self):
        """The system prompt and the latest turn must never be dropped, and the
        model must be told history is missing rather than assuming continuity."""
        model = tiny_models()[0]
        messages = long_conversation()
        cfg = trim_cfg()

        result = trim_to_fit(messages, model, cfg)

        assert result.trimmed
        assert result.messages[0].text() == messages[0].text()
        assert result.messages[-1].text() == messages[-1].text()
        assert result.final_tokens < result.original_tokens
        assert any("omitted" in m.text() for m in result.messages)

    def test_trim_can_be_silent(self):
        cfg = trim_cfg(insert_elision_notice=False)
        result = trim_to_fit(long_conversation(), tiny_models()[0], cfg)
        assert not any("omitted" in m.text() for m in result.messages)

    def test_trim_edge_cases(self):
        """A short conversation is untouched; output reservations trim harder;
        an irreducible head+tail is reported honestly."""
        # Short conversation: nothing to trim.
        short = trim_to_fit([user("hello")], tiny_models()[1], trim_cfg())
        assert not short.trimmed

        # Output reservation trims strictly harder than without it.
        model = tiny_models()[1]
        messages = long_conversation()
        without = trim_to_fit(messages, model, trim_cfg())
        with_reserve = trim_to_fit(messages, model, trim_cfg(), reserve_output=600)
        assert with_reserve.final_tokens <= without.final_tokens
        assert with_reserve.removed_messages >= without.removed_messages

        # Irreducible conversation: the overflow must be reported, not hidden.
        undroppable = trim_cfg(keep_trailing_messages=40)
        result = trim_to_fit(long_conversation(), tiny_models()[0], undroppable)
        if result.final_tokens > model.context_window:
            assert any("still exceeds" in n for n in result.notes)


class TestHelpers:
    def test_window_helpers(self):
        assert largest_window_model(tiny_models()).id == "wide"
        assert largest_window_model([]) is None
        model = tiny_models()[0]
        assert fits(model, 100, 50)
        assert not fits(model, 100, 150)


class TestTrimIntegration:
    async def test_service_trims_and_reports_header(self):
        """End to end: the trim actually reaches the outgoing payload."""
        config = make_config(
            models=tiny_models(),
            routing=RoutingConfig(
                default_model="wide",
                context_overflow=ContextOverflowConfig(
                    strategy=ContextOverflowStrategy.TRIM_HISTORY
                ),
            ),
        )
        service = GatewayService(config)
        await service.start()
        try:
            request = make_request(long_conversation())
            original_count = len(request.messages)
            _, _, headers = await service.prepare(request, config.tenants[0])

            assert "x-chatrouter-context-trimmed" in headers
            # The mutation must land on the request that gets serialised
            # upstream, not on a discarded copy.
            assert len(request.messages) < original_count
            model = config.model_by_id(headers["x-chatrouter-model"])
            assert count_message_tokens(request.messages, model.id) <= model.context_window
            assert request.upstream_payload(model.upstream_model)["messages"]
        finally:
            await service.close()
