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


class TestOverflowRouting:
    async def test_overflow_routes_to_widest_window(self):
        """Default strategy: prefer the model that can actually hold the prompt."""
        config = make_config(
            models=tiny_models(),
            routing=RoutingConfig(default_model="small"),
        )
        service = GatewayService(config)
        await service.start()
        try:
            decision = await route(service, make_request(long_conversation()))
            assert decision.model.id == "wide"
            assert any("context overflow" in n for n in decision.notes)
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

    async def test_normal_conversation_is_untouched(self):
        """Overflow handling must not perturb ordinary requests."""
        config = make_config(models=tiny_models(), routing=RoutingConfig(default_model="small"))
        service = GatewayService(config)
        await service.start()
        try:
            decision = await route(service, make_request([user("hi")]))
            assert not any("context overflow" in n for n in decision.notes)
        finally:
            await service.close()


class TestTrimming:
    def test_trim_preserves_system_and_latest_turn(self):
        """The two things that must never be dropped.

        Losing the system prompt changes the model's instructions; losing the
        last user turn changes the question being asked. Either would silently
        answer something other than what was requested.
        """
        model = tiny_models()[0]
        messages = long_conversation()
        cfg = ContextOverflowConfig(strategy=ContextOverflowStrategy.TRIM_HISTORY)

        result = trim_to_fit(messages, model, cfg)

        assert result.trimmed
        assert result.messages[0].text() == messages[0].text()
        assert result.messages[-1].text() == messages[-1].text()

    def test_trim_reduces_token_count(self):
        model = tiny_models()[0]
        messages = long_conversation()
        cfg = ContextOverflowConfig(strategy=ContextOverflowStrategy.TRIM_HISTORY)

        result = trim_to_fit(messages, model, cfg)

        assert result.final_tokens < result.original_tokens
        assert len(result.messages) < len(messages)

    def test_trim_inserts_elision_notice(self):
        """The model must know history is missing, or it may assume continuity."""
        model = tiny_models()[0]
        cfg = ContextOverflowConfig(strategy=ContextOverflowStrategy.TRIM_HISTORY)

        result = trim_to_fit(long_conversation(), model, cfg)

        assert any("omitted" in m.text() for m in result.messages)

    def test_trim_can_be_silent(self):
        cfg = ContextOverflowConfig(
            strategy=ContextOverflowStrategy.TRIM_HISTORY, insert_elision_notice=False
        )
        result = trim_to_fit(long_conversation(), tiny_models()[0], cfg)
        assert not any("omitted" in m.text() for m in result.messages)

    def test_short_conversation_is_not_trimmed(self):
        cfg = ContextOverflowConfig(strategy=ContextOverflowStrategy.TRIM_HISTORY)
        messages = [user("hello")]
        result = trim_to_fit(messages, tiny_models()[1], cfg)
        assert not result.trimmed
        assert len(result.messages) == 1

    def test_trim_respects_output_reservation(self):
        """Reserving output space must tighten the trim budget.

        Trimming is best-effort: the protected system prompt and recent turns
        are irreducible, so a large reservation may still not be satisfiable.
        What is guaranteed is that a reservation trims strictly harder, and
        that an unmet budget is reported rather than hidden.
        """
        model = tiny_models()[1]
        cfg = ContextOverflowConfig(strategy=ContextOverflowStrategy.TRIM_HISTORY)
        messages = long_conversation()

        without = trim_to_fit(messages, model, cfg)
        with_reserve = trim_to_fit(messages, model, cfg, reserve_output=600)

        assert with_reserve.final_tokens <= without.final_tokens
        assert with_reserve.removed_messages >= without.removed_messages

        budget = model.context_window - 600
        if with_reserve.final_tokens > budget:
            assert any("still exceeds" in n for n in with_reserve.notes)

    def test_undroppable_conversation_is_reported(self):
        """When the protected head+tail alone overflow, say so rather than
        returning a result that quietly still does not fit."""
        model = tiny_models()[0]
        cfg = ContextOverflowConfig(
            strategy=ContextOverflowStrategy.TRIM_HISTORY,
            keep_trailing_messages=40,
        )
        result = trim_to_fit(long_conversation(), model, cfg)
        if result.final_tokens > model.context_window:
            assert any("still exceeds" in n for n in result.notes)


class TestHelpers:
    def test_largest_window_model(self):
        assert largest_window_model(tiny_models()).id == "wide"

    def test_largest_window_of_empty_list(self):
        assert largest_window_model([]) is None

    def test_fits_accounts_for_output_reserve(self):
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
