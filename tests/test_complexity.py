"""Tests for full-conversation complexity analysis.

The central claim of the gateway is that context-aware scoring beats
single-turn scoring; these tests encode that claim as executable assertions.
"""

from __future__ import annotations

from chatrouter.config.models import ContextRoutingConfig, ModelTier, TierThresholds
from chatrouter.routing.complexity import ComplexityAnalyzer

from .conftest import assistant, make_request, system, user


def analyzer(**ctx_overrides) -> ComplexityAnalyzer:
    return ComplexityAnalyzer(ContextRoutingConfig(**ctx_overrides), TierThresholds())


class TestSingleTurn:
    def test_greeting_scores_low(self):
        result = analyzer().analyse(make_request([user("hi")]))
        assert result.tier is ModelTier.ECONOMY
        assert result.score < 0.25

    def test_chinese_greeting_scores_low(self):
        result = analyzer().analyse(make_request([user("你好")]))
        assert result.tier is ModelTier.ECONOMY

    def test_proof_request_scores_high(self):
        result = analyzer().analyse(
            make_request([user("Prove by induction that the sum of the first n odd numbers is n^2.")])
        )
        assert result.score > 0.25
        assert result.signals.reasoning_keywords >= 0.9

    def test_chinese_reasoning_detected(self):
        result = analyzer().analyse(make_request([user("请推导这个定理并给出完整证明过程")]))
        assert result.signals.reasoning_keywords >= 0.9

    def test_code_block_raises_code_signal(self):
        text = "Why does this fail?\n```python\ndef f(x):\n    return x / 0\n```"
        result = analyzer().analyse(make_request([user(text)]))
        assert result.signals.code_content > 0.3


class TestContextAwareness:
    """The differentiating capability: history changes the decision."""

    def test_terse_followup_inherits_earlier_complexity(self):
        """A short follow-up must not be scored as if it were trivial."""
        hard_thread = [
            user("Derive the time complexity of this recursive algorithm and prove it is optimal."),
            assistant("The recurrence is T(n) = 2T(n/2) + O(n), giving O(n log n)."),
            user("and the other case?"),
        ]
        contextual = analyzer().analyse(make_request(hard_thread))
        isolated = analyzer().analyse(make_request([user("and the other case?")]))

        assert contextual.score > isolated.score
        assert contextual.latent_escalation > 0

    def test_system_prompt_requirement_is_remembered(self):
        """A hard requirement stated up-front survives many later turns."""
        messages = [
            system("Every answer must be a rigorous mathematical proof with full derivations."),
            user("What is 2+2?"),
            assistant("4"),
            user("And 3+3?"),
        ]
        result = analyzer().analyse(make_request(messages))
        assert result.signals.reasoning_keywords > 0.3

    def test_escalation_memory_can_be_disabled(self):
        messages = [
            user("Prove this theorem by induction."),
            assistant("Here is the proof."),
            user("ok"),
        ]
        with_memory = analyzer(escalation_memory=0.9).analyse(make_request(messages))
        without_memory = analyzer(escalation_memory=0.0).analyse(make_request(messages))
        assert with_memory.score > without_memory.score

    def test_repeated_dissatisfaction_escalates(self):
        """Failed attempts are the strongest signal that the tier is too low."""
        messages = [
            user("Fix this bug."),
            assistant("Try changing the loop bound."),
            user("still wrong"),
            assistant("Then adjust the index."),
            user("that's incorrect, it still fails"),
        ]
        result = analyzer().analyse(make_request(messages))
        assert result.signals.unresolved_thread > 0.5
        assert result.tier.rank >= ModelTier.STANDARD.rank

    def test_chinese_dissatisfaction_detected(self):
        messages = [
            user("帮我修复这个问题"),
            assistant("试试修改循环边界"),
            user("还是不对，依然报错"),
        ]
        result = analyzer().analyse(make_request(messages))
        assert result.signals.unresolved_thread > 0

    def test_conversation_depth_increases_score(self):
        shallow = [user("Summarise this paragraph.")]
        deep = []
        for i in range(8):
            deep.append(user(f"Now refine constraint {i} while keeping all previous ones."))
            deep.append(assistant(f"Refined {i}."))
        shallow_result = analyzer().analyse(make_request(shallow))
        deep_result = analyzer().analyse(make_request(deep))
        assert deep_result.signals.conversation_depth > shallow_result.signals.conversation_depth

    def test_trivial_shortcut_not_applied_to_deep_thread(self):
        """'ok' inside a hard debugging thread must not drop to economy."""
        messages = [
            user("Debug this failing distributed transaction and prove correctness."),
            assistant("Here's the analysis..."),
            user("still wrong"),
            assistant("Another approach..."),
            user("ok"),
        ]
        result = analyzer().analyse(make_request(messages))
        assert result.tier.rank > ModelTier.ECONOMY.rank

    def test_greeting_in_shallow_thread_still_cheap(self):
        result = analyzer().analyse(make_request([user("hello")]))
        assert result.tier is ModelTier.ECONOMY


class TestCapabilityDetection:
    def test_tools_flagged(self):
        request = make_request(
            [user("What is the weather?")],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        result = analyzer().analyse(request)
        assert result.requires_tools
        assert result.signals.tool_usage > 0

    def test_vision_flagged(self):
        from chatrouter.core.schemas import ChatMessage

        message = ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
            ],
        )
        result = analyzer().analyse(make_request([message]))
        assert result.requires_vision

    def test_context_pressure_from_long_prompt(self):
        long_text = "word " * 20000
        result = analyzer().analyse(make_request([user(long_text)]), context_window_hint=32000)
        assert result.signals.context_length > 0.5

    def test_structured_output_detected(self):
        result = analyzer().analyse(make_request([user("Return the result as strict JSON.")]))
        assert result.signals.structured_output > 0.5


class TestScoreBounds:
    def test_score_always_in_range(self):
        cases = [
            [user("")],
            [user("hi")],
            [user("Prove " * 500)],
            [user("```" * 50)],
        ]
        for messages in cases:
            result = analyzer().analyse(make_request(messages))
            assert 0.0 <= result.score <= 1.0

    def test_empty_messages_handled(self):
        result = analyzer().analyse(make_request([]))
        assert result.score == 0.0
        assert result.turn_count == 0

    def test_keyword_stuffing_does_not_saturate(self):
        """Repeating one keyword must not push the score to the maximum."""
        stuffed = "prove " * 200
        result = analyzer().analyse(make_request([user(stuffed)]))
        assert result.signals.reasoning_keywords <= 1.0
        assert result.score < 0.95
