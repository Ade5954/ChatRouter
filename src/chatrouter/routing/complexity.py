"""Full-conversation complexity analysis.

Naive single-turn routers score only the latest user turn, which systematically
under-estimates threads where the hard requirement was stated earlier ("answer
everything below as a formal proof"), where difficulty accumulated across turns
(repeated clarifications, a failing test being debugged), or where the last
message is a terse follow-up ("and the other case?") that inherits all the
difficulty of its context. ChatRouter shares the "look at history" design goal
of research routers such as MTRouter and Router-R1, but reaches it with
explainable rule signals instead of learned history-model embeddings.

This analyser therefore evaluates the *whole* dialogue with recency weighting
plus an explicit escalation memory, so latent complexity is preserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config.models import ContextRoutingConfig, TierThresholds
from ..core.schemas import ChatCompletionRequest, ChatMessage
from ..core.tokens import count_message_tokens, count_tools_tokens
from ..core.types import ComplexityAssessment, ComplexitySignals

# --- Lexical evidence -------------------------------------------------------
# Patterns are deliberately bilingual (English + Chinese) because production
# traffic in this deployment is mixed-language.

_REASONING_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
    # Explicit demands for multi-step reasoning. Noun forms matter as much as
    # verbs: system prompts usually phrase requirements as "a rigorous proof"
    # rather than "prove this".
    (re.compile(r"\b(prove[sd]?|proofs?|deriv(e[sd]?|ation[s]?)|theorems?|lemmas?|induction)\b", re.I), 1.0),
    (re.compile(r"证明|推导|定理|归纳法"), 1.0),
    (re.compile(r"\b(step[- ]by[- ]step|chain[- ]of[- ]thought|reason through)\b", re.I), 0.9),
    (re.compile(r"一步一步|逐步推理|详细推导"), 0.9),
    (re.compile(r"\b(optimi[sz]e|complexity|algorithm|np-hard|time complexity)\b", re.I), 0.7),
    (re.compile(r"复杂度|最优解|算法设计"), 0.7),
    (re.compile(r"\b(architect|design (a|the) system|trade[- ]?offs?|scalab)\w*", re.I), 0.7),
    (re.compile(r"架构设计|技术选型|权衡|可扩展性"), 0.7),
    (re.compile(r"\b(debug|root cause|why does .* fail|stack ?trace)\b", re.I), 0.6),
    (re.compile(r"排查|根因|为什么.*(失败|报错|不对)"), 0.6),
    (re.compile(r"\b(compare|contrast|evaluate|analy[sz]e|assess)\b", re.I), 0.5),
    (re.compile(r"对比|分析|评估|论证"), 0.5),
    (re.compile(r"\b(plan|strategy|roadmap|multi[- ]step)\b", re.I), 0.4),
    (re.compile(r"规划|方案|策略"), 0.4),
)

# Cheap, closed-form requests that should stay on the economy tier.
_TRIVIAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|yes|no|good morning)\W*$", re.I),
    re.compile(r"^\s*(你好|谢谢|好的|嗯|是的|不用了|再见)\W*$"),
    re.compile(r"^\s*(translate|翻译)\b.{0,80}$", re.I),
)

_CODE_FENCE = re.compile(r"```")
_CODE_HINTS = re.compile(
    r"\b(def |class |function |import |from \w+ import|SELECT .* FROM|#include|public static|"
    r"=>|async def|const |let |var |return |\bnull\b|\bvoid\b)",
    re.I,
)
_STACK_TRACE = re.compile(r"(Traceback \(most recent call last\)|Exception in thread|\bat [\w.$]+\(.*\.java:\d+\))")

_STRUCTURED_OUTPUT = re.compile(
    r"\b(json|yaml|xml|csv|markdown table|schema|typescript interface|按照以下格式|输出为)\b", re.I
)
_LONG_OUTPUT = re.compile(
    r"\b(\d{3,}\s*(words|字|tokens)|essay|report|comprehensive|in detail|详细|完整的?文档|长文)\b", re.I
)

# Signals that the previous answer did not resolve the user's need.
_DISSATISFACTION = re.compile(
    r"\b(still (wrong|failing|broken|not)|doesn'?t work|that'?s (wrong|incorrect)|not what i (meant|asked)|"
    r"try again|you misunderstood|again please)\b",
    re.I,
)
_DISSATISFACTION_CN = re.compile(
    r"还是(不对|不行|错|报错)|不对啊|不是这个意思|你理解错了|重新(回答|写|生成)|再试一次|依然(失败|报错)"
)

# Follow-ups that carry no standalone meaning and must inherit context.
_ANAPHORIC = re.compile(
    r"^\s*(and |also |what about|how about|then\??|continue|go on|next|"
    r"那|然后呢?|继续|接着|还有呢?|另一个呢?)",
    re.I,
)

_INSTRUCTION_MARKERS = re.compile(
    r"(^\s*[-*\u2022]\s+|^\s*\d+[.)]\s+|\bmust\b|\bshould\b|\brequire[ds]?\b|\bensure\b|必须|需要|要求)",
    re.I | re.M,
)

_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


@dataclass(slots=True)
class _TurnEvidence:
    """Per-message evidence, before recency weighting is applied."""

    reasoning: float = 0.0
    code: float = 0.0
    structured: float = 0.0
    long_output: float = 0.0
    dissatisfaction: float = 0.0
    instruction_density: float = 0.0
    multilingual: float = 0.0
    trivial: bool = False
    anaphoric: bool = False


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _saturating(value: float, half_point: float) -> float:
    """Map [0, inf) into [0, 1) with ``half_point`` mapping to 0.5."""
    if value <= 0:
        return 0.0
    return value / (value + half_point)


def _analyse_turn(message: ChatMessage) -> _TurnEvidence:
    """Extract raw complexity evidence from a single message."""
    evidence = _TurnEvidence()
    text = message.text()
    if not text.strip():
        # A tool result with no text still implies an agentic, harder flow.
        if message.role == "tool":
            evidence.reasoning = 0.3
        return evidence

    stripped = text.strip()

    # Reasoning vocabulary: keep the strongest match rather than summing, so a
    # keyword-stuffed prompt cannot game the score.
    strongest = 0.0
    matches = 0
    for pattern, weight in _REASONING_PATTERNS:
        if pattern.search(text):
            strongest = max(strongest, weight)
            matches += 1
    # Multiple distinct demands genuinely add difficulty, but with diminishing
    # returns.
    evidence.reasoning = _clamp(strongest + 0.08 * max(0, matches - 1))

    fences = len(_CODE_FENCE.findall(text)) // 2
    code_hits = len(_CODE_HINTS.findall(text))
    evidence.code = _clamp(
        _saturating(fences, 1.0) * 0.7 + _saturating(code_hits, 6.0) * 0.5
    )
    if _STACK_TRACE.search(text):
        evidence.code = _clamp(evidence.code + 0.35)

    if _STRUCTURED_OUTPUT.search(text):
        evidence.structured = 0.7
    if message.role == "user" and _LONG_OUTPUT.search(text):
        evidence.long_output = 0.8

    if message.role == "user" and (_DISSATISFACTION.search(text) or _DISSATISFACTION_CN.search(text)):
        evidence.dissatisfaction = 1.0

    instruction_hits = len(_INSTRUCTION_MARKERS.findall(text))
    evidence.instruction_density = _saturating(instruction_hits, 5.0)

    has_cjk = bool(_CJK.search(text))
    has_latin = len(_LATIN_WORD.findall(text)) >= 3
    if has_cjk and has_latin:
        evidence.multilingual = 0.6

    if message.role == "user":
        evidence.trivial = any(p.match(stripped) for p in _TRIVIAL_PATTERNS) and len(stripped) < 80
        evidence.anaphoric = bool(_ANAPHORIC.match(stripped)) and len(stripped) < 120

    return evidence


class ComplexityAnalyzer:
    """Turns a conversation into a normalised complexity score in [0, 1]."""

    def __init__(self, config: ContextRoutingConfig, thresholds: TierThresholds) -> None:
        self._config = config
        self._thresholds = thresholds

    def analyse(
        self,
        request: ChatCompletionRequest,
        context_window_hint: int = 128_000,
        prompt_tokens: int | None = None,
    ) -> ComplexityAssessment:
        """Score the request against the full dialogue history.

        ``prompt_tokens`` lets the caller reuse a token count computed earlier
        (e.g. by the gateway's prepare stage) instead of tokenising the
        conversation again — tokenisation is one of the more expensive steps.
        """
        messages = request.messages
        cfg = self._config
        explanation: list[str] = []

        analysed = messages[-cfg.max_messages_analysed :] if cfg.enabled else messages[-1:]
        if not analysed:
            analysed = messages[-1:] if messages else []

        if prompt_tokens is None:
            prompt_tokens = count_message_tokens(messages) + count_tools_tokens(request.tools)

        if not analysed:
            signals = ComplexitySignals()
            return ComplexityAssessment(
                score=0.0,
                tier=self._thresholds.tier_for(0.0),
                signals=signals,
                prompt_tokens_estimate=prompt_tokens,
                turn_count=0,
                explanation=["empty conversation"],
            )

        evidences = [_analyse_turn(m) for m in analysed]

        # --- Recency weighting -------------------------------------------
        # The last message matters most, but earlier turns retain influence so
        # that a hard requirement stated up-front is never forgotten.
        decay = cfg.recency_decay
        weights = [decay ** (len(evidences) - 1 - i) for i in range(len(evidences))]
        weight_sum = sum(weights) or 1.0

        def weighted(attr: str) -> float:
            return sum(getattr(e, attr) * w for e, w in zip(evidences, weights)) / weight_sum

        def peak(attr: str) -> float:
            return max((getattr(e, attr) for e in evidences), default=0.0)

        # --- Escalation memory --------------------------------------------
        # Blend the weighted average with the historical peak: once a thread has
        # demonstrated hard requirements, a short follow-up must not silently
        # drop it back to a cheap model.
        memory = cfg.escalation_memory

        def with_memory(attr: str) -> float:
            avg = weighted(attr)
            top = peak(attr)
            return _clamp(avg + memory * max(0.0, top - avg))

        reasoning = with_memory("reasoning")
        code = with_memory("code")
        structured = with_memory("structured")
        long_output = with_memory("long_output")
        instruction_density = with_memory("instruction_density")
        multilingual = weighted("multilingual")

        # --- Conversation depth -------------------------------------------
        # Long threads accumulate constraints the model must simultaneously
        # satisfy; depth is measured in user turns, not raw message count.
        user_turns = sum(1 for m in analysed if m.role == "user")
        depth = _saturating(max(0, user_turns - 1), 4.0)

        # --- Context pressure ---------------------------------------------
        # A prompt filling a large share of the window needs strong long-context
        # attention, independent of its wording.
        usage_ratio = prompt_tokens / max(1, context_window_hint)
        pressure = _clamp(usage_ratio / max(cfg.context_pressure_threshold, 1e-6))

        # --- Unresolved thread ---------------------------------------------
        # Repeated dissatisfaction is the strongest online signal that the
        # current tier is failing this conversation.
        dissatisfaction_events = sum(1 for e in evidences if e.dissatisfaction > 0)
        recent_dissatisfaction = any(e.dissatisfaction > 0 for e in evidences[-4:])
        unresolved = _clamp(
            _saturating(dissatisfaction_events, 1.5) + (0.25 if recent_dissatisfaction else 0.0)
        )
        if dissatisfaction_events:
            explanation.append(
                f"{dissatisfaction_events} dissatisfaction signal(s) in thread → escalate"
            )

        # --- Tool usage -----------------------------------------------------
        tool_messages = sum(1 for m in analysed if m.role == "tool" or m.tool_calls)
        declared_tools = len(request.tools or [])
        tool_usage = _clamp(
            _saturating(declared_tools, 4.0) * 0.6 + _saturating(tool_messages, 2.0) * 0.7
        )

        # --- Requested output size -------------------------------------------
        requested_max = request.requested_max_tokens or 0
        output_size = _clamp(max(long_output, _saturating(requested_max, 4096.0)))

        signals = ComplexitySignals(
            conversation_depth=depth,
            context_length=pressure,
            reasoning_keywords=reasoning,
            code_content=code,
            structured_output=structured,
            tool_usage=tool_usage,
            multilingual=multilingual,
            unresolved_thread=unresolved,
            instruction_density=instruction_density,
            requested_output_size=output_size,
        )

        weights_cfg = cfg.signal_weights
        signal_weights = {
            "conversation_depth": weights_cfg.conversation_depth,
            "context_length": weights_cfg.context_length,
            "reasoning_keywords": weights_cfg.reasoning_keywords,
            "code_content": weights_cfg.code_content,
            "structured_output": weights_cfg.structured_output,
            "tool_usage": weights_cfg.tool_usage,
            "multilingual": weights_cfg.multilingual,
            "unresolved_thread": weights_cfg.unresolved_thread,
            "instruction_density": weights_cfg.instruction_density,
            "requested_output_size": weights_cfg.requested_output_size,
        }
        values = signals.as_dict()
        score = self._aggregate(values, signal_weights)

        # --- Trivial-turn shortcut -------------------------------------------
        # Only applies when the *whole* thread is shallow, otherwise a polite
        # "thanks, now continue" would be mis-routed to the cheapest model.
        last_evidence = evidences[-1]
        thread_is_shallow = score < self._thresholds.standard_max and unresolved == 0.0
        if last_evidence.trivial and thread_is_shallow and user_turns <= 2:
            score = min(score, self._thresholds.economy_max * 0.6)
            explanation.append("trivial greeting/short task in a shallow thread → economy")

        # --- Anaphoric follow-up ---------------------------------------------
        # "and the other case?" inherits the difficulty of what came before, so
        # it must never be scored on its own surface form.
        if last_evidence.anaphoric and len(evidences) > 1:
            prior_scores = [
                max(e.reasoning, e.code, e.structured) for e in evidences[:-1]
            ]
            inherited = max(prior_scores, default=0.0)
            if inherited > score:
                explanation.append(
                    "short follow-up inherits complexity from earlier turns"
                )
                score = _clamp(score + memory * (inherited - score))

        score = _clamp(score)

        latent = _clamp(score - self._score_last_turn_only(evidences[-1], request))
        if latent > 0.05:
            explanation.append(
                f"context adds +{latent:.2f} over single-turn scoring"
            )

        tier = self._thresholds.tier_for(score)
        requires_vision = any(m.has_image() for m in messages)
        requires_tools = bool(request.tools) or any(m.tool_calls for m in messages)

        if pressure > 0.5:
            explanation.append(f"context pressure {usage_ratio:.0%} of window")
        if not explanation:
            explanation.append("routine request scored on aggregate signals")

        return ComplexityAssessment(
            score=score,
            tier=tier,
            signals=signals,
            prompt_tokens_estimate=prompt_tokens,
            turn_count=user_turns,
            requires_vision=requires_vision,
            requires_tools=requires_tools,
            latent_escalation=latent,
            explanation=explanation,
        )

    @staticmethod
    def _aggregate(values: dict[str, float], weights: dict[str, float]) -> float:
        """Combine the individual signals into one score.

        A plain weighted mean is the obvious choice but behaves badly here: most
        signals are zero for any given request, so a single unambiguous demand
        ("prove this theorem") would be averaged down into the economy band by
        the eight unrelated signals that happen to be silent.

        Instead the score blends two views:

        * ``mean`` — the breadth of evidence: many mild signals together still
          indicate a demanding request.
        * ``dominant`` — the strength of the best evidence, weighted by that
          signal's importance, so one decisive signal can carry the decision.

        The dominant term leads, with the mean adding the remaining headroom.
        This keeps the score monotonic in every signal while ensuring neither
        breadth nor intensity alone is ignored.
        """
        total_weight = sum(weights.values()) or 1.0
        mean = sum(values[k] * w for k, w in weights.items()) / total_weight

        max_weight = max(weights.values()) or 1.0
        dominant = max(
            (values[k] * (w / max_weight) for k, w in weights.items()),
            default=0.0,
        )

        # ``dominant`` sets the floor; ``mean`` fills part of what remains.
        return _clamp(dominant * 0.6 + mean * 0.4 + dominant * mean * 0.35)

    def _score_last_turn_only(
        self, evidence: _TurnEvidence, request: ChatCompletionRequest
    ) -> float:
        """Baseline score of the final turn, used to quantify context gain.

        Scored through the same aggregation as the full conversation so the
        difference reflects added context rather than a change of formula.
        """
        weights_cfg = self._config.signal_weights
        values = {
            "conversation_depth": 0.0,
            "context_length": 0.0,
            "reasoning_keywords": evidence.reasoning,
            "code_content": evidence.code,
            "structured_output": evidence.structured,
            "tool_usage": 0.0,
            "multilingual": evidence.multilingual,
            "unresolved_thread": 0.0,
            "instruction_density": evidence.instruction_density,
            "requested_output_size": evidence.long_output,
        }
        weights = {
            "conversation_depth": weights_cfg.conversation_depth,
            "context_length": weights_cfg.context_length,
            "reasoning_keywords": weights_cfg.reasoning_keywords,
            "code_content": weights_cfg.code_content,
            "structured_output": weights_cfg.structured_output,
            "tool_usage": weights_cfg.tool_usage,
            "multilingual": weights_cfg.multilingual,
            "unresolved_thread": weights_cfg.unresolved_thread,
            "instruction_density": weights_cfg.instruction_density,
            "requested_output_size": weights_cfg.requested_output_size,
        }
        return self._aggregate(values, weights)
