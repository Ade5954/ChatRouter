"""Making an over-long conversation fit a model's context window.

The capability filter in :mod:`chatrouter.routing.router` drops any model whose
window cannot hold the prompt. When *every* model is dropped the request would
otherwise fail outright, which is the wrong behaviour for a gateway: long
conversations are routine, not exceptional.

This module implements the degradation path. Trimming is deliberately
conservative — it removes the *middle* of a conversation while preserving the
system prompt and the most recent turns, because those carry the instruction
and the actual question. Dropping either would silently change the meaning of
the request, which is worse than returning an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config.models import ContextOverflowConfig, ModelConfig
from ..core.schemas import ChatMessage
from ..core.tokens import count_message_tokens

# Marker inserted where turns were removed, so the model is not misled into
# reading the remaining turns as one contiguous exchange.
_ELISION_NOTICE = (
    "[earlier turns in this conversation were omitted because they exceeded "
    "the context window]"
)


@dataclass(slots=True)
class ContextFitResult:
    """Outcome of fitting a conversation to a model."""

    messages: list[ChatMessage]
    trimmed: bool = False
    removed_messages: int = 0
    original_tokens: int = 0
    final_tokens: int = 0
    notes: list[str] = field(default_factory=list)


def largest_window_model(models: list[ModelConfig]) -> ModelConfig | None:
    """The candidate able to hold the most context."""
    if not models:
        return None
    return max(models, key=lambda m: m.context_window)


def fits(model: ModelConfig, prompt_tokens: int, reserve_output: int = 0) -> bool:
    """Whether a prompt (plus reserved completion budget) fits a model."""
    return prompt_tokens + max(0, reserve_output) <= model.context_window


def trim_to_fit(
    messages: list[ChatMessage],
    model: ModelConfig,
    config: ContextOverflowConfig,
    reserve_output: int = 0,
) -> ContextFitResult:
    """Drop middle turns until the conversation fits ``model``.

    Preserves the leading messages (system prompt / task framing) and the
    trailing messages (the current question and its immediate context). Only
    the middle is eligible for removal, and messages are dropped oldest-first
    so the most recent context survives longest.
    """
    original_tokens = count_message_tokens(messages, model.id)
    budget = int(model.context_window * config.trim_target_ratio)
    if reserve_output:
        budget = min(budget, model.context_window - reserve_output)
    budget = max(budget, 1)

    if original_tokens <= budget:
        return ContextFitResult(
            messages=list(messages),
            original_tokens=original_tokens,
            final_tokens=original_tokens,
        )

    lead = min(config.keep_leading_messages, len(messages))
    tail = min(config.keep_trailing_messages, max(0, len(messages) - lead))

    head_part = list(messages[:lead])
    tail_part = list(messages[len(messages) - tail :]) if tail else []
    middle = list(messages[lead : len(messages) - tail]) if tail else list(messages[lead:])

    removed = 0
    # Oldest-first removal: recent context is the most relevant to the answer.
    while middle:
        candidate = head_part + middle + tail_part
        if count_message_tokens(candidate, model.id) <= budget:
            break
        middle.pop(0)
        removed += 1

    result_messages = head_part + middle + tail_part

    if removed and config.insert_elision_notice:
        notice = ChatMessage(role="system", content=_ELISION_NOTICE)
        result_messages = head_part + [notice] + middle + tail_part

    final_tokens = count_message_tokens(result_messages, model.id)
    notes: list[str] = []
    if removed:
        notes.append(
            f"trimmed {removed} message(s) to fit {model.id} "
            f"({original_tokens} → {final_tokens} tokens)"
        )

    # The protected head+tail (plus the elision notice) may be irreducible and
    # still exceed the budget. Report honestly rather than pretending the trim
    # succeeded — the caller may need to fail the request or pick a wider
    # model. Compared against the budget, not just the raw window, so that a
    # squeezed output reservation is also surfaced.
    if final_tokens > budget:
        notes.append(
            f"conversation still exceeds the {budget}-token budget for "
            f"{model.id} after trimming ({final_tokens} tokens); the "
            "protected system prompt and recent turns cannot be reduced further"
        )

    return ContextFitResult(
        messages=result_messages,
        trimmed=bool(removed),
        removed_messages=removed,
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        notes=notes,
    )
