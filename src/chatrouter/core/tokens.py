"""Token estimation.

Uses ``tiktoken`` when available and falls back to a language-aware heuristic
so the gateway never fails because an encoding could not be downloaded.
"""

from __future__ import annotations

import functools
import json
from typing import Any

from .schemas import ChatMessage

# Overhead per message in the OpenAI chat format (role + delimiters).
_TOKENS_PER_MESSAGE = 4
_TOKENS_PER_NAME = 1
_REPLY_PRIMER_TOKENS = 3
# Approximate cost of an image part when we cannot inspect the payload.
_IMAGE_TOKENS = 850


@functools.lru_cache(maxsize=8)
def _encoding(model: str) -> Any | None:
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - optional dependency
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover - no encoding files available
            return None


def _heuristic_tokens(text: str) -> int:
    """Rough estimate that accounts for CJK being denser than Latin text.

    Latin scripts average ~4 characters per token, CJK closer to ~1.5.
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff")
    other = len(text) - cjk
    return int(cjk / 1.5 + other / 4) + 1


def count_text_tokens(text: str, model: str = "gpt-4o") -> int:
    """Token count of a plain string."""
    if not text:
        return 0
    enc = _encoding(model)
    if enc is None:
        return _heuristic_tokens(text)
    try:
        return len(enc.encode(text, disallowed_special=()))
    except Exception:  # pragma: no cover - defensive
        return _heuristic_tokens(text)


def count_message_tokens(messages: list[ChatMessage], model: str = "gpt-4o") -> int:
    """Estimate the prompt tokens of a full conversation."""
    total = 0
    for message in messages:
        total += _TOKENS_PER_MESSAGE
        total += count_text_tokens(message.role, model)
        total += count_text_tokens(message.text(), model)
        if message.name:
            total += _TOKENS_PER_NAME + count_text_tokens(message.name, model)
        if message.tool_calls:
            total += count_text_tokens(json.dumps(message.tool_calls, ensure_ascii=False), model)
        if message.has_image():
            total += _IMAGE_TOKENS
    return total + _REPLY_PRIMER_TOKENS


def count_tools_tokens(tools: list[dict[str, Any]] | None, model: str = "gpt-4o") -> int:
    """Tool/function schemas also consume prompt budget."""
    if not tools:
        return 0
    return count_text_tokens(json.dumps(tools, ensure_ascii=False), model)


def estimate_request_tokens(
    messages: list[ChatMessage],
    tools: list[dict[str, Any]] | None = None,
    max_output_tokens: int | None = None,
    model: str = "gpt-4o",
) -> tuple[int, int]:
    """Return ``(prompt_tokens, projected_total_tokens)``.

    The projection is used up-front by the TPM limiter and quota checks before
    the real usage is known.
    """
    prompt = count_message_tokens(messages, model) + count_tools_tokens(tools, model)
    # Without an explicit cap, assume a completion of ~35% of the prompt.
    completion = max_output_tokens if max_output_tokens else max(256, int(prompt * 0.35))
    return prompt, prompt + completion
