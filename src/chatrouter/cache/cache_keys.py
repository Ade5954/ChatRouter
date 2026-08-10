"""Deterministic serialisation of a completion request into a cache key string.

The key must include exactly the fields that affect the generated text. We
deliberately exclude ``chatrouter`` hints *except* ``pin_model`` (which changes
the answering model) — ``session_id``, ``min_tier`` etc. do not alter what the
model produces, and including them would fragment the cache without improving
correctness. ``stream`` is excluded too because streaming is rejected before
this point and the underlying completion is identical.
"""

from __future__ import annotations

import json
from typing import Any

from ..config.models import ModelConfig
from ..core.schemas import ChatCompletionRequest

# Fields of the request that change the produced completion. Order does not
# matter (json.dumps sorts keys) but completeness is critical.
_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "n",
    "stop",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "seed",
    "response_format",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
)


def cache_key_for_request(
    request: ChatCompletionRequest, resolved_model: ModelConfig
) -> str:
    """Build a canonical, order-independent key string for the request."""
    parts: dict[str, Any] = {
        # The model that actually answers. For "auto" the router resolves this,
        # so two identical prompts routed to different models must not share a
        # key.
        "model": resolved_model.id,
        "messages": [m.model_dump(exclude_none=True) for m in request.messages],
    }

    # Sampling parameters that influence the output.
    for field in _SAMPLING_FIELDS:
        value = getattr(request, field, None)
        if value is not None:
            parts[field] = value

    # Only pin_model among the hints can change the answering model; the rest
    # are routing preferences that leave the text unchanged.
    hints = request.chatrouter
    if hints is not None and hints.pin_model:
        parts["pin_model"] = hints.pin_model

    canonical = json_dumps_sorted(parts)
    return f"{resolved_model.id}|{canonical}"


def json_dumps_sorted(obj: Any) -> str:
    """Serialise with sorted keys and stable encoding for reproducible digests."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
