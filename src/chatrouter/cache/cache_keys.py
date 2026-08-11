"""Deterministic serialisation of a completion request into a cache key string.

The key must include exactly the fields that affect the generated text. We
deliberately exclude ``chatrouter`` hints *except* ``pin_model`` (which changes
the answering model) — ``min_tier`` etc. do not alter what the model produces,
and including them would fragment the cache without improving correctness.
``stream`` is excluded too because streaming is rejected before this point and
the underlying completion is identical.

Session affinity vs. cache keys
--------------------------------
When a ``session_id`` is present and affinity awareness is on, the key becomes
*session-scoped*: it includes the session id together with the resolved model.
Because session affinity keeps a conversation on one model, identical turns
within that session stay on the same model and can share the cache, while a
session that drifts to a different model gets a different key — so a stale
answer from the previous model can never be served. Without a session id the
key is unchanged from the legacy form.
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
    request: ChatCompletionRequest,
    resolved_model: ModelConfig,
    affinity_aware: bool = True,
) -> str:
    """Build a canonical, order-independent key string for the request.

    ``affinity_aware`` controls whether a ``session_id`` hint is folded into the
    key (True) or ignored (False, legacy behaviour where the caller bypasses the
    cache for sessions entirely).
    """
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

    # Session affinity awareness: scope the key to (session, model). The model
    # component already isolates turns that drifted to a different model, so the
    # cache is safe within a sticky session and stacks with prefix caching.
    if affinity_aware and hints is not None and hints.session_id:
        parts["session"] = {"id": hints.session_id, "model": resolved_model.id}

    canonical = json_dumps_sorted(parts)
    # Prefix with the model for readability and to keep the model dimension in
    # the key even when affinity awareness is off (legacy guarantee).
    return f"{resolved_model.id}|{canonical}"


def json_dumps_sorted(obj: Any) -> str:
    """Serialise with sorted keys and stable encoding for reproducible digests."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
