"""Exact-match response cache for non-streaming completions.

The cache key must capture *every* request field that can change the generated
text. If any such field is omitted, two semantically different requests would
collide and a caller could receive another conversation's answer. We therefore
hash the resolved target model, the message list, and the full set of sampling
parameters (temperature, top_p, max_tokens, stop, tools, tool_choice,
response_format, seed, user, n) plus the routing hints — because ``pin_model``
can change which model answers.

Session affinity vs. the response cache
--------------------------------------
With ``affinity_aware`` (default True) a request carrying a ``session_id`` is
*not* bypassed. Instead the key is scoped to ``(session_id, resolved_model)``
(see ``cache_keys``), so a sticky session reuses cached answers while a session
that drifts to a different model gets an isolated key. Streaming requests are
still excluded by the caller (the SSE stream is live and must not be replayed),
and any hint listed in ``bypass_hints`` forces a bypass. When affinity awareness
is off, ``session_id`` becomes a bypass hint automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config.models import ModelConfig, ResponseCacheConfig, TenantConfig
from ..core.schemas import ChatCompletionRequest
from ..storage.base import Storage
from .cache_keys import cache_key_for_request

_CacheEntry = dict[str, Any]


@dataclass(slots=True)
class CacheOutcome:
    """Result of a cache lookup, used for metrics and response headers."""

    hit: bool
    reason: str  # "hit", "miss", "disabled", "excluded", "bypass_hint", "streaming"
    model_id: str | None = None


class ResponseCache:
    """Read-through / write-through cache keyed on the normalised request."""

    def __init__(self, config: ResponseCacheConfig, storage: Storage) -> None:
        self._config = config
        self._storage = storage

    def should_participate(self, request: ChatCompletionRequest, tenant: TenantConfig) -> bool:
        """Whether this request may use the cache at all.

        Returns ``False`` (and a reason) when the cache is disabled, the tenant
        is excluded, the request is streaming, or a bypass hint is present.
        """
        if not self._config.enabled:
            return False
        if tenant.id in self._config.excluded_tenants:
            return False
        if request.stream:
            return False
        hints = request.chatrouter
        if hints is not None:
            for name in self._config.bypass_hints:
                if getattr(hints, name, None):
                    return False
        return True

    def key_for(
        self, request: ChatCompletionRequest, resolved_model: ModelConfig
    ) -> str:
        """Deterministic cache key incorporating every output-affecting field."""
        return cache_key_for_request(
            request, resolved_model, affinity_aware=self._config.affinity_aware
        )

    async def get(self, key: str) -> _CacheEntry | None:
        raw = await self._storage.read_record(self._entry_key(key))
        if not isinstance(raw, dict):
            return None
        # Defensive: a corrupt or partial entry must not be served as a real
        # completion. Treat anything missing the payload as a miss.
        if "payload" not in raw:
            return None
        return raw

    async def put(
        self,
        key: str,
        payload: dict[str, Any],
        model_id: str,
        affinity_ttl_seconds: int | None = None,
    ) -> None:
        """Store a successful completion.

        The cached value keeps the model id so accounting on a cache hit can
        attribute cost and tokens to the correct model even though no upstream
        call happens.

        ``affinity_ttl_seconds`` is the remaining lifetime of the session's
        affinity binding. When a cached answer is scoped to a session (affinity
        awareness on), the entry must not outlive the affinity binding, or we
        could serve a cached answer from a model the session has since drifted
        away from. We therefore cap the entry TTL at ``min(cache_ttl,
        affinity_ttl)``.
        """
        ttl = self._config.ttl_seconds
        if affinity_ttl_seconds is not None:
            ttl = min(ttl, max(0, affinity_ttl_seconds))
        entry: _CacheEntry = {
            "payload": payload,
            "model_id": model_id,
        }
        await self._storage.put_record(self._entry_key(key), entry, ttl_seconds=ttl)

    @staticmethod
    def _entry_key(cache_key: str) -> str:
        return f"resp_cache:{cache_key}"
