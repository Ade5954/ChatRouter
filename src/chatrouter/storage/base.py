"""Storage abstraction for counters, windows and learned statistics.

Two implementations are provided: an in-process backend for single-replica
deployments and tests, and a Redis backend for horizontally scaled clusters.
Both share this interface so the rest of the gateway is backend agnostic.
"""

from __future__ import annotations

import abc
import collections.abc
from dataclasses import dataclass
from typing import Any

from ..core.types import SessionAffinityBinding


@dataclass(slots=True)
class RateLimitVerdict:
    """Outcome of a rate-limit check."""

    allowed: bool
    limit: int | None = None
    remaining: int | None = None
    retry_after: float | None = None
    reason: str | None = None


@dataclass(slots=True)
class QuotaUsage:
    """Accumulated usage inside the current quota window."""

    requests: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    window_reset_seconds: float = 0.0


class Storage(abc.ABC):
    """Backend-agnostic persistence for gateway runtime state."""

    @abc.abstractmethod
    async def start(self) -> None:
        """Initialise connections."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Release resources."""

    # -- Sliding-window counters (rate limiting) -------------------------

    @abc.abstractmethod
    async def incr_window(self, key: str, amount: int, window_seconds: int) -> int:
        """Add ``amount`` to a fixed window bucket and return the new total."""

    @abc.abstractmethod
    async def get_window(self, key: str, window_seconds: int) -> int:
        """Read the current value of a fixed window bucket."""

    @abc.abstractmethod
    async def window_ttl(self, key: str, window_seconds: int) -> float:
        """Seconds remaining before the window resets."""

    # -- Quota accounting ------------------------------------------------

    @abc.abstractmethod
    async def add_usage(
        self, key: str, requests: int, tokens: int, cost_usd: float, window_seconds: int
    ) -> QuotaUsage:
        """Accumulate usage and return the running totals."""

    @abc.abstractmethod
    async def get_usage(self, key: str, window_seconds: int) -> QuotaUsage:
        """Read the running totals for a quota window."""

    # -- Concurrency gauges ----------------------------------------------

    @abc.abstractmethod
    async def incr_gauge(self, key: str, amount: int = 1) -> int:
        """Adjust a gauge (e.g. in-flight requests) and return the new value."""

    @abc.abstractmethod
    async def get_gauge(self, key: str) -> int:
        """Read a gauge value."""

    # -- Learned statistics (feedback loop) ------------------------------

    @abc.abstractmethod
    async def get_stats(self, key: str) -> dict[str, Any] | None:
        """Read a serialised statistics record."""

    @abc.abstractmethod
    async def set_stats(self, key: str, value: dict[str, Any], ttl_seconds: int | None = None) -> None:
        """Persist a serialised statistics record."""

    @abc.abstractmethod
    async def get_all_stats(self, prefix: str) -> dict[str, dict[str, Any]]:
        """Read every statistics record under a prefix, keyed by suffix."""

    @abc.abstractmethod
    async def update_stats(
        self, key: str, mutate: Any, ttl_seconds: int | None = None
    ) -> dict[str, Any]:
        """Atomically read-modify-write a statistics record.

        ``mutate`` receives the current record (or an empty dict) and returns
        the updated record.
        """

    # -- Short-lived records (request → feedback correlation) -------------

    @abc.abstractmethod
    async def put_record(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        """Store a record that expires automatically."""

    @abc.abstractmethod
    async def read_record(self, key: str) -> dict[str, Any] | None:
        """Read a record without consuming it.

        Use this for introspection only. Anything that mutates learned state
        on the basis of a record must use :meth:`claim_record` instead.
        """

    @abc.abstractmethod
    async def claim_record(self, key: str) -> dict[str, Any] | None:
        """Atomically read *and* consume a record.

        Returns the record to exactly one caller; every subsequent call for the
        same key returns ``None``. This is the primitive that makes
        request→feedback correlation single-shot, preventing a client that
        holds a ``request_id`` from replaying feedback to poison the adaptive
        routing statistics.
        """

    # -- Session affinity (keep a conversation on one model) --------------

    @abc.abstractmethod
    async def get_session_affinity(self, session_id: str) -> SessionAffinityBinding | None:
        """Return the session's affinity binding (model id + cached prefix size)."""

    @abc.abstractmethod
    async def set_session_affinity(
        self, session_id: str, model_id: str, prefix_tokens: int, ttl_seconds: int
    ) -> None:
        """Remember the model a session was routed to, plus its cached prefix size."""

    async def get_session_model(self, session_id: str) -> str | None:
        """Backwards-compatible view: just the pinned model id (or ``None``)."""
        binding = await self.get_session_affinity(session_id)
        return binding.model_id if binding is not None else None

    async def session_affinity_ttl(self, session_id: str) -> int:
        """Remaining lifetime (seconds) of a session's affinity binding.

        Returns 0 when the session has no live binding. Used to cap the TTL of a
        response-cache entry scoped to the session, so a cached answer cannot
        outlive the model the session is actually pinned to.
        """
        binding = await self.get_session_affinity(session_id)
        return binding.ttl_remaining if binding is not None else 0

    # -- Externalised configuration (multi-replica) ----------------------

    async def get_config(self) -> dict[str, Any] | None:
        """Read the authoritative configuration document, or ``None`` if unset.

        Used at startup to load the gateway's configuration from a shared
        store so every replica converges on the same providers, models,
        tenants and routing policy. The document is the already-expanded
        configuration tree (no ``${ENV}`` placeholders); callers pass it
        straight to ``AppConfig.model_validate``.
        """
        raise NotImplementedError

    async def set_config(self, data: dict[str, Any]) -> int:
        """Persist the authoritative configuration and return the new version.

        Every write bumps a monotonic version number so subscribers can decide
        whether a reload notification carries a configuration they have
        already applied (Phase 2 will publish that version via Pub/Sub).
        """
        raise NotImplementedError

    async def get_config_version(self) -> int:
        """Current configuration version, or 0 when no configuration is stored."""
        raise NotImplementedError

    # -- Cross-replica reload notifications (Phase 2) --------------------

    async def publish_config_reload(self, version: int) -> None:
        """Notify every replica that a new configuration version is available.

        The message body is just the version number; subscribers pull the
        payload from :meth:`get_config` themselves. This keeps the publish
        cheap and avoids serialising a large document twice (it is already
        stored via :meth:`set_config`).
        """
        raise NotImplementedError

    def config_reload_events(self) -> "collections.abc.AsyncIterator[int]":
        """Return an async iterator yielding configuration version numbers.

        Each yielded value is the version announced by a publisher (i.e. a
        replica that just wrote a new configuration). Subscribers compare it
        against the version they have already applied and reload from storage
        only when the announcement is newer.

        The iterator runs forever; cancelling the task consuming it
        unsubscribes. Backends without a real pub/sub mechanism (the in-memory
        backend used by tests) implement this with an ``asyncio.Queue``.
        """
        raise NotImplementedError
