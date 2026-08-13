"""In-process storage backend.

Suitable for single-replica deployments, development and tests. All state is
lost on restart and is *not* shared between workers.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from .base import QuotaUsage, Storage
from ..core.types import SessionAffinityBinding


class _Bucket:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, expires_at: float | None) -> None:
        self.value = value
        self.expires_at = expires_at

    def expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at


class MemoryStorage(Storage):
    """Thread-safe-enough (single event loop) in-memory implementation."""

    def __init__(self, key_prefix: str = "chatrouter") -> None:
        self._prefix = key_prefix
        self._counters: dict[str, _Bucket] = {}
        self._usage: dict[str, _Bucket] = {}
        self._gauges: dict[str, int] = {}
        self._stats: dict[str, _Bucket] = {}
        self._records: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()
        self._last_sweep = 0.0

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        self._counters.clear()
        self._usage.clear()
        self._gauges.clear()
        self._stats.clear()
        self._records.clear()

    # -- helpers ---------------------------------------------------------

    def _k(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    @staticmethod
    def _window_start(now: float, window_seconds: int) -> float:
        return now - (now % window_seconds)

    def _window_key(self, key: str, window_seconds: int, now: float) -> str:
        return f"{self._k(key)}:{window_seconds}:{int(self._window_start(now, window_seconds))}"

    def _sweep(self, now: float) -> None:
        """Drop expired entries; runs at most once per second."""
        if now - self._last_sweep < 1.0:
            return
        self._last_sweep = now
        for store in (self._counters, self._usage, self._stats, self._records):
            for key in [k for k, b in store.items() if b.expired(now)]:
                store.pop(key, None)

    # -- window counters -------------------------------------------------

    async def incr_window(self, key: str, amount: int, window_seconds: int) -> int:
        now = time.time()
        async with self._lock:
            self._sweep(now)
            wkey = self._window_key(key, window_seconds, now)
            bucket = self._counters.get(wkey)
            if bucket is None or bucket.expired(now):
                bucket = _Bucket(0, self._window_start(now, window_seconds) + window_seconds)
                self._counters[wkey] = bucket
            bucket.value += amount
            return int(bucket.value)

    async def get_window(self, key: str, window_seconds: int) -> int:
        now = time.time()
        wkey = self._window_key(key, window_seconds, now)
        bucket = self._counters.get(wkey)
        if bucket is None or bucket.expired(now):
            return 0
        return int(bucket.value)

    async def window_ttl(self, key: str, window_seconds: int) -> float:
        now = time.time()
        return max(0.0, self._window_start(now, window_seconds) + window_seconds - now)

    # -- quota usage -----------------------------------------------------

    async def add_usage(
        self, key: str, requests: int, tokens: int, cost_usd: float, window_seconds: int
    ) -> QuotaUsage:
        now = time.time()
        async with self._lock:
            self._sweep(now)
            wkey = self._window_key(key, window_seconds, now)
            expires = self._window_start(now, window_seconds) + window_seconds
            bucket = self._usage.get(wkey)
            if bucket is None or bucket.expired(now):
                bucket = _Bucket({"requests": 0, "tokens": 0, "cost_usd": 0.0}, expires)
                self._usage[wkey] = bucket
            bucket.value["requests"] += requests
            bucket.value["tokens"] += tokens
            bucket.value["cost_usd"] += cost_usd
            return QuotaUsage(
                requests=int(bucket.value["requests"]),
                tokens=int(bucket.value["tokens"]),
                cost_usd=float(bucket.value["cost_usd"]),
                window_reset_seconds=max(0.0, expires - now),
            )

    async def get_usage(self, key: str, window_seconds: int) -> QuotaUsage:
        now = time.time()
        wkey = self._window_key(key, window_seconds, now)
        expires = self._window_start(now, window_seconds) + window_seconds
        bucket = self._usage.get(wkey)
        if bucket is None or bucket.expired(now):
            return QuotaUsage(window_reset_seconds=max(0.0, expires - now))
        return QuotaUsage(
            requests=int(bucket.value["requests"]),
            tokens=int(bucket.value["tokens"]),
            cost_usd=float(bucket.value["cost_usd"]),
            window_reset_seconds=max(0.0, expires - now),
        )

    # -- gauges ----------------------------------------------------------

    async def incr_gauge(self, key: str, amount: int = 1) -> int:
        async with self._lock:
            value = self._gauges.get(self._k(key), 0) + amount
            value = max(0, value)
            self._gauges[self._k(key)] = value
            return value

    async def get_gauge(self, key: str) -> int:
        return self._gauges.get(self._k(key), 0)

    # -- stats -----------------------------------------------------------

    async def get_stats(self, key: str) -> dict[str, Any] | None:
        now = time.time()
        bucket = self._stats.get(self._k(key))
        if bucket is None or bucket.expired(now):
            return None
        return dict(bucket.value)

    async def set_stats(self, key: str, value: dict[str, Any], ttl_seconds: int | None = None) -> None:
        expires = time.time() + ttl_seconds if ttl_seconds else None
        self._stats[self._k(key)] = _Bucket(dict(value), expires)

    async def get_all_stats(self, prefix: str) -> dict[str, dict[str, Any]]:
        now = time.time()
        full = self._k(prefix)
        result: dict[str, dict[str, Any]] = {}
        for key, bucket in self._stats.items():
            if key.startswith(full) and not bucket.expired(now):
                result[key[len(full) :].lstrip(":")] = dict(bucket.value)
        return result

    async def update_stats(
        self,
        key: str,
        mutate: Callable[[dict[str, Any]], dict[str, Any]],
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            current = await self.get_stats(key) or {}
            updated = mutate(current)
            expires = time.time() + ttl_seconds if ttl_seconds else None
            self._stats[self._k(key)] = _Bucket(dict(updated), expires)
            return updated

    # -- short-lived records ---------------------------------------------

    async def put_record(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        self._records[self._k(key)] = _Bucket(dict(value), time.time() + ttl_seconds)

    async def read_record(self, key: str) -> dict[str, Any] | None:
        now = time.time()
        bucket = self._records.get(self._k(key))
        if bucket is None or bucket.expired(now):
            self._records.pop(self._k(key), None)
            return None
        return dict(bucket.value)

    async def claim_record(self, key: str) -> dict[str, Any] | None:
        # The pop must happen under the lock: two concurrent claims for the
        # same key must not both observe a live bucket.
        now = time.time()
        async with self._lock:
            bucket = self._records.pop(self._k(key), None)
            if bucket is None or bucket.expired(now):
                return None
            return dict(bucket.value)

    # -- session affinity ------------------------------------------------

    async def get_session_affinity(self, session_id: str) -> SessionAffinityBinding | None:
        now = time.time()
        bucket = self._records.get(self._k(f"affinity:{session_id}"))
        if bucket is None or bucket.expired(now):
            self._records.pop(self._k(f"affinity:{session_id}"), None)
            return None
        value = bucket.value
        ttl_remaining = int(max(0, bucket.expires_at - now))
        return SessionAffinityBinding(
            model_id=value.get("model"),
            prefix_tokens=int(value.get("prefix_tokens", 0) or 0),
            ttl_remaining=ttl_remaining,
        )

    async def set_session_affinity(
        self, session_id: str, model_id: str, prefix_tokens: int, ttl_seconds: int
    ) -> None:
        self._records[self._k(f"affinity:{session_id}")] = _Bucket(
            {"model": model_id, "prefix_tokens": int(prefix_tokens)},
            time.time() + ttl_seconds,
        )
