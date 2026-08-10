"""Redis storage backend for multi-replica deployments.

Counter and quota updates are executed as Lua scripts so that increment plus
expiry is atomic, which keeps limits correct across concurrent workers.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from .base import QuotaUsage, Storage

# INCRBY + set TTL only when the key is new, returning the new total.
_INCR_WINDOW_LUA = """
local current = redis.call('INCRBY', KEYS[1], ARGV[1])
if current == tonumber(ARGV[1]) then
    redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return current
"""

# Accumulate the three quota dimensions in one hash, atomically.
_ADD_USAGE_LUA = """
local requests = redis.call('HINCRBY', KEYS[1], 'requests', ARGV[1])
local tokens = redis.call('HINCRBY', KEYS[1], 'tokens', ARGV[2])
local cost = redis.call('HINCRBYFLOAT', KEYS[1], 'cost_usd', ARGV[3])
if requests == tonumber(ARGV[1]) then
    redis.call('EXPIRE', KEYS[1], ARGV[4])
end
return {requests, tokens, cost}
"""

# Read and delete in one step. A GET followed by a DEL would let two replicas
# both observe the record and apply feedback twice.
_CLAIM_RECORD_LUA = """
local value = redis.call('GET', KEYS[1])
if value then
    redis.call('DEL', KEYS[1])
end
return value
"""


class RedisStorage(Storage):
    """Redis-backed implementation of the storage contract."""

    def __init__(self, url: str, key_prefix: str = "chatrouter") -> None:
        self._url = url
        self._prefix = key_prefix
        self._client: Any = None
        self._incr_window_script: Any = None
        self._add_usage_script: Any = None
        self._claim_record_script: Any = None

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        try:
            from redis.asyncio import Redis
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("redis package is required for the redis storage backend") from exc

        self._client = Redis.from_url(self._url, encoding="utf-8", decode_responses=True)
        await self._client.ping()
        self._incr_window_script = self._client.register_script(_INCR_WINDOW_LUA)
        self._add_usage_script = self._client.register_script(_ADD_USAGE_LUA)
        self._claim_record_script = self._client.register_script(_CLAIM_RECORD_LUA)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- helpers ---------------------------------------------------------

    def _k(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    @staticmethod
    def _window_start(now: float, window_seconds: int) -> int:
        return int(now - (now % window_seconds))

    def _window_key(self, key: str, window_seconds: int) -> str:
        now = time.time()
        return f"{self._k(key)}:{window_seconds}:{self._window_start(now, window_seconds)}"

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("redis storage used before start()")
        return self._client

    # -- window counters -------------------------------------------------

    async def incr_window(self, key: str, amount: int, window_seconds: int) -> int:
        self._require_client()
        result = await self._incr_window_script(
            keys=[self._window_key(key, window_seconds)],
            args=[amount, window_seconds + 5],
        )
        return int(result)

    async def get_window(self, key: str, window_seconds: int) -> int:
        client = self._require_client()
        value = await client.get(self._window_key(key, window_seconds))
        return int(value) if value else 0

    async def window_ttl(self, key: str, window_seconds: int) -> float:
        now = time.time()
        return max(0.0, self._window_start(now, window_seconds) + window_seconds - now)

    # -- quota usage -----------------------------------------------------

    async def add_usage(
        self, key: str, requests: int, tokens: int, cost_usd: float, window_seconds: int
    ) -> QuotaUsage:
        self._require_client()
        result = await self._add_usage_script(
            keys=[self._window_key(key, window_seconds)],
            args=[requests, tokens, cost_usd, window_seconds + 5],
        )
        return QuotaUsage(
            requests=int(result[0]),
            tokens=int(result[1]),
            cost_usd=float(result[2]),
            window_reset_seconds=await self.window_ttl(key, window_seconds),
        )

    async def get_usage(self, key: str, window_seconds: int) -> QuotaUsage:
        client = self._require_client()
        data = await client.hgetall(self._window_key(key, window_seconds))
        return QuotaUsage(
            requests=int(data.get("requests", 0) or 0),
            tokens=int(data.get("tokens", 0) or 0),
            cost_usd=float(data.get("cost_usd", 0.0) or 0.0),
            window_reset_seconds=await self.window_ttl(key, window_seconds),
        )

    # -- gauges ----------------------------------------------------------

    async def incr_gauge(self, key: str, amount: int = 1) -> int:
        client = self._require_client()
        value = int(await client.incrby(self._k(key), amount))
        if value < 0:
            # Self-heal if a worker died mid-request and leaked a decrement.
            await client.set(self._k(key), 0)
            return 0
        # Gauges must not live forever if traffic stops.
        await client.expire(self._k(key), 3600)
        return value

    async def get_gauge(self, key: str) -> int:
        client = self._require_client()
        value = await client.get(self._k(key))
        return int(value) if value else 0

    # -- stats -----------------------------------------------------------

    async def get_stats(self, key: str) -> dict[str, Any] | None:
        client = self._require_client()
        raw = await client.get(self._k(key))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set_stats(self, key: str, value: dict[str, Any], ttl_seconds: int | None = None) -> None:
        client = self._require_client()
        payload = json.dumps(value, ensure_ascii=False)
        if ttl_seconds:
            await client.set(self._k(key), payload, ex=ttl_seconds)
        else:
            await client.set(self._k(key), payload)

    async def get_all_stats(self, prefix: str) -> dict[str, dict[str, Any]]:
        client = self._require_client()
        full = self._k(prefix)
        result: dict[str, dict[str, Any]] = {}
        async for key in client.scan_iter(match=f"{full}*", count=200):
            raw = await client.get(key)
            if not raw:
                continue
            try:
                result[key[len(full) :].lstrip(":")] = json.loads(raw)
            except json.JSONDecodeError:
                continue
        return result

    async def update_stats(
        self,
        key: str,
        mutate: Callable[[dict[str, Any]], dict[str, Any]],
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Optimistic-locked read-modify-write via WATCH/MULTI/EXEC."""
        client = self._require_client()
        full = self._k(key)
        for _ in range(5):
            async with client.pipeline() as pipe:
                try:
                    await pipe.watch(full)
                    raw = await pipe.get(full)
                    current = json.loads(raw) if raw else {}
                    updated = mutate(current)
                    pipe.multi()
                    payload = json.dumps(updated, ensure_ascii=False)
                    if ttl_seconds:
                        await pipe.set(full, payload, ex=ttl_seconds)
                    else:
                        await pipe.set(full, payload)
                    await pipe.execute()
                    return updated
                except Exception as exc:
                    if type(exc).__name__ != "WatchError":
                        raise
                    continue
        # Contention beyond the retry budget: apply a last-writer-wins update.
        current = await self.get_stats(key) or {}
        updated = mutate(current)
        await self.set_stats(key, updated, ttl_seconds)
        return updated

    # -- short-lived records ---------------------------------------------

    async def put_record(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        client = self._require_client()
        await client.set(self._k(key), json.dumps(value, ensure_ascii=False), ex=ttl_seconds)

    async def read_record(self, key: str) -> dict[str, Any] | None:
        client = self._require_client()
        raw = await client.get(self._k(key))
        return self._decode_record(raw)

    async def claim_record(self, key: str) -> dict[str, Any] | None:
        self._require_client()
        raw = await self._claim_record_script(keys=[self._k(key)], args=[])
        return self._decode_record(raw)

    @staticmethod
    def _decode_record(raw: Any) -> dict[str, Any] | None:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
