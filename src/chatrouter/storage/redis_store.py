"""Redis storage backend for multi-replica deployments.

Counter and quota updates are executed as Lua scripts so that increment plus
expiry is atomic, which keeps limits correct across concurrent workers.

Config-reload notifications use a Redis Stream with one consumer group per
replica (group name = ``replica_id``). Streams persist messages, so a
subscriber that was offline when a notification was published can catch up on
restart; the consumer-group pending-entries list plus ``XAUTOCLAIM`` recover
messages that were delivered but never acked (e.g. the replica crashed mid-
reload). This is strictly stronger than Pub/Sub, which drops any message
published while no subscriber is connected.
"""

from __future__ import annotations

import asyncio
import collections.abc
import json
import os
import socket
import time
from collections.abc import Callable
from typing import Any

from .base import QuotaUsage, Storage
from ..core.types import SessionAffinityBinding

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

    def __init__(
        self,
        url: str,
        key_prefix: str = "chatrouter",
        *,
        replica_id: str | None = None,
    ) -> None:
        self._url = url
        self._prefix = key_prefix
        # Resolve a stable per-replica identity for the consumer group. The
        # order is: explicit config > env var > hostname > "default". The
        # "default" fallback is only safe for single-instance deployments;
        # multi-replica deployments MUST set distinct values (docker-compose
        # gives each container a distinct hostname, which is why hostname is
        # the third fallback).
        self._replica_id = (
            replica_id
            or os.environ.get("CHATROUTER_REPLICA_ID")
            or socket.gethostname()
            or "default"
        )
        self._client: Any = None
        self._incr_window_script: Any = None
        self._add_usage_script: Any = None
        self._claim_record_script: Any = None
        self._consumer_group_ready = False

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        # Idempotent: the app factory may start the storage before handing it
        # to the service (to load the externalised configuration), and the
        # service's own start() will call this again. Reusing the existing
        # client avoids leaking connections and clobbering registered scripts.
        if self._client is not None:
            return
        try:
            from redis.asyncio import Redis
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("redis package is required for the redis storage backend") from exc

        self._client = Redis.from_url(self._url, encoding="utf-8", decode_responses=True)
        await self._client.ping()
        self._incr_window_script = self._client.register_script(_INCR_WINDOW_LUA)
        self._add_usage_script = self._client.register_script(_ADD_USAGE_LUA)
        self._claim_record_script = self._client.register_script(_CLAIM_RECORD_LUA)
        # Ensure the stream and this replica's consumer group exist before any
        # publish() or consume() call. Creating it here (rather than lazily in
        # the consumer) means a publish() before any subscriber has started
        # still lands in a real stream that late joiners will read from their
        # group's last-delivered-id onwards.
        await self._ensure_consumer_group()

    async def _ensure_consumer_group(self) -> None:
        """Create the config-reload stream and this replica's consumer group.

        New groups start from ``$`` (only future messages): a freshly-joined
        replica does not replay historical notifications — it has already
        loaded the authoritative configuration from storage at startup, so
        replaying old versions would be a no-op (they would all be <= the
        applied version and get skipped by the watcher anyway).
        """
        if self._consumer_group_ready:
            return
        try:
            from redis.exceptions import ResponseError
        except ImportError:  # pragma: no cover - optional dependency
            ResponseError = Exception
        try:
            await self._client.xgroup_create(
                self._k(self._CONFIG_RELOAD_STREAM),
                self._replica_id,
                id="$",
                mkstream=True,
            )
        except ResponseError as exc:
            # BUSYGROUP means the group already exists (e.g. this replica is
            # restarting). That is the common case on restart; the group's
            # last-delivered-id is preserved, so we resume exactly where we
            # left off and the pending list still holds any unacked messages
            # for XAUTOCLAIM to recover.
            if "BUSYGROUP" not in str(exc):
                raise
        self._consumer_group_ready = True

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

    # -- session affinity ------------------------------------------------

    async def get_session_affinity(self, session_id: str) -> SessionAffinityBinding | None:
        client = self._require_client()
        raw = await client.get(self._k(f"affinity:{session_id}"))
        if not raw:
            return None
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # Tolerate a plain-string binding written by an older revision.
            return SessionAffinityBinding(model_id=raw, prefix_tokens=0, ttl_remaining=0)
        ttl = await client.ttl(self._k(f"affinity:{session_id}"))
        # Redis returns -1 (no expiry) or -2 (missing); neither should be capped
        # to the cache TTL, so treat them as "no live binding".
        ttl_remaining = int(ttl) if ttl and ttl > 0 else 0
        return SessionAffinityBinding(
            model_id=value.get("model"),
            prefix_tokens=int(value.get("prefix_tokens", 0) or 0),
            ttl_remaining=ttl_remaining,
        )

    async def set_session_affinity(
        self, session_id: str, model_id: str, prefix_tokens: int, ttl_seconds: int
    ) -> None:
        client = self._require_client()
        payload = json.dumps({"model": model_id, "prefix_tokens": int(prefix_tokens)})
        await client.set(self._k(f"affinity:{session_id}"), payload, ex=ttl_seconds)

    @staticmethod
    def _decode_record(raw: Any) -> dict[str, Any] | None:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    # -- externalised configuration -------------------------------------

    # Two keys: payload (JSON) and a monotonic version counter. The version
    # is bumped atomically with INCR before the payload write so a reader
    # observing a new version is guaranteed to find the matching payload
    # already committed (writes are ordered: version, then payload). Readers
    # that see a payload without a version (race during bootstrap) treat it
    # as version 0, which simply triggers a reload on the next bump.

    _CONFIG_PAYLOAD_KEY = "config:payload"
    _CONFIG_VERSION_KEY = "config:version"

    async def get_config(self) -> dict[str, Any] | None:
        client = self._require_client()
        raw = await client.get(self._k(self._CONFIG_PAYLOAD_KEY))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    async def set_config(self, data: dict[str, Any]) -> int:
        client = self._require_client()
        # Bump version first; readers checking version before payload will
        # wait for the next poll, readers checking payload will see the new
        # value only after the subsequent SET, which we issue right here.
        version = await client.incr(self._k(self._CONFIG_VERSION_KEY))
        payload = json.dumps(data, ensure_ascii=False)
        await client.set(self._k(self._CONFIG_PAYLOAD_KEY), payload)
        return int(version)

    async def get_config_version(self) -> int:
        client = self._require_client()
        value = await client.get(self._k(self._CONFIG_VERSION_KEY))
        return int(value) if value else 0

    # -- cross-replica reload notifications (Redis Streams) --------------

    # Stream holding config-reload notifications. Each entry's sole field is
    # the new config version as a string; subscribers pull the actual config
    # document from the payload key. MAXLEN ~ 100 trims the stream so it does
    # not grow without bound; 100 entries is far more than any realistic
    # catch-up window (a replica would have to miss 100 successive config
    # writes before the oldest one is evicted).
    _CONFIG_RELOAD_STREAM = "config:reload"

    # Messages delivered to a consumer but not acked within this idle time
    # (ms) become candidates for XAUTOCLAIM. We use 60s — long enough that a
    # healthy in-progress reload never gets stolen mid-flight, short enough
    # that a crashed replica's pending messages are recovered within a minute
    # of the next subscriber starting.
    # Block for at most this long (ms) on XREADGROUP before looping back to let
    # the async runtime service other tasks (e.g. shutdown cancellation).
    _READ_BLOCK_MS = 1_000

    async def publish_config_reload(self, version: int) -> None:
        client = self._require_client()
        # XADD with MAXLEN ~ trims approximately to 100 entries (the "~" means
        # Redis may keep a few more for efficiency). Returns the entry id,
        # which we don't need: subscribers identify messages by their position
        # in the stream, not by an application-level id.
        await client.xadd(
            self._k(self._CONFIG_RELOAD_STREAM),
            {"version": str(version)},
            maxlen=100,
            approximate=True,
        )

    async def _config_reload_iterator(self) -> collections.abc.AsyncIterator[int]:
        # A subscriber connection blocks on XREADGROUP, so it must not share
        # the command client. We open a fresh client here and close it when
        # the iterator is torn down. The consumer group was already created
        # by start() on the main client; we only consume here.
        from redis.asyncio import Redis

        client = Redis.from_url(self._url, encoding="utf-8", decode_responses=True)
        stream_key = self._k(self._CONFIG_RELOAD_STREAM)
        try:
            # 1. Recover any messages delivered to this replica (under the
            #    same consumer name) but never acked — e.g. because the
            #    process crashed between XREADGROUP and XACK. XAUTOCLAIM with
            #    min_idle_time=0 re-delivers everything currently pending for
            #    this group; the version check in the watcher dedups any we
            #    already applied before the crash.
            for version, entry_id in await self._claim_pending(client, stream_key):
                yield version
                await client.xack(stream_key, self._replica_id, entry_id)

            # 2. Enter the blocking read loop for new messages.
            while True:
                # XREADGROUP with the special id ">" reads only messages this
                # consumer has never seen. BLOCK lets the coroutine yield to
                # the event loop periodically so shutdown cancellation lands.
                messages = await client.xreadgroup(
                    self._replica_id,
                    self._replica_id,
                    {stream_key: ">"},
                    count=10,
                    block=self._READ_BLOCK_MS,
                )
                if not messages:
                    # A BLOCK timeout with no messages: loop and block again.
                    # Cancellation lands as CancelledError on the next await.
                    continue
                for _stream, entries in messages:
                    for entry_id, fields in entries:
                        version = self._parse_version(fields)
                        if version is not None:
                            yield version
                        # Ack regardless of whether we yielded: a malformed
                        # entry (version is None) must not pin the pending
                        # list forever.
                        await client.xack(stream_key, self._replica_id, entry_id)
        except asyncio.CancelledError:
            raise
        finally:
            await client.aclose()

    async def _claim_pending(
        self, client: Any, stream_key: str
    ) -> list[tuple[int, str]]:
        """Return (version, entry_id) pairs for messages never acked.

        Called once at the start of the consume loop. Uses XAUTOCLAIM with
        min_idle_time=0 to grab the entire pending entries list for this
        consumer group. The caller yields the version to the watcher (which
        dedups via the applied-version check) and acks the entry id. This is
        the recovery path for a crash between delivery and ack.
        """
        try:
            from redis.exceptions import ResponseError
        except ImportError:  # pragma: no cover
            ResponseError = Exception

        claimed_pairs: list[tuple[int, str]] = []
        cursor = "0-0"
        while True:
            try:
                # XAUTOCLAIM returns (next_cursor, claimed_entries, deleted_ids).
                result = await client.xautoclaim(
                    stream_key,
                    self._replica_id,
                    self._replica_id,
                    min_idle_time=0,
                    count=100,
                    start_id=cursor,
                )
            except ResponseError as exc:
                # NOGROUP means the group was lost (e.g. stream evicted by
                # MAXLEN before anyone re-created it). Recreate it from "$"
                # and move on — the periodic reconciler will catch up the
                # version drift.
                if "NOGROUP" in str(exc):
                    await self._ensure_consumer_group()
                    return claimed_pairs
                raise
            if not result or len(result) < 2:
                return claimed_pairs
            next_cursor, claimed = result[0], result[1]
            for entry_id, fields in claimed:
                version = self._parse_version(fields)
                if version is not None:
                    claimed_pairs.append((version, entry_id))
                else:
                    # Malformed entry: ack immediately so it doesn't pin
                    # the pending list forever.
                    await client.xack(stream_key, self._replica_id, entry_id)
            # cursor "0-0" means we've scanned the whole pending list.
            if next_cursor in ("0-0", b"0-0"):
                return claimed_pairs
            cursor = next_cursor

    @staticmethod
    def _parse_version(fields: Any) -> int | None:
        """Extract the version number from a stream entry's fields."""
        if not isinstance(fields, dict):
            return None
        raw = fields.get("version")
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    def config_reload_events(self) -> collections.abc.AsyncIterator[int]:
        return self._config_reload_iterator()
