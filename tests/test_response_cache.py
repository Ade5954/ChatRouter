"""Tests for the exact-match response cache.

The cache must only serve a result when every output-affecting field matches;
otherwise two different prompts would collide. It must also still run the full
accounting path on a hit (quota, billing, the feedback loop) so a cache hit is
indistinguishable from a real generation to the rest of the gateway.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from chatrouter.app import create_app
from chatrouter.config.models import FeedbackConfig, ResponseCacheConfig, RoutingConfig

from .conftest import make_config

UPSTREAM = "https://p1.test/v1/chat/completions"
AUTH = {"Authorization": "Bearer sk-test-acme"}


@pytest.fixture(autouse=True)
def _isolate_respx():
    """Clear any routes left by other test modules so `respx.post` bindings
    in this module are not shadowed by earlier mocks."""
    respx.clear()
    yield
    respx.clear()


def completion_body(model: str = "mid-1", content: str = "hello") -> dict:
    return {
        "id": "chatcmpl-upstream",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }


@pytest.fixture
def cached_client():
    config = make_config(
        routing=RoutingConfig(
            default_model="mid",
            response_cache=ResponseCacheConfig(enabled=True),
            # Disable epsilon-greedy exploration so identical requests always
            # route to the same model. The response cache keys on the resolved
            # model; with exploration on, the same request can land on different
            # models across calls and spuriously miss the cache.
            feedback=FeedbackConfig(enabled=False),
        )
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client


def _ask(client, content: str = "hi", **extra) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": content}], **extra},
        headers=AUTH,
    )


class TestResponseCache:
    def test_identical_request_served_from_cache(self, cached_client):
        """The second identical call must not reach the upstream."""
        upstream_calls = {"n": 0}

        def _count(_):
            upstream_calls["n"] += 1
            return httpx.Response(200, json=completion_body())

        with respx.mock:
            respx.post(UPSTREAM).mock(side_effect=_count)
            first = _ask(cached_client)
            assert first.headers.get("x-chatrouter-cache") is None
            second = _ask(cached_client)

        assert second.status_code == 200
        assert second.headers["x-chatrouter-cache"] == "HIT"
        assert second.json()["choices"][0]["message"]["content"] == "hello"
        # Exactly one upstream call despite two completions.
        assert upstream_calls["n"] == 1

    def test_cache_hit_still_accounts(self, cached_client):
        """A hit must record usage so quota and billing stay accurate."""
        with respx.mock:
            respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=completion_body()))
            _ask(cached_client)
            second = _ask(cached_client)

        assert second.headers["x-chatrouter-cache"] == "HIT"
        service = cached_client.app.state.service
        tenant = cached_client.app.state.config.tenants[0]
        # Quota recorded for both requests: 2 * 20 tokens.
        snap = _run(service.quotas.snapshot(tenant))
        assert snap["tokens"] == 40

    def test_different_messages_miss(self, cached_client):
        calls = {"n": 0}

        def _mk(content):
            def _c(_):
                calls["n"] += 1
                return httpx.Response(200, json=completion_body(content=content))

            return _c

        with respx.mock:
            respx.post(UPSTREAM).mock(
                side_effect=[_mk("one"), _mk("two")]
            )
            a = _ask(cached_client, content="alpha")
            b = _ask(cached_client, content="beta")
        assert a.json()["choices"][0]["message"]["content"] == "one"
        assert b.json()["choices"][0]["message"]["content"] == "two"
        assert calls["n"] == 2

    def test_different_temperature_miss(self, cached_client):
        calls = {"n": 0}

        def _mk(content):
            def _c(_):
                calls["n"] += 1
                return httpx.Response(200, json=completion_body(content=content))

            return _c

        with respx.mock:
            respx.post(UPSTREAM).mock(side_effect=[_mk("t0"), _mk("t1")])
            _ask(cached_client, temperature=0.0)
            other = _ask(cached_client, temperature=1.0)
        assert other.json()["choices"][0]["message"]["content"] == "t1"
        assert calls["n"] == 2

    def test_streaming_never_cached(self, cached_client):
        calls = {"n": 0}
        sse = (
            b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n'
            b"data: [DONE]\n\n"
        )

        def _c(_):
            calls["n"] += 1
            return httpx.Response(200, content=sse)

        with respx.mock:
            respx.post(UPSTREAM).mock(side_effect=_c)
            first = _ask(cached_client, stream=True)
            assert first.status_code == 200
            second = _ask(cached_client, stream=True)
            assert second.status_code == 200
        # Both hit the upstream; streaming bypasses the cache entirely.
        assert calls["n"] == 2

    def test_session_id_bypasses_cache(self, cached_client):
        calls = {"n": 0}

        def _mk(content):
            def _c(_):
                calls["n"] += 1
                return httpx.Response(200, json=completion_body(content=content))

            return _c

        with respx.mock:
            respx.post(UPSTREAM).mock(side_effect=[_mk("s1"), _mk("s2")])
            a = _ask(cached_client, chatrouter={"session_id": "sess-1"})
            b = _ask(cached_client, chatrouter={"session_id": "sess-1"})
        assert a.json()["choices"][0]["message"]["content"] == "s1"
        assert b.json()["choices"][0]["message"]["content"] == "s2"
        assert calls["n"] == 2

    def test_disabled_cache_does_not_short_circuit(self):
        config = make_config()  # response_cache.enabled defaults to False
        app = create_app(config)
        calls = {"n": 0}

        def _mk(content):
            def _c(_):
                calls["n"] += 1
                return httpx.Response(200, json=completion_body(content=content))

            return _c

        with TestClient(app) as client, respx.mock:
            respx.post(UPSTREAM).mock(side_effect=[_mk("a"), _mk("b")])
            _ask(client)
            again = _ask(client)
        assert again.json()["choices"][0]["message"]["content"] == "b"
        assert "x-chatrouter-cache" not in again.headers
        assert calls["n"] == 2

    def test_different_routed_model_miss(self, cached_client):
        """`auto` and an explicit model must not share a cache entry."""
        calls = {"n": 0}

        def _mk(model, content):
            def _c(_):
                calls["n"] += 1
                return httpx.Response(200, json=completion_body(model=model, content=content))

            return _c

        with respx.mock:
            respx.post(UPSTREAM).mock(
                side_effect=[_mk("mid-1", "auto"), _mk("strong-1", "explicit")]
            )
            via_auto = _ask(cached_client, model="auto")
            via_explicit = _ask(cached_client, model="strong")
        assert via_auto.json()["choices"][0]["message"]["content"] == "auto"
        assert via_explicit.json()["choices"][0]["message"]["content"] == "explicit"
        assert calls["n"] == 2


def _run(coro):
    import asyncio

    return asyncio.run(coro)


class TestResponseCacheExpiry:
    def test_entry_expires_after_ttl(self, cached_client, monkeypatch):
        """After the TTL elapses the cache miss path runs and hits upstream."""
        import time

        real_time = time.time
        upstream_calls = {"n": 0}

        def _count(_):
            upstream_calls["n"] += 1
            return httpx.Response(200, json=completion_body())

        # Freeze time at a fixed point so the cache entry is written with a
        # deterministic expiry, then jump past the TTL before the third call.
        frozen = {"t": real_time()}

        def _fake_time():
            return frozen["t"]

        monkeypatch.setattr(time, "time", _fake_time)

        with respx.mock:
            respx.post(UPSTREAM).mock(side_effect=_count)
            # First call populates the cache while time is frozen.
            _ask(cached_client)
            assert upstream_calls["n"] == 1
            # Still within the TTL: the second call is a hit.
            second = _ask(cached_client)
            assert second.headers["x-chatrouter-cache"] == "HIT"
            assert upstream_calls["n"] == 1

            # Jump past the TTL (default 300s) and re-issue.
            frozen["t"] = real_time() + 301
            third = _ask(cached_client)
            assert "x-chatrouter-cache" not in third.headers
            assert upstream_calls["n"] == 2
