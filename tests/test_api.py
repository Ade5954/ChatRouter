"""End-to-end tests through the HTTP surface with mocked upstreams."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from chatrouter.app import create_app

from .conftest import make_config

UPSTREAM = "https://p1.test/v1/chat/completions"
AUTH = {"Authorization": "Bearer sk-test-acme"}


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


def sse_stream() -> bytes:
    chunks = [
        {"id": "x", "choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"id": "x", "choices": [{"index": 0, "delta": {"content": "Hi"}}]},
        {
            "id": "x",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + "data: [DONE]\n\n").encode()


@pytest.fixture
def client():
    app = create_app(make_config())
    with TestClient(app) as test_client:
        yield test_client


class TestAuth:
    def test_missing_or_invalid_key_rejected(self, client):
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
                headers=headers,
            )
            assert response.status_code == 401

    def test_x_api_key_header_accepted(self, client):
        with respx.mock:
            respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=completion_body()))
            response = client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
                headers={"x-api-key": "sk-test-acme"},
            )
        assert response.status_code == 200


class TestChatCompletions:
    def test_basic_completion(self, client):
        with respx.mock:
            respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=completion_body()))
            response = client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
                headers=AUTH,
            )
        assert response.status_code == 200
        body = response.json()
        assert body["choices"][0]["message"]["content"] == "hello"
        assert "x-chatrouter-model" in response.headers
        assert "x-chatrouter-request-id" in response.headers

    def test_routing_headers_and_model_mapping(self, client):
        """Clients see the gateway model id; headers expose the decision."""
        with respx.mock:
            # Explicit model: upstream receives the mapped name, no assessment.
            route = respx.post(UPSTREAM).mock(
                return_value=httpx.Response(200, json=completion_body())
            )
            explicit = client.post(
                "/v1/chat/completions",
                json={"model": "strong", "messages": [{"role": "user", "content": "hi"}]},
                headers=AUTH,
            )
            sent = json.loads(route.calls[0].request.content)
            assert explicit.json()["model"] == "strong"
            assert sent["model"] == "strong-1"

            # Auto routing: decision headers carry the complexity assessment.
            auto = client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
                headers=AUTH,
            )
        assert auto.headers["x-chatrouter-routing-reason"]
        assert 0 <= float(auto.headers["x-chatrouter-complexity"]) <= 1

    def test_invalid_requests_rejected(self, client):
        assert (
            client.post(
                "/v1/chat/completions", json={"model": "auto", "messages": []}, headers=AUTH
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "hi"}], "n": 3},
                headers=AUTH,
            ).status_code
            == 400
        )

    def test_unknown_model_returns_404(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
            headers=AUTH,
        )
        assert response.status_code == 404


class TestStreaming:
    def test_stream_relays_and_rewrites_model(self, client):
        with respx.mock:
            respx.post(UPSTREAM).mock(
                return_value=httpx.Response(
                    200, content=sse_stream(), headers={"content-type": "text/event-stream"}
                )
            )
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "cheap",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
                headers=AUTH,
            ) as response:
                assert response.status_code == 200
                body = "".join(response.iter_text())
        assert "data:" in body
        assert "[DONE]" in body
        assert "Hi" in body
        events = [
            json.loads(line[5:])
            for line in body.splitlines()
            if line.startswith("data:") and "[DONE]" not in line
        ]
        assert all(event["model"] == "cheap" for event in events)


class TestFailover:
    def test_retries_on_5xx_then_succeeds(self, client):
        with respx.mock:
            respx.post(UPSTREAM).mock(
                side_effect=[
                    httpx.Response(503, json={"error": {"message": "overloaded"}}),
                    httpx.Response(200, json=completion_body()),
                ]
            )
            response = client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
                headers=AUTH,
            )
        assert response.status_code == 200

    def test_client_error_not_retried(self, client):
        """A 400 will fail identically on every model, so fail fast."""
        with respx.mock:
            route = respx.post(UPSTREAM).mock(
                return_value=httpx.Response(400, json={"error": {"message": "bad request"}})
            )
            response = client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
                headers=AUTH,
            )
        assert response.status_code == 400
        assert route.call_count == 1

    def test_all_failures_return_502(self, client):
        with respx.mock:
            respx.post(UPSTREAM).mock(
                return_value=httpx.Response(503, json={"error": {"message": "down"}})
            )
            response = client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
                headers=AUTH,
            )
        assert response.status_code == 502


class TestRateLimitingEndToEnd:
    def test_rpm_limit_returns_429(self):
        from chatrouter.config.models import QuotaConfig, RateLimitConfig, TenantConfig

        config = make_config(
            tenants=[
                TenantConfig(
                    id="acme",
                    api_keys=["sk-test-acme"],
                    rate_limit=RateLimitConfig(rpm=2),
                    quota=QuotaConfig(),
                )
            ]
        )
        app = create_app(config)
        with TestClient(app) as client, respx.mock:
            respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=completion_body()))
            payload = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
            ok = client.post("/v1/chat/completions", json=payload, headers=AUTH)
            assert ok.status_code == 200
            assert "x-ratelimit-limit-requests" in ok.headers
            assert client.post("/v1/chat/completions", json=payload, headers=AUTH).status_code == 200
            limited = client.post("/v1/chat/completions", json=payload, headers=AUTH)
            assert limited.status_code == 429
            assert "Retry-After" in limited.headers


class TestModelsEndpoint:
    def test_lists_models_and_aliases(self, client):
        response = client.get("/v1/models", headers=AUTH)
        assert response.status_code == 200
        ids = [m["id"] for m in response.json()["data"]]
        assert "auto" in ids
        assert "mid" in ids

    def test_retrieve_single_model(self, client):
        response = client.get("/v1/models/cheap", headers=AUTH)
        assert response.status_code == 200
        assert response.json()["id"] == "cheap"

    def test_retrieve_unknown_model_404(self, client):
        assert client.get("/v1/models/ghost", headers=AUTH).status_code == 404


class TestFeedbackEndpoint:
    def test_feedback_round_trip(self, client):
        with respx.mock:
            respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=completion_body()))
            completion = client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
                headers=AUTH,
            )
        request_id = completion.headers["x-chatrouter-request-id"]

        response = client.post(
            "/v1/feedback", json={"request_id": request_id, "thumb": "down"}, headers=AUTH
        )
        assert response.status_code == 200
        body = response.json()
        assert body["accepted"] is True
        assert body["applied_score"] == 0.0

    def test_unknown_request_id_is_discarded(self, client):
        response = client.post(
            "/v1/feedback", json={"request_id": "nope", "score": 1.0}, headers=AUTH
        )
        assert response.status_code == 200
        assert response.json()["accepted"] is False

    def test_feedback_cannot_be_replayed(self, client):
        """A request_id must be scoreable exactly once; replays must not inflate
        feedback_count (the confidence weight for adaptive routing)."""
        with respx.mock:
            respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=completion_body()))
            completion = client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
                headers=AUTH,
            )
        request_id = completion.headers["x-chatrouter-request-id"]
        model_id = completion.headers["x-chatrouter-model"]

        first = client.post(
            "/v1/feedback", json={"request_id": request_id, "thumb": "down"}, headers=AUTH
        )
        assert first.json()["accepted"] is True

        for _ in range(5):
            replay = client.post(
                "/v1/feedback", json={"request_id": request_id, "thumb": "down"}, headers=AUTH
            )
            assert replay.status_code == 200
            assert replay.json()["accepted"] is False

        service = client.app.state.service
        stats = asyncio.run(service.feedback.get_stats(model_id, None))
        assert stats.feedback_count == 1

    def test_feedback_without_signal_rejected(self, client):
        response = client.post("/v1/feedback", json={"request_id": "x"}, headers=AUTH)
        assert response.status_code == 400


class TestExplainEndpoint:
    def test_explain_returns_decision_without_upstream_call(self, client):
        with respx.mock:
            route = respx.post(UPSTREAM).mock(
                return_value=httpx.Response(200, json=completion_body())
            )
            response = client.post(
                "/v1/routing/explain",
                json={
                    "model": "auto",
                    "messages": [
                        {"role": "user", "content": "Prove this theorem with full derivations."}
                    ],
                },
                headers=AUTH,
            )
        assert response.status_code == 200
        decision = response.json()["decision"]
        assert decision["model"]
        assert decision["assessment"]["score"] > 0
        assert decision["candidates"]
        assert route.call_count == 0


class TestOpsEndpoints:
    def test_health_and_readiness(self, client):
        assert client.get("/healthz").json()["status"] == "ok"
        assert client.get("/readyz").status_code == 200

    def test_metrics_exposed(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_admin_requires_key(self, client):
        assert client.get("/admin/status").status_code == 401

    def test_admin_status_and_config(self, client):
        response = client.get("/admin/status", headers={"x-admin-key": "admin-secret"})
        assert response.status_code == 200
        assert "models" in response.json()

        config = client.get("/admin/config", headers={"x-admin-key": "admin-secret"})
        assert config.status_code == 200
        body = config.json()
        assert all("api_key" not in p or p.get("api_key") is None for p in body["providers"])
        for tenant in body["tenants"]:
            for key in tenant["api_keys"]:
                assert key.startswith("***")
