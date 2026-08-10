"""Feedback normalisation: collapse every feedback idiom into a [0,1] score.

These tests pin the behaviour of :class:`FeedbackNormalizer` and its contract
with :class:`FeedbackRequest` / :meth:`GatewayService.submit_feedback`. They are
deliberately pure (no upstream mocks) except where the HTTP path is exercised.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from chatrouter.app import create_app
from chatrouter.config.models import FeedbackConfig, FeedbackNormalizationConfig
from chatrouter.core.schemas import FeedbackRequest, FeedbackResponse
from chatrouter.routing.feedback_normalizer import FeedbackNormalizer

from .conftest import make_config

UPSTREAM = "https://p1.test/v1/chat/completions"
AUTH = {"Authorization": "Bearer sk-test-acme"}


# -- pure normalizer --------------------------------------------------------


def test_explicit_score_passthrough():
    cfg = FeedbackNormalizationConfig()
    n = FeedbackNormalizer(cfg)
    result = n.normalize(FeedbackRequest(request_id="r", score=0.37))
    assert result.score == pytest.approx(0.37)
    assert result.source == "score"


def test_rating_linear_mapping():
    cfg = FeedbackNormalizationConfig()
    n = FeedbackNormalizer(cfg)
    assert n.normalize(FeedbackRequest(request_id="r", rating=1)).score == 0.0
    assert n.normalize(FeedbackRequest(request_id="r", rating=3)).score == pytest.approx(0.5)
    assert n.normalize(FeedbackRequest(request_id="r", rating=5)).score == 1.0


def test_thumb_mapping_defaults():
    cfg = FeedbackNormalizationConfig()
    n = FeedbackNormalizer(cfg)
    up = n.normalize(FeedbackRequest(request_id="r", thumb="up"))
    assert up.score == pytest.approx(1.0) and up.source == "thumb"
    down = n.normalize(FeedbackRequest(request_id="r", thumb="down"))
    assert down.score == pytest.approx(0.0) and down.source == "thumb"


def test_accepted_mapping_defaults():
    cfg = FeedbackNormalizationConfig()
    n = FeedbackNormalizer(cfg)
    acc = n.normalize(FeedbackRequest(request_id="r", accepted=True))
    assert acc.score == pytest.approx(1.0) and acc.source == "accepted"
    rej = n.normalize(FeedbackRequest(request_id="r", accepted=False))
    assert rej.score == pytest.approx(0.0) and rej.source == "accepted"


def test_behavioural_signals_fallback():
    cfg = FeedbackNormalizationConfig()
    n = FeedbackNormalizer(cfg)
    reg = n.normalize(FeedbackRequest(request_id="r", regenerated=True))
    assert reg.score == pytest.approx(0.2) and reg.source == "regenerated"
    # edited wins over regenerated only when regenerated is absent
    ed = n.normalize(FeedbackRequest(request_id="r", edited=True))
    assert ed.score == pytest.approx(0.5) and ed.source == "edited"


def test_no_signal_is_none():
    n = FeedbackNormalizer(FeedbackNormalizationConfig())
    assert n.normalize(FeedbackRequest(request_id="r")) is None


def test_priority_ordering():
    """More deliberate signals win when several are present."""
    n = FeedbackNormalizer(FeedbackNormalizationConfig())
    # thumb + rating + regenerated -> rating should win (higher priority)
    fb = FeedbackRequest(request_id="r", rating=4, thumb="down", regenerated=True)
    normalized = n.normalize(fb)
    assert normalized.source == "rating"
    assert normalized.score == pytest.approx(0.75)


def test_config_driven_scores():
    """Operators can retune the mapping without touching code."""
    cfg = FeedbackNormalizationConfig(
        thumb_up_score=0.9, thumb_down_score=0.1, regenerated_score=0.4
    )
    n = FeedbackNormalizer(cfg)
    assert n.normalize(FeedbackRequest(request_id="r", thumb="up")).score == 0.9
    assert n.normalize(FeedbackRequest(request_id="r", thumb="down")).score == 0.1
    assert n.normalize(FeedbackRequest(request_id="r", regenerated=True)).score == 0.4


def test_from_feedback_config_wires_defaults():
    fc = FeedbackConfig()
    n = FeedbackNormalizer.from_feedback_config(fc)
    assert n.normalize(FeedbackRequest(request_id="r", thumb="up")).score == 1.0


# -- schema delegation ------------------------------------------------------


def test_schema_normalised_score_uses_normalizer():
    # The convenience method must agree with the central normalizer.
    fb = FeedbackRequest(request_id="r", rating=2)
    assert fb.normalised_score() == pytest.approx(0.25)


def test_schema_normalised_score_none_without_signal():
    assert FeedbackRequest(request_id="r").normalised_score() is None


# -- HTTP path: source echoed back -----------------------------------------


@pytest.fixture
def client():
    app = create_app(make_config())
    with TestClient(app) as test_client:
        yield test_client


def _serve(client):
    with respx.mock:
        respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 1, "model": "mid-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }))
        return client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
            headers=AUTH,
        )


def test_response_echoes_source_for_thumb(client):
    response = _serve(client)
    request_id = response.headers["x-chatrouter-request-id"]
    fb = client.post(
        "/v1/feedback", json={"request_id": request_id, "thumb": "down"}, headers=AUTH
    )
    body = fb.json()
    assert body["accepted"] is True
    assert body["applied_score"] == 0.0
    assert body["source"] == "thumb"


def test_response_echoes_source_for_rating(client):
    response = _serve(client)
    request_id = response.headers["x-chatrouter-request-id"]
    fb = client.post(
        "/v1/feedback", json={"request_id": request_id, "rating": 5}, headers=AUTH
    )
    body = fb.json()
    assert body["applied_score"] == 1.0
    assert body["source"] == "rating"


def test_discarded_feedback_has_no_source(client):
    fb = client.post("/v1/feedback", json={"request_id": "nope", "score": 1.0}, headers=AUTH)
    body = fb.json()
    assert body["accepted"] is False
    assert body["source"] is None
    assert body["applied_score"] is None


def test_feedback_response_model_includes_source_field():
    resp = FeedbackResponse(accepted=True, request_id="r", applied_score=0.5, source="edited")
    dumped = resp.model_dump()
    assert dumped["source"] == "edited"
