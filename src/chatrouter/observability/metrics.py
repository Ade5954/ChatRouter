"""Prometheus metrics.

All instruments degrade to no-ops when ``prometheus_client`` is absent, so the
gateway never hard-depends on the metrics stack.
"""

from __future__ import annotations

from typing import Any

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"


class _NoopMetric:
    """Stand-in that swallows every metric operation."""

    def labels(self, *args: Any, **kwargs: Any) -> _NoopMetric:
        return self

    def inc(self, *args: Any, **kwargs: Any) -> None:
        return None

    def dec(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set(self, *args: Any, **kwargs: Any) -> None:
        return None

    def observe(self, *args: Any, **kwargs: Any) -> None:
        return None


def _counter(name: str, doc: str, labels: list[str]) -> Any:
    return Counter(name, doc, labels) if _AVAILABLE else _NoopMetric()


def _histogram(name: str, doc: str, labels: list[str], buckets: tuple[float, ...]) -> Any:
    return Histogram(name, doc, labels, buckets=buckets) if _AVAILABLE else _NoopMetric()


def _gauge(name: str, doc: str, labels: list[str]) -> Any:
    return Gauge(name, doc, labels) if _AVAILABLE else _NoopMetric()


_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
_TOKEN_BUCKETS = (64, 256, 1024, 4096, 16384, 65536, 262144)
_SCORE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

REQUESTS_TOTAL = _counter(
    "chatrouter_requests_total",
    "Total gateway requests.",
    ["tenant", "model", "status"],
)

REQUEST_DURATION = _histogram(
    "chatrouter_request_duration_seconds",
    "End-to-end gateway request duration.",
    ["tenant", "model"],
    _LATENCY_BUCKETS,
)

TIME_TO_FIRST_TOKEN = _histogram(
    "chatrouter_time_to_first_token_seconds",
    "Time until the first streamed token.",
    ["model"],
    _LATENCY_BUCKETS,
)

TOKENS_TOTAL = _counter(
    "chatrouter_tokens_total",
    "Tokens processed, split by direction.",
    ["tenant", "model", "direction"],
)

PROMPT_TOKENS = _histogram(
    "chatrouter_prompt_tokens",
    "Prompt size distribution.",
    ["model"],
    _TOKEN_BUCKETS,
)

COST_TOTAL = _counter(
    "chatrouter_cost_usd_total",
    "Accumulated upstream spend in USD.",
    ["tenant", "model"],
)

ROUTING_DECISIONS = _counter(
    "chatrouter_routing_decisions_total",
    "Routing decisions by reason and selected tier.",
    ["reason", "tier", "model"],
)

COMPLEXITY_SCORE = _histogram(
    "chatrouter_complexity_score",
    "Distribution of conversation complexity scores.",
    ["tier"],
    _SCORE_BUCKETS,
)

CONTEXT_ESCALATIONS = _counter(
    "chatrouter_context_escalations_total",
    "Requests whose tier was raised by full-context analysis.",
    ["tier"],
)

RATE_LIMITED = _counter(
    "chatrouter_rate_limited_total",
    "Requests rejected by rate limiting.",
    ["tenant", "kind"],
)

QUOTA_EVENTS = _counter(
    "chatrouter_quota_events_total",
    "Quota outcomes such as rejection or downgrade.",
    ["tenant", "action"],
)

CONTEXT_TRIMMED = _counter(
    "chatrouter_context_trimmed_total",
    "Conversations whose history was trimmed to fit the context window.",
    ["model"],
)

OVERFLOW_EVENTS = _counter(
    "chatrouter_overflow_total",
    "Requests rescheduled because the preferred model was saturated.",
    ["from_model", "to_model"],
)

FAILOVER_EVENTS = _counter(
    "chatrouter_failover_total",
    "Requests that degraded to another model after a failure.",
    ["from_model", "to_model"],
)

UPSTREAM_ERRORS = _counter(
    "chatrouter_upstream_errors_total",
    "Upstream call failures.",
    ["model", "status"],
)

CIRCUIT_STATE = _gauge(
    "chatrouter_circuit_state",
    "Circuit breaker state (0=closed, 1=half-open, 2=open).",
    ["model"],
)

MODEL_INFLIGHT = _gauge(
    "chatrouter_model_inflight",
    "In-flight requests per model.",
    ["model"],
)

MODEL_QUALITY = _gauge(
    "chatrouter_model_quality",
    "Feedback-adjusted quality score per model.",
    ["model"],
)

FEEDBACK_TOTAL = _counter(
    "chatrouter_feedback_total",
    "Explicit feedback submissions.",
    ["model", "polarity"],
)


def render_metrics() -> tuple[bytes, str]:
    """Render the metrics exposition payload."""
    if not _AVAILABLE:
        return b"# prometheus_client is not installed\n", CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST


def metrics_available() -> bool:
    return _AVAILABLE
