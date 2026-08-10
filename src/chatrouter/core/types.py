"""Internal domain types shared across routing, governance and dispatch."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..config.models import ModelConfig, ModelTier, TenantConfig
from .schemas import ChatCompletionRequest


class RoutingDecisionReason(str, Enum):
    """Why the router picked the model it picked."""

    EXPLICIT_MODEL = "explicit_model"
    PINNED = "pinned"
    CONTEXT_AWARE = "context_aware"
    FEEDBACK_ADAPTIVE = "feedback_adaptive"
    EXPLORATION = "exploration"
    OVERFLOW = "overflow"
    QUOTA_DOWNGRADE = "quota_downgrade"
    TENANT_CEILING = "tenant_ceiling"
    FALLBACK_CHAIN = "fallback_chain"
    DEFAULT_MODEL = "default_model"


@dataclass(slots=True)
class ComplexitySignals:
    """Per-signal breakdown of the conversation complexity analysis."""

    conversation_depth: float = 0.0
    context_length: float = 0.0
    reasoning_keywords: float = 0.0
    code_content: float = 0.0
    structured_output: float = 0.0
    tool_usage: float = 0.0
    multilingual: float = 0.0
    unresolved_thread: float = 0.0
    instruction_density: float = 0.0
    requested_output_size: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "conversation_depth": self.conversation_depth,
            "context_length": self.context_length,
            "reasoning_keywords": self.reasoning_keywords,
            "code_content": self.code_content,
            "structured_output": self.structured_output,
            "tool_usage": self.tool_usage,
            "multilingual": self.multilingual,
            "unresolved_thread": self.unresolved_thread,
            "instruction_density": self.instruction_density,
            "requested_output_size": self.requested_output_size,
        }


@dataclass(slots=True)
class ComplexityAssessment:
    """Result of analysing the full conversation."""

    score: float
    tier: ModelTier
    signals: ComplexitySignals
    prompt_tokens_estimate: int
    turn_count: int
    requires_vision: bool = False
    requires_tools: bool = False
    # Signals whose evidence came from earlier turns rather than the last one.
    latent_escalation: float = 0.0
    explanation: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "tier": self.tier.value,
            "signals": {k: round(v, 4) for k, v in self.signals.as_dict().items()},
            "prompt_tokens_estimate": self.prompt_tokens_estimate,
            "turn_count": self.turn_count,
            "requires_vision": self.requires_vision,
            "requires_tools": self.requires_tools,
            "latent_escalation": round(self.latent_escalation, 4),
            "explanation": self.explanation,
        }


@dataclass(slots=True)
class ScoredCandidate:
    """A model considered by the router, with its utility breakdown."""

    model: ModelConfig
    utility: float
    quality: float
    cost_score: float
    latency_score: float
    load_score: float
    tier_penalty: float
    exploration_bonus: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.id,
            "tier": self.model.tier.value,
            "utility": round(self.utility, 4),
            "quality": round(self.quality, 4),
            "cost_score": round(self.cost_score, 4),
            "latency_score": round(self.latency_score, 4),
            "load_score": round(self.load_score, 4),
            "tier_penalty": round(self.tier_penalty, 4),
            "exploration_bonus": round(self.exploration_bonus, 4),
        }


@dataclass(slots=True)
class RoutingDecision:
    """The routing outcome handed to the dispatcher."""

    model: ModelConfig
    reason: RoutingDecisionReason
    assessment: ComplexityAssessment | None
    # Ordered failover chain used when the primary target fails.
    fallback_chain: list[ModelConfig] = field(default_factory=list)
    candidates: list[ScoredCandidate] = field(default_factory=list)
    exploration: bool = False
    notes: list[str] = field(default_factory=list)

    def chain(self) -> list[ModelConfig]:
        """Primary model followed by its fallbacks, de-duplicated."""
        seen: set[str] = set()
        ordered: list[ModelConfig] = []
        for model in (self.model, *self.fallback_chain):
            if model.id not in seen:
                seen.add(model.id)
                ordered.append(model)
        return ordered

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.id,
            "reason": self.reason.value,
            "exploration": self.exploration,
            "fallback_chain": [m.id for m in self.fallback_chain],
            "candidates": [c.as_dict() for c in self.candidates],
            "assessment": self.assessment.as_dict() if self.assessment else None,
            "notes": self.notes,
        }


@dataclass(slots=True)
class RequestContext:
    """Everything the pipeline needs to know about one inbound request."""

    request_id: str
    tenant: TenantConfig
    request: ChatCompletionRequest
    session_id: str | None = None
    client_ip: str | None = None
    received_at: float = field(default_factory=time.monotonic)
    # Populated as the request travels through the pipeline.
    decision: RoutingDecision | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)
    quota_downgraded: bool = False
    # Set when the completion was served from the exact-match cache, so the API
    # layer can advertise it and the caller knows no fresh generation occurred.
    cache_hit: bool = False

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.received_at) * 1000


@dataclass(slots=True)
class AttemptRecord:
    """One upstream call attempt (used for retries and feedback)."""

    model_id: str
    started_at: float
    finished_at: float | None = None
    success: bool = False
    status_code: int | None = None
    error: str | None = None
    first_token_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncated: bool = False

    @property
    def latency_ms(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at) * 1000


@dataclass(slots=True)
class DispatchResult:
    """Outcome of a completed non-streaming dispatch."""

    model: ModelConfig
    payload: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    attempts: int
    truncated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class ModelRuntimeStats:
    """Live health and quality statistics for one model."""

    model_id: str
    requests: int = 0
    successes: int = 0
    failures: int = 0
    quality_ema: float | None = None
    latency_ema_ms: float | None = None
    feedback_count: int = 0
    inflight: int = 0
    updated_at: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        if self.requests == 0:
            return 1.0
        return self.successes / self.requests

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "requests": self.requests,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": round(self.success_rate, 4),
            "quality_ema": round(self.quality_ema, 4) if self.quality_ema is not None else None,
            "latency_ema_ms": round(self.latency_ema_ms, 1) if self.latency_ema_ms is not None else None,
            "feedback_count": self.feedback_count,
            "inflight": self.inflight,
        }


def estimate_cost_usd(model: ModelConfig, prompt_tokens: int, completion_tokens: int) -> float:
    """Cost of one call in USD given the model price sheet."""
    return (
        prompt_tokens / 1000 * model.input_cost_per_1k
        + completion_tokens / 1000 * model.output_cost_per_1k
    )
