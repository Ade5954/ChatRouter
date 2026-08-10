"""Configuration schema for ChatRouter.

The gateway is fully driven by declarative configuration: model pool,
tenants, routing policy and resilience behaviour are all described here.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ModelTier(str, Enum):
    """Capability tier of an upstream model.

    Tiers are ordered: ECONOMY < STANDARD < PREMIUM < REASONING.
    The router maps a task complexity score onto the cheapest tier that is
    still expected to satisfy the request.
    """

    ECONOMY = "economy"
    STANDARD = "standard"
    PREMIUM = "premium"
    REASONING = "reasoning"

    @property
    def rank(self) -> int:
        return _TIER_ORDER[self]


_TIER_ORDER: dict[ModelTier, int] = {
    ModelTier.ECONOMY: 0,
    ModelTier.STANDARD: 1,
    ModelTier.PREMIUM: 2,
    ModelTier.REASONING: 3,
}

TIERS_ASCENDING: list[ModelTier] = sorted(ModelTier, key=lambda t: t.rank)


class ProviderConfig(BaseModel):
    """An upstream OpenAI-compatible API endpoint."""

    name: str
    base_url: str = Field(description="Base URL, e.g. https://api.openai.com/v1")
    api_key_env: str | None = Field(
        default=None,
        description="Environment variable holding the API key for this provider.",
    )
    api_key: str | None = Field(default=None, description="Inline key; prefer api_key_env.")
    timeout_seconds: float = 120.0
    connect_timeout_seconds: float = 10.0
    max_connections: int = 200
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalise_base_url(self) -> ProviderConfig:
        self.base_url = self.base_url.rstrip("/")
        return self


class ModelConfig(BaseModel):
    """A routable target: one model served by one provider."""

    id: str = Field(description="Unique routing id exposed by the gateway.")
    provider: str = Field(description="Name of the provider serving this model.")
    upstream_model: str = Field(description="Model name sent to the upstream API.")
    tier: ModelTier = ModelTier.STANDARD

    # Economics & capability metadata used by the scoring function.
    input_cost_per_1k: float = Field(default=0.0, ge=0.0)
    output_cost_per_1k: float = Field(default=0.0, ge=0.0)
    context_window: int = Field(default=128_000, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    supports_tools: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True

    # Baseline quality prior in [0, 1]; refined online by the feedback loop.
    quality_prior: float = Field(default=0.5, ge=0.0, le=1.0)
    # Expected latency in ms, used for load-aware tie breaking.
    latency_prior_ms: float = Field(default=2_000.0, gt=0.0)

    # Capacity limits for overflow scheduling.
    max_rpm: int | None = Field(default=None, gt=0)
    max_tpm: int | None = Field(default=None, gt=0)
    max_concurrency: int | None = Field(default=None, gt=0)

    weight: float = Field(default=1.0, gt=0.0, description="Static load-balancing weight.")
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)

    @property
    def avg_cost_per_1k(self) -> float:
        """Blended cost assuming a 3:1 prompt/completion ratio."""
        return self.input_cost_per_1k * 0.75 + self.output_cost_per_1k * 0.25


class RateLimitConfig(BaseModel):
    """Token-bucket style limits applied per tenant."""

    rpm: int | None = Field(default=None, gt=0)
    tpm: int | None = Field(default=None, gt=0)
    max_concurrency: int | None = Field(default=None, gt=0)


class QuotaConfig(BaseModel):
    """Rolling-window spend and volume quota for a tenant."""

    period: Literal["hour", "day", "month"] = "day"
    max_tokens: int | None = Field(default=None, gt=0)
    max_requests: int | None = Field(default=None, gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0.0)
    # When exceeded: reject the call, or degrade to the cheapest allowed tier.
    on_exceed: Literal["reject", "downgrade"] = "reject"


class TenantConfig(BaseModel):
    """An API consumer with its own keys, limits and routing constraints."""

    id: str
    name: str = ""
    api_keys: list[str] = Field(default_factory=list)
    enabled: bool = True

    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    quota: QuotaConfig = Field(default_factory=QuotaConfig)

    allowed_models: list[str] = Field(
        default_factory=list, description="Empty means every enabled model is allowed."
    )
    denied_models: list[str] = Field(default_factory=list)
    max_tier: ModelTier | None = Field(
        default=None, description="Hard ceiling on the tier this tenant may reach."
    )
    # Per-tenant override of the router's cost/quality trade-off.
    quality_bias: float | None = Field(default=None, ge=0.0, le=1.0)
    priority: int = Field(default=0, description="Higher priority wins under contention.")

    @model_validator(mode="after")
    def _default_name(self) -> TenantConfig:
        if not self.name:
            self.name = self.id
        return self


class ComplexitySignalWeights(BaseModel):
    """Weights of the individual signals feeding the complexity score.

    Every signal produces a value in [0, 1]; the final score is a weighted
    mean, so the weights do not need to sum to one.
    """

    conversation_depth: float = Field(default=1.0, ge=0.0)
    context_length: float = Field(default=1.0, ge=0.0)
    reasoning_keywords: float = Field(default=1.5, ge=0.0)
    code_content: float = Field(default=1.2, ge=0.0)
    structured_output: float = Field(default=0.8, ge=0.0)
    tool_usage: float = Field(default=1.0, ge=0.0)
    multilingual: float = Field(default=0.5, ge=0.0)
    unresolved_thread: float = Field(default=1.3, ge=0.0)
    instruction_density: float = Field(default=0.9, ge=0.0)
    requested_output_size: float = Field(default=0.7, ge=0.0)


class ContextRoutingConfig(BaseModel):
    """Controls how much of the dialogue history shapes the decision."""

    enabled: bool = True
    max_messages_analysed: int = Field(default=40, gt=0)
    # Older turns matter less; weight of turn i is decay ** (distance from tail).
    recency_decay: float = Field(default=0.85, gt=0.0, le=1.0)
    # A latent-complexity carry-over: once a thread proves hard, keep it hard.
    escalation_memory: float = Field(default=0.6, ge=0.0, le=1.0)
    # Fraction of context window usage that already implies a demanding task.
    context_pressure_threshold: float = Field(default=0.35, gt=0.0, le=1.0)
    signal_weights: ComplexitySignalWeights = Field(default_factory=ComplexitySignalWeights)


class TierThresholds(BaseModel):
    """Complexity score cut-offs mapping a score onto a tier."""

    economy_max: float = Field(default=0.25, ge=0.0, le=1.0)
    standard_max: float = Field(default=0.55, ge=0.0, le=1.0)
    premium_max: float = Field(default=0.80, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_monotonic(self) -> TierThresholds:
        if not self.economy_max < self.standard_max < self.premium_max:
            raise ValueError("tier thresholds must be strictly increasing")
        return self

    def tier_for(self, score: float) -> ModelTier:
        if score <= self.economy_max:
            return ModelTier.ECONOMY
        if score <= self.standard_max:
            return ModelTier.STANDARD
        if score <= self.premium_max:
            return ModelTier.PREMIUM
        return ModelTier.REASONING


class FeedbackNormalizationConfig(BaseModel):
    """How heterogeneous feedback shapes collapse into a [0, 1] quality score.

    Clients report satisfaction in many idioms — an explicit score, a 1–5
    rating, a thumb, an accept/reject flag, or the behavioural signals of
    regenerating / editing the answer. Each maps to a number, but hard-coding
    the mapping inside the request schema hides it from operators and makes it
    impossible to tune without a code change. Centralising the mapping here
    keeps the routing loop explainable: the same normalised value flows into
    the statistics regardless of how the client expressed themselves.
    """

    # Direct, unambiguous signals.
    thumb_up_score: float = Field(default=1.0, ge=0.0, le=1.0)
    thumb_down_score: float = Field(default=0.0, ge=0.0, le=1.0)
    accept_score: float = Field(default=1.0, ge=0.0, le=1.0)
    reject_score: float = Field(default=0.0, ge=0.0, le=1.0)
    # Behavioural signals: regenerating the answer is a weak negative, editing
    # it before use is a mild negative (the answer helped but needed work).
    regenerated_score: float = Field(default=0.2, ge=0.0, le=1.0)
    edited_score: float = Field(default=0.5, ge=0.0, le=1.0)


class FeedbackConfig(BaseModel):
    """Online feedback loop that adapts routing to observed quality."""

    enabled: bool = True
    # Exponential moving average factor for quality/latency statistics.
    ema_alpha: float = Field(default=0.1, gt=0.0, le=1.0)
    # Minimum observations before online stats override the configured prior.
    min_samples: int = Field(default=20, ge=1)
    # How strongly learned quality can move the effective score, in [0, 1].
    learning_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    # Epsilon-greedy exploration so under-used models keep collecting evidence.
    exploration_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    # Sliding window (seconds) used for the online statistics.
    window_seconds: int = Field(default=3600, gt=0)
    # Below this success rate a model is treated as degraded and demoted.
    degraded_success_rate: float = Field(default=0.85, ge=0.0, le=1.0)
    # Implicit signals derived from the request lifecycle.
    treat_retry_as_negative: bool = True
    treat_truncation_as_negative: bool = True
    # Mapping that collapses the various feedback idioms into one score.
    normalization: FeedbackNormalizationConfig = Field(
        default_factory=FeedbackNormalizationConfig
    )


class ContextOverflowStrategy(str, Enum):
    """What to do when the prompt exceeds every candidate's context window."""

    # Fail fast with 400. Safest: the client learns the truth.
    REJECT = "reject"
    # Route to the largest-window model even if its tier is a poor match.
    LARGEST_WINDOW = "largest_window"
    # Drop middle history to make the prompt fit. Lossy, so it is opt-in.
    TRIM_HISTORY = "trim_history"


class ResponseCacheConfig(BaseModel):
    """Short-lived cache of exact-match completion results.

    A request is served from cache only when every field that can change the
    generated text is identical: the resolved target model, the full message
    list, and the sampling parameters. Streaming requests are never cached
    (the SSE stream is delivered live), and requests carrying a ``session_id``
    hint are excluded so that multi-turn threads — which depend on prior state
    the cache cannot see — never get a stale answer.
    """

    enabled: bool = False
    # How long a cached completion stays usable.
    ttl_seconds: int = Field(default=300, gt=0)
    # Names of chatrouter hints whose presence forces a cache bypass. A session
    # id means multi-turn state the cache cannot observe.
    bypass_hints: list[str] = Field(default_factory=lambda: ["session_id"])
    # Tenants listed here never read from or write to the cache. Useful for
    # tenants that must always reach the model (e.g. audit or evaluation work).
    excluded_tenants: list[str] = Field(default_factory=list)


class ContextOverflowConfig(BaseModel):
    """Handling for prompts that no candidate model can accommodate.

    Without this the gateway hard-fails, because the capability filter simply
    drops every model whose window is too small and routing then finds no
    candidate. In production a long conversation is a normal occurrence, not an
    error, so there needs to be a defined degradation path.
    """

    strategy: ContextOverflowStrategy = ContextOverflowStrategy.LARGEST_WINDOW
    # Fraction of the window the prompt may occupy after trimming, leaving
    # room for the completion.
    trim_target_ratio: float = Field(default=0.7, gt=0.0, le=1.0)
    # Leading messages that must survive trimming (system prompt, task setup).
    keep_leading_messages: int = Field(default=1, ge=0)
    # Trailing messages that must survive: the actual question plus context.
    keep_trailing_messages: int = Field(default=4, ge=1)
    # Insert a marker where content was removed so the model is not misled
    # into thinking the conversation was contiguous.
    insert_elision_notice: bool = True


class SessionAffinityConfig(BaseModel):
    """Keep a multi-turn conversation on one model to preserve prompt caches.

    Prefix caching (DeepSeek, Claude, Gemini, GPT-4o-class models) gives 75–90%
    cheaper input tokens when the same prefix repeats on the *same* model. Routing
    each turn to the cheapest-fit model shatters that prefix and forfeits the
    saving — frequently worse than the routing saved. Session affinity counters
    this by preferring the model a session already uses, unless the task's
    complexity has drifted far enough to justify a switch.

    ``stickiness`` is the cost penalty applied to *switching away* from the
    session's current model, expressed in utility units (0 disables affinity,
    1 strongly favours staying put). ``max_drift_tiers`` caps how far the chosen
    model may sit from the natural target tier before affinity is overridden —
    a session that suddenly needs reasoning will always be upgraded, never
    pinned to a too-small model.
    """

    enabled: bool = True
    stickiness: float = Field(default=0.4, ge=0.0, le=1.0)
    max_drift_tiers: int = Field(default=1, ge=0)
    ttl_seconds: int = Field(default=1800, gt=0)


class RoutingConfig(BaseModel):
    """Top-level routing policy."""

    default_model: str | None = Field(
        default=None, description="Fallback target when routing yields no candidate."
    )
    # 0 => pure cost optimisation, 1 => pure quality optimisation.
    quality_bias: float = Field(default=0.6, ge=0.0, le=1.0)
    latency_bias: float = Field(default=0.15, ge=0.0, le=1.0)
    # Allow serving from one tier above/below the target when capacity demands.
    allow_upgrade: bool = True
    allow_downgrade: bool = True
    # Honour an explicit model name from the client instead of routing.
    respect_explicit_model: bool = True
    # Virtual model names that always trigger routing, e.g. "auto".
    auto_model_aliases: list[str] = Field(default_factory=lambda: ["auto", "chatrouter-auto"])
    thresholds: TierThresholds = Field(default_factory=TierThresholds)
    context: ContextRoutingConfig = Field(default_factory=ContextRoutingConfig)
    context_overflow: ContextOverflowConfig = Field(default_factory=ContextOverflowConfig)
    response_cache: ResponseCacheConfig = Field(default_factory=ResponseCacheConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    session_affinity: SessionAffinityConfig = Field(default_factory=SessionAffinityConfig)


class CircuitBreakerConfig(BaseModel):
    """Per-model circuit breaker guarding against failing upstreams."""

    enabled: bool = True
    failure_threshold: int = Field(default=5, ge=1)
    failure_rate_threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    min_requests: int = Field(default=10, ge=1)
    open_seconds: float = Field(default=30.0, gt=0.0)
    half_open_max_calls: int = Field(default=3, ge=1)
    window_seconds: float = Field(default=60.0, gt=0.0)


class RetryConfig(BaseModel):
    """Retry and cross-model failover behaviour."""

    max_attempts: int = Field(default=3, ge=1)
    # Attempts beyond the first may hop to another model in the fallback chain.
    enable_model_failover: bool = True
    backoff_base_seconds: float = Field(default=0.25, ge=0.0)
    backoff_max_seconds: float = Field(default=4.0, ge=0.0)
    retry_on_status: list[int] = Field(default_factory=lambda: [408, 409, 425, 429, 500, 502, 503, 504])
    # A streamed response cannot be retried once bytes reached the client.
    retry_streaming_before_first_chunk: bool = True


class OverflowConfig(BaseModel):
    """Load overflow scheduling across the model pool."""

    enabled: bool = True
    # Utilisation above which a model is considered saturated.
    saturation_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    # Queue instead of failing when the whole tier is saturated.
    queue_enabled: bool = True
    queue_max_wait_seconds: float = Field(default=5.0, ge=0.0)
    queue_max_depth: int = Field(default=500, ge=0)
    # Permit spilling into a cheaper tier when everything else is full.
    allow_cross_tier_overflow: bool = True


class ResilienceConfig(BaseModel):
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    overflow: OverflowConfig = Field(default_factory=OverflowConfig)


class StorageConfig(BaseModel):
    """Backing store for counters, quotas and feedback statistics."""

    backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    key_prefix: str = "chatrouter"


class ObservabilityConfig(BaseModel):
    metrics_enabled: bool = True
    metrics_path: str = "/metrics"
    log_level: str = "INFO"
    log_json: bool = True
    # Never log message bodies unless explicitly enabled for debugging.
    log_request_bodies: bool = False


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, gt=0, lt=65536)
    workers: int = Field(default=1, ge=1)
    # Requires a valid tenant API key on every gateway call.
    require_auth: bool = True
    admin_api_key_env: str = "CHATROUTER_ADMIN_KEY"
    admin_api_key: str | None = None
    cors_origins: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    """Root configuration document."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    providers: list[ProviderConfig] = Field(default_factory=list)
    models: list[ModelConfig] = Field(default_factory=list)
    tenants: list[TenantConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_references(self) -> AppConfig:
        provider_names = {p.name for p in self.providers}
        if len(provider_names) != len(self.providers):
            raise ValueError("duplicate provider names in configuration")

        model_ids = {m.id for m in self.models}
        if len(model_ids) != len(self.models):
            raise ValueError("duplicate model ids in configuration")

        for model in self.models:
            if model.provider not in provider_names:
                raise ValueError(f"model '{model.id}' references unknown provider '{model.provider}'")

        tenant_ids = {t.id for t in self.tenants}
        if len(tenant_ids) != len(self.tenants):
            raise ValueError("duplicate tenant ids in configuration")

        for tenant in self.tenants:
            for ref in (*tenant.allowed_models, *tenant.denied_models):
                if ref not in model_ids:
                    raise ValueError(f"tenant '{tenant.id}' references unknown model '{ref}'")

        if self.routing.default_model and self.routing.default_model not in model_ids:
            raise ValueError(f"routing.default_model '{self.routing.default_model}' is not a known model")

        return self

    def provider_by_name(self, name: str) -> ProviderConfig | None:
        return next((p for p in self.providers if p.name == name), None)

    def model_by_id(self, model_id: str) -> ModelConfig | None:
        return next((m for m in self.models if m.id == model_id), None)
