"""The routing engine.

Decision flow for one request:

1. Honour an explicit pin / concrete model name if policy allows it.
2. Score the *whole conversation* to obtain a complexity tier.
3. Build the candidate set (tenant permissions, capability requirements,
   circuit-breaker health, capacity headroom).
4. Rank candidates by a utility function combining feedback-adjusted quality,
   cost, latency and current load.
5. Emit the winner plus an ordered fallback chain for the dispatcher.
"""

from __future__ import annotations

import random

from ..config.models import (
    AppConfig,
    ContextOverflowStrategy,
    ModelConfig,
    ModelTier,
    RoutingConfig,
    TenantConfig,
)
from ..core.errors import (
    ContextOverflowError,
    ModelNotFoundError,
    NoCandidateError,
    PermissionError_,
)
from ..core.schemas import ChatCompletionRequest
from ..core.types import (
    ComplexityAssessment,
    RequestContext,
    RoutingDecision,
    RoutingDecisionReason,
    ScoredCandidate,
    SessionAffinityBinding,
)
from ..governance.circuit_breaker import CircuitBreakerRegistry
from ..governance.load import LoadSnapshot, ModelLoadTracker
from ..storage.base import Storage
from .complexity import ComplexityAnalyzer
from .context_fit import largest_window_model
from .feedback import FeedbackStore

# Utility penalty applied per tier of distance from the target tier.
_TIER_DISTANCE_PENALTY = 0.18
# Cost normalisation ceiling (USD / 1k tokens) — above this, cost score is 0.
_COST_CEILING = 0.05
# Latency normalisation ceiling in milliseconds.
_LATENCY_CEILING = 15_000.0

# Convert a one-time prefix-cache-loss (USD) into router utility points. Tuned so
# a sizeable cache loss on a premium model is a strong (but bounded) deterrent to
# switching, while cheap models stay freely routable (SeqRoute, arXiv 2026).
_SWITCH_PENALTY_USD_PER_UTILITY = 1.0
_MAX_AFFINITY_PENALTY = 2.0


class Router:
    """Selects the model that serves a request."""

    def __init__(
        self,
        config: AppConfig,
        feedback: FeedbackStore,
        load_tracker: ModelLoadTracker,
        breakers: CircuitBreakerRegistry,
        storage: Storage | None = None,
    ) -> None:
        self._config = config
        self._routing: RoutingConfig = config.routing
        self._feedback = feedback
        self._load = load_tracker
        self._breakers = breakers
        self._storage = storage
        self._analyzer = ComplexityAnalyzer(config.routing.context, config.routing.thresholds)
        self._models = [m for m in config.models if m.enabled]

    # -- public API ---------------------------------------------------------

    @property
    def models(self) -> list[ModelConfig]:
        return self._models

    def analyse(self, request: ChatCompletionRequest) -> ComplexityAssessment:
        """Expose the complexity assessment (used by the explain endpoint)."""
        window_hint = max((m.context_window for m in self._models), default=128_000)
        return self._analyzer.analyse(request, window_hint)

    async def route(self, context: RequestContext, projected_tokens: int = 0) -> RoutingDecision:
        """Produce a routing decision for the given request."""
        request = context.request
        tenant = context.tenant
        hints = request.chatrouter

        # --- 1. Explicit targeting ----------------------------------------
        if hints and hints.pin_model:
            model = self._require_model(hints.pin_model)
            self._assert_tenant_allows(tenant, model)
            return RoutingDecision(
                model=model,
                reason=RoutingDecisionReason.PINNED,
                assessment=None,
                notes=[f"pinned to {model.id} by request hint"],
            )

        requested = request.model
        is_auto = requested in self._routing.auto_model_aliases
        if not is_auto and self._routing.respect_explicit_model:
            model = self._config.model_by_id(requested)
            if model is not None and model.enabled:
                self._assert_tenant_allows(tenant, model)
                fallbacks = await self._build_fallbacks(model, tenant, projected_tokens, exclude={model.id})
                return RoutingDecision(
                    model=model,
                    reason=RoutingDecisionReason.EXPLICIT_MODEL,
                    assessment=None,
                    fallback_chain=fallbacks,
                    notes=[f"client requested {model.id} explicitly"],
                )
            if model is None and not is_auto:
                # Unknown name: fall through to routing rather than 404, but
                # only when the alias looks like a routing request.
                if not self._looks_like_virtual(requested):
                    raise ModelNotFoundError(f"model '{requested}' does not exist")

        # --- 2. Complexity assessment --------------------------------------
        assessment = self.analyse(request)
        target_tier = assessment.tier
        notes: list[str] = list(assessment.explanation)

        # --- 3. Candidate construction --------------------------------------
        allowed = self._allowed_models(tenant, hints)
        if not allowed:
            raise NoCandidateError("no model is available for this tenant")

        capable = [m for m in allowed if self._is_capable(m, assessment, projected_tokens)]
        if not capable:
            capable = self._handle_no_capable(allowed, assessment, projected_tokens, notes)

        # Apply the tenant ceiling *after* scoring the task honestly, so the
        # decision log shows what the task needed versus what it was allowed.
        ceiling = self._effective_ceiling(tenant, hints)
        if ceiling is not None and target_tier.rank > ceiling.rank:
            notes.append(
                f"task needs {target_tier.value} but tenant ceiling is {ceiling.value}"
            )
            target_tier = ceiling

        if context.quota_downgraded:
            cheapest = self._cheapest_available_tier(capable)
            if cheapest is not None and cheapest.rank < target_tier.rank:
                notes.append("quota exhausted → downgraded tier")
                target_tier = cheapest

        min_tier = self._parse_tier(hints.min_tier if hints else None)
        if min_tier and target_tier.rank < min_tier.rank:
            target_tier = min_tier
            notes.append(f"request hint raised tier to {min_tier.value}")

        loads = await self._load.snapshot_many(capable)

        affinity_cfg = self._routing.session_affinity
        affinity_model_id = None
        affinity_prefix_tokens = 0
        if affinity_cfg.enabled and context.session_id and self._storage is not None:
            binding = await self._storage.get_session_affinity(context.session_id)
            if binding is not None:
                affinity_model_id = binding.model_id
                affinity_prefix_tokens = binding.prefix_tokens

        # --- 4. Scoring -------------------------------------------------------
        scored = await self._score_candidates(
            capable,
            target_tier,
            tenant,
            hints,
            loads,
            projected_tokens,
            affinity_model_id=affinity_model_id,
            affinity_prefix_tokens=affinity_prefix_tokens,
            stickiness=affinity_cfg.stickiness if affinity_cfg.enabled else 0.0,
        )
        if not scored:
            raise NoCandidateError("all candidate models are unavailable")

        healthy = [c for c in scored if self._breakers.allows(c.model.id)]
        pool = healthy or scored
        if not healthy:
            notes.append("all candidates are circuit-open; attempting best-effort")

        with_capacity = [
            c for c in pool if not self._load.is_saturated(loads[c.model.id], projected_tokens)
        ]
        reason = RoutingDecisionReason.CONTEXT_AWARE
        if with_capacity:
            pool = with_capacity
        elif self._config.resilience.overflow.enabled:
            notes.append("preferred candidates saturated → overflow scheduling")
            reason = RoutingDecisionReason.OVERFLOW

        pool.sort(key=lambda c: c.utility, reverse=True)
        winner = pool[0]
        exploration = False

        # --- 4b. Session affinity override -----------------------------------
        # If this session already routes to a model and the task has not drifted
        # far from that model's tier, keep it there to preserve the upstream
        # prefix cache (a 75–90% input-cost saving that routing would otherwise
        # destroy). Affinity never overrides a hard capability/ceiling mismatch.
        if (
            affinity_model_id
            and reason in (RoutingDecisionReason.CONTEXT_AWARE, RoutingDecisionReason.OVERFLOW)
        ):
            sticky = next((c for c in pool if c.model.id == affinity_model_id), None)
            if sticky is not None:
                drift = abs(sticky.model.tier.rank - target_tier.rank)
                if drift <= affinity_cfg.max_drift_tiers:
                    winner = sticky
                    reason = RoutingDecisionReason.SESSION_AFFINITY
                    notes.append(
                        f"session affinity: staying on {winner.model.id} "
                        f"(tier drift {drift} <= {affinity_cfg.max_drift_tiers})"
                    )

        # Quantify, for observability, the real business cost of breaking this
        # session's prefix cache by switching models (SeqRoute switch penalty).
        if affinity_model_id and affinity_prefix_tokens > 0:
            aff = self._config.model_by_id(affinity_model_id)
            if aff is not None:
                loss_usd = (affinity_prefix_tokens / 1000.0) * max(
                    0.0, aff.input_cost_per_1k - aff.cached_input_cost_eff
                )
                notes.append(
                    f"session affinity: switching forfeits ${loss_usd:.6f} of prefix "
                    f"cache over {affinity_prefix_tokens} historical tokens"
                )

        # --- 5. Exploration ---------------------------------------------------
        # Occasionally try a runner-up so under-served models keep producing
        # evidence; without this the feedback loop can lock onto a local optimum.
        # Skipped when affinity already pinned the session to a sensible model.
        explore_ratio = self._routing.feedback.exploration_ratio
        if (
            reason is not RoutingDecisionReason.SESSION_AFFINITY
            and self._routing.feedback.enabled
            and explore_ratio > 0
            and len(pool) > 1
            and random.random() < explore_ratio
        ):
            alternatives = [c for c in pool[1:] if c.model.tier.rank >= target_tier.rank - 1]
            if alternatives:
                winner = random.choice(alternatives[: max(2, len(alternatives) // 2)])
                exploration = True
                reason = RoutingDecisionReason.EXPLORATION
                notes.append(f"exploration: sampling {winner.model.id}")

        if not exploration and self._routing.feedback.enabled and winner.quality != winner.model.quality_prior:
            reason = (
                RoutingDecisionReason.FEEDBACK_ADAPTIVE
                if reason is RoutingDecisionReason.CONTEXT_AWARE
                else reason
            )

        if context.quota_downgraded:
            reason = RoutingDecisionReason.QUOTA_DOWNGRADE

        # --- 4c. Persist affinity --------------------------------------------
        # Remember the model we chose so the next turn of this session reuses it
        # (and thus the prefix cache). The historical prefix size is preserved
        # here; the completion path (_update_session_affinity) then adds the
        # just-finished turn's prompt tokens so the router can later price the
        # switch penalty (prefix_tokens * (c_in - c_cache)). Pinned/explicit
        # models are stable by definition and are skipped.
        if (
            affinity_cfg.enabled
            and context.session_id
            and self._storage is not None
            and reason
            not in (
                RoutingDecisionReason.PINNED,
                RoutingDecisionReason.EXPLICIT_MODEL,
            )
        ):
            existing_prefix = binding.prefix_tokens if binding is not None else 0
            await self._storage.set_session_affinity(
                context.session_id, winner.model.id, existing_prefix, affinity_cfg.ttl_seconds
            )

        fallbacks = await self._build_fallbacks(
            winner.model, tenant, projected_tokens, exclude={winner.model.id}, hints=hints
        )

        return RoutingDecision(
            model=winner.model,
            reason=reason,
            assessment=assessment,
            fallback_chain=fallbacks,
            candidates=pool[:8],
            exploration=exploration,
            notes=notes,
        )

    # -- candidate helpers ---------------------------------------------------

    def _looks_like_virtual(self, name: str) -> bool:
        """Treat unknown ``auto``-ish names as routing requests."""
        lowered = name.lower()
        return any(alias in lowered for alias in ("auto", "router", "chatrouter"))

    def _require_model(self, model_id: str) -> ModelConfig:
        model = self._config.model_by_id(model_id)
        if model is None or not model.enabled:
            raise ModelNotFoundError(f"model '{model_id}' does not exist")
        return model

    @staticmethod
    def _assert_tenant_allows(tenant: TenantConfig, model: ModelConfig) -> None:
        if model.id in tenant.denied_models:
            raise PermissionError_(f"tenant '{tenant.id}' may not use model '{model.id}'")
        if tenant.allowed_models and model.id not in tenant.allowed_models:
            raise PermissionError_(f"tenant '{tenant.id}' may not use model '{model.id}'")
        if tenant.max_tier and model.tier.rank > tenant.max_tier.rank:
            raise PermissionError_(
                f"tenant '{tenant.id}' is limited to the '{tenant.max_tier.value}' tier"
            )

    def _allowed_models(self, tenant: TenantConfig, hints) -> list[ModelConfig]:
        """Models this tenant and request may use."""
        excluded = set(hints.exclude_models) if hints else set()
        result: list[ModelConfig] = []
        for model in self._models:
            if model.id in excluded or model.id in tenant.denied_models:
                continue
            if tenant.allowed_models and model.id not in tenant.allowed_models:
                continue
            if tenant.max_tier and model.tier.rank > tenant.max_tier.rank:
                continue
            result.append(model)
        return result

    @staticmethod
    def _is_capable(
        model: ModelConfig, assessment: ComplexityAssessment, projected_tokens: int
    ) -> bool:
        """Hard capability filter — a model that cannot serve must be dropped."""
        if assessment.requires_vision and not model.supports_vision:
            return False
        if assessment.requires_tools and not model.supports_tools:
            return False
        if assessment.prompt_tokens_estimate > model.context_window:
            return False
        if projected_tokens and projected_tokens > model.context_window:
            return False
        return True

    def _handle_no_capable(
        self,
        allowed: list[ModelConfig],
        assessment: ComplexityAssessment,
        projected_tokens: int,
        notes: list[str],
    ) -> list[ModelConfig]:
        """Recover when the capability filter eliminated every model.

        Context overflow is treated separately from other capability misses:
        relaxing a vision/tools requirement only degrades quality, but ignoring
        the context window guarantees an upstream 400. So when overflow is the
        cause we narrow to the models that can actually hold the prompt, and
        only fall back to "relax everything" when it is not.
        """
        needed = max(assessment.prompt_tokens_estimate, projected_tokens)
        overflowed = [m for m in allowed if needed > m.context_window]

        if overflowed and len(overflowed) == len(allowed):
            strategy = self._routing.context_overflow.strategy
            if strategy is ContextOverflowStrategy.REJECT:
                largest = largest_window_model(allowed)
                limit = largest.context_window if largest else 0
                raise ContextOverflowError(
                    f"conversation requires ~{needed} tokens but the largest "
                    f"available model holds {limit}"
                )
            # For both largest_window and trim_history the routing target is
            # the same: the widest window available. Trimming (if enabled) is
            # applied by the service once the model is known.
            largest = largest_window_model(allowed)
            if largest is not None:
                notes.append(
                    f"context overflow (~{needed} tokens) → routing to "
                    f"widest-window model {largest.id} ({largest.context_window})"
                )
                return [largest]

        notes.append("no model satisfied capability requirements; relaxing constraints")
        return allowed

    def _effective_ceiling(self, tenant: TenantConfig, hints) -> ModelTier | None:
        ceilings = [t for t in (tenant.max_tier, self._parse_tier(hints.max_tier if hints else None)) if t]
        if not ceilings:
            return None
        return min(ceilings, key=lambda t: t.rank)

    @staticmethod
    def _parse_tier(value: str | None) -> ModelTier | None:
        if not value:
            return None
        try:
            return ModelTier(value)
        except ValueError:
            return None

    @staticmethod
    def _cheapest_available_tier(models: list[ModelConfig]) -> ModelTier | None:
        if not models:
            return None
        return min((m.tier for m in models), key=lambda t: t.rank)

    # -- scoring -------------------------------------------------------------

    async def _score_candidates(
        self,
        models: list[ModelConfig],
        target_tier: ModelTier,
        tenant: TenantConfig,
        hints,
        loads: dict[str, LoadSnapshot],
        projected_tokens: int,
        affinity_model_id: str | None = None,
        affinity_prefix_tokens: int = 0,
        stickiness: float = 0.0,
    ) -> list[ScoredCandidate]:
        """Rank models by utility for the target tier.

        Session affinity is priced by the real cost of breaking the session's
        upstream prefix cache (SeqRoute, arXiv 2026): switching away from the
        session's current model forfeits ``affinity_prefix_tokens * (c_in - c_cache)``
        USD of cache, so every non-sticky model pays that penalty (scaled by
        ``stickiness``; 0 disables affinity).
        """
        quality_bias = self._resolve_quality_bias(tenant, hints)
        latency_bias = self._routing.latency_bias
        cost_bias = max(0.0, 1.0 - quality_bias - latency_bias)
        preferred = set(hints.prefer_models) if hints else set()

        # Real, business-cost switch penalty (SeqRoute, arXiv 2026):
        #   penalty_usd = historical_prefix_tokens * (c_in - c_cache)
        # i.e. the one-time cache loss from abandoning this session's upstream
        # prefix cache. We convert the USD loss into the router's utility scale
        # with a single documented constant and let ``stickiness`` scale it.
        affinity_penalty = 0.0
        if affinity_model_id and affinity_prefix_tokens > 0:
            aff = self._config.model_by_id(affinity_model_id)
            if aff is not None:
                cache_loss_per_1k = max(0.0, aff.input_cost_per_1k - aff.cached_input_cost_eff)
                switch_loss_usd = (affinity_prefix_tokens / 1000.0) * cache_loss_per_1k
                affinity_penalty = min(
                    switch_loss_usd / _SWITCH_PENALTY_USD_PER_UTILITY,
                    _MAX_AFFINITY_PENALTY,
                ) * stickiness

        candidates: list[ScoredCandidate] = []
        for model in models:
            distance = model.tier.rank - target_tier.rank
            # Tiers below the target risk quality; tiers above waste money.
            if distance < 0 and not self._routing.allow_downgrade:
                continue
            if distance > 0 and not self._routing.allow_upgrade:
                continue

            stats = await self._feedback.get_stats(model.id, target_tier)
            quality = self._feedback.effective_quality(model.quality_prior, stats)

            cost_score = 1.0 - min(1.0, model.avg_cost_per_1k / _COST_CEILING)
            latency_ms = stats.latency_ema_ms or model.latency_prior_ms
            latency_score = 1.0 - min(1.0, latency_ms / _LATENCY_CEILING)

            snapshot = loads.get(model.id)
            load_score = 1.0 - min(1.0, snapshot.utilisation) if snapshot else 1.0

            # Under-shooting the required tier is penalised harder than
            # over-shooting: quality regressions hurt more than a little cost.
            penalty = abs(distance) * _TIER_DISTANCE_PENALTY
            if distance < 0:
                penalty *= 1.6

            health_penalty = self._breakers.health_penalty(model.id)

            utility = (
                quality_bias * quality
                + cost_bias * cost_score
                + latency_bias * latency_score
                + 0.12 * load_score
                - penalty
                - 0.5 * health_penalty
            )
            utility += 0.05 * (model.weight - 1.0)
            if model.id in preferred:
                utility += 0.25
            if affinity_model_id and model.id != affinity_model_id:
                utility -= affinity_penalty

            # Nudge exploration towards models with little evidence so the
            # feedback loop can actually learn about them.
            exploration_bonus = 0.0
            if self._routing.feedback.enabled and stats.requests < self._routing.feedback.min_samples:
                exploration_bonus = 0.03 * (
                    1 - stats.requests / max(self._routing.feedback.min_samples, 1)
                )
                utility += exploration_bonus

            candidates.append(
                ScoredCandidate(
                    model=model,
                    utility=utility,
                    quality=quality,
                    cost_score=cost_score,
                    latency_score=latency_score,
                    load_score=load_score,
                    tier_penalty=penalty,
                    exploration_bonus=exploration_bonus,
                )
            )

        candidates.sort(key=lambda c: c.utility, reverse=True)
        return candidates

    def _resolve_quality_bias(self, tenant: TenantConfig, hints) -> float:
        if hints and hints.quality_bias is not None:
            return hints.quality_bias
        if tenant.quality_bias is not None:
            return tenant.quality_bias
        return self._routing.quality_bias

    # -- fallback chain ------------------------------------------------------

    async def _build_fallbacks(
        self,
        primary: ModelConfig,
        tenant: TenantConfig,
        projected_tokens: int,
        exclude: set[str],
        hints=None,
    ) -> list[ModelConfig]:
        """Order alternatives to try when the primary model fails.

        Preference order: same tier (equivalent quality), then one tier up
        (safe for quality), then cheaper tiers as a last resort so the request
        degrades rather than fails.
        """
        allowed = [m for m in self._allowed_models(tenant, hints) if m.id not in exclude]
        if not allowed:
            return []

        def sort_key(model: ModelConfig) -> tuple[int, float]:
            distance = model.tier.rank - primary.tier.rank
            if distance == 0:
                group = 0
            elif distance > 0:
                group = 1
            else:
                group = 2
            health = self._breakers.health_penalty(model.id)
            return (group, health - model.quality_prior)

        candidates = sorted(allowed, key=sort_key)
        chain: list[ModelConfig] = []
        for model in candidates:
            if self._breakers.allows(model.id):
                chain.append(model)
            if len(chain) >= max(1, self._config.resilience.retry.max_attempts):
                break

        default_id = self._routing.default_model
        if default_id and default_id not in exclude:
            default_model = self._config.model_by_id(default_id)
            if default_model and default_model.enabled and all(m.id != default_id for m in chain):
                chain.append(default_model)
        return chain

    def find_overflow_target(
        self,
        exclude: set[str],
        tenant: TenantConfig,
        target_tier: ModelTier,
        loads: dict[str, LoadSnapshot],
        projected_tokens: int,
    ) -> ModelConfig | None:
        """Pick a model with spare capacity when the preferred pool is full."""
        overflow_cfg = self._config.resilience.overflow
        allowed = [m for m in self._allowed_models(tenant, None) if m.id not in exclude]
        viable = [
            m
            for m in allowed
            if self._breakers.allows(m.id)
            and m.id in loads
            and loads[m.id].has_headroom(projected_tokens)
        ]
        if not viable:
            return None
        if not overflow_cfg.allow_cross_tier_overflow:
            viable = [m for m in viable if m.tier.rank >= target_tier.rank]
            if not viable:
                return None
        # Closest tier first, then the least loaded instance.
        viable.sort(key=lambda m: (abs(m.tier.rank - target_tier.rank), loads[m.id].utilisation))
        return viable[0]
