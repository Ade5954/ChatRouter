"""Online feedback loop.

The router does not stay frozen at its configured priors. Every served request
contributes implicit evidence (success, latency, truncation, retries) and every
explicit rating contributes direct evidence. Both are folded into exponential
moving averages per ``(model, tier-band)`` so the effective quality used for
scoring tracks what production actually observes.

Statistics are segmented by complexity band: a model may be excellent on easy
traffic and weak on hard reasoning, and a single global average would hide it.
"""

from __future__ import annotations

import time
from typing import Any

from ..config.models import FeedbackConfig, ModelTier
from ..core.types import ModelRuntimeStats
from ..storage.base import Storage

_STATS_PREFIX = "stats:model"
_REQUEST_PREFIX = "req"
# Correlation records must outlive the user's chance to click 👍/👎.
_REQUEST_TTL_SECONDS = 24 * 3600


def _band(tier: ModelTier | None) -> str:
    """Complexity band a sample belongs to; ``all`` is the global aggregate."""
    return tier.value if tier else "all"


class FeedbackStore:
    """Reads and writes the adaptive routing statistics."""

    def __init__(self, storage: Storage, config: FeedbackConfig) -> None:
        self._storage = storage
        self._config = config

    # -- keys -------------------------------------------------------------

    @staticmethod
    def _stats_key(model_id: str, band: str) -> str:
        return f"{_STATS_PREFIX}:{model_id}:{band}"

    @staticmethod
    def _request_key(request_id: str) -> str:
        return f"{_REQUEST_PREFIX}:{request_id}"

    # -- request correlation ----------------------------------------------

    async def remember_request(
        self,
        request_id: str,
        model_id: str,
        tenant_id: str,
        tier: ModelTier | None,
        complexity_score: float | None,
    ) -> None:
        """Persist enough context to attribute later feedback to a decision."""
        await self._storage.put_record(
            self._request_key(request_id),
            {
                "model_id": model_id,
                "tenant_id": tenant_id,
                "band": _band(tier),
                "complexity_score": complexity_score,
                "created_at": time.time(),
            },
            _REQUEST_TTL_SECONDS,
        )

    async def lookup_request(self, request_id: str) -> dict[str, Any] | None:
        return await self._storage.take_record(self._request_key(request_id))

    # -- statistics updates -------------------------------------------------

    async def record_outcome(
        self,
        model_id: str,
        tier: ModelTier | None,
        *,
        success: bool,
        latency_ms: float | None = None,
        implicit_score: float | None = None,
    ) -> None:
        """Fold one served request into the statistics.

        Updates both the per-band record and the global aggregate so the router
        can fall back to the aggregate while a band is still sparse.
        """
        for band in {_band(tier), "all"}:
            await self._storage.update_stats(
                self._stats_key(model_id, band),
                lambda current: self._apply_outcome(
                    current, success=success, latency_ms=latency_ms, implicit_score=implicit_score
                ),
                ttl_seconds=self._config.window_seconds * 24,
            )

    async def record_feedback(
        self, model_id: str, band: str, score: float
    ) -> None:
        """Fold an explicit user rating into the statistics."""
        for target_band in {band, "all"}:
            await self._storage.update_stats(
                self._stats_key(model_id, target_band),
                lambda current: self._apply_feedback(current, score),
                ttl_seconds=self._config.window_seconds * 24,
            )

    def _apply_outcome(
        self,
        current: dict[str, Any],
        *,
        success: bool,
        latency_ms: float | None,
        implicit_score: float | None,
    ) -> dict[str, Any]:
        alpha = self._config.ema_alpha
        record = self._decay_if_stale(current)

        record["requests"] = int(record.get("requests", 0)) + 1
        if success:
            record["successes"] = int(record.get("successes", 0)) + 1
        else:
            record["failures"] = int(record.get("failures", 0)) + 1

        if latency_ms is not None:
            previous = record.get("latency_ema_ms")
            record["latency_ema_ms"] = (
                latency_ms if previous is None else previous * (1 - alpha) + latency_ms * alpha
            )

        if implicit_score is not None:
            # Implicit evidence is weaker than an explicit rating, so it moves
            # the average at a reduced rate.
            weak_alpha = alpha * 0.5
            previous = record.get("quality_ema")
            record["quality_ema"] = (
                implicit_score
                if previous is None
                else previous * (1 - weak_alpha) + implicit_score * weak_alpha
            )
            record["quality_samples"] = float(record.get("quality_samples", 0.0)) + 0.5

        record["updated_at"] = time.time()
        return record

    def _apply_feedback(self, current: dict[str, Any], score: float) -> dict[str, Any]:
        alpha = self._config.ema_alpha
        record = self._decay_if_stale(current)
        previous = record.get("quality_ema")
        record["quality_ema"] = score if previous is None else previous * (1 - alpha) + score * alpha
        record["quality_samples"] = float(record.get("quality_samples", 0.0)) + 1.0
        record["feedback_count"] = int(record.get("feedback_count", 0)) + 1
        record["updated_at"] = time.time()
        return record

    def _decay_if_stale(self, record: dict[str, Any]) -> dict[str, Any]:
        """Fade out counters older than the configured window.

        Without this, a model that failed badly last week would stay penalised
        forever; with it, evidence has a bounded half-life.
        """
        record = dict(record)
        updated_at = float(record.get("updated_at", 0.0))
        if not updated_at:
            return record
        age = time.time() - updated_at
        window = self._config.window_seconds
        if age <= window:
            return record
        # One full window of silence halves the accumulated evidence.
        factor = 0.5 ** min(age / window, 8)
        for key in ("requests", "successes", "failures", "feedback_count"):
            if key in record:
                record[key] = int(float(record[key]) * factor)
        if "quality_samples" in record:
            record["quality_samples"] = float(record["quality_samples"]) * factor
        return record

    # -- reads ---------------------------------------------------------------

    async def get_stats(self, model_id: str, tier: ModelTier | None) -> ModelRuntimeStats:
        """Effective statistics for a model in a complexity band.

        Falls back to the global aggregate while the band has too few samples.
        """
        band = _band(tier)
        record = await self._storage.get_stats(self._stats_key(model_id, band)) or {}
        if int(record.get("requests", 0)) < self._config.min_samples and band != "all":
            aggregate = await self._storage.get_stats(self._stats_key(model_id, "all")) or {}
            if int(aggregate.get("requests", 0)) > int(record.get("requests", 0)):
                record = aggregate
        return self._to_stats(model_id, self._decay_if_stale(record))

    async def get_all_stats(self) -> dict[str, ModelRuntimeStats]:
        """Every model's global aggregate, for the admin endpoint."""
        raw = await self._storage.get_all_stats(_STATS_PREFIX)
        result: dict[str, ModelRuntimeStats] = {}
        for suffix, record in raw.items():
            parts = suffix.split(":")
            if len(parts) != 2 or parts[1] != "all":
                continue
            result[parts[0]] = self._to_stats(parts[0], record)
        return result

    @staticmethod
    def _to_stats(model_id: str, record: dict[str, Any]) -> ModelRuntimeStats:
        return ModelRuntimeStats(
            model_id=model_id,
            requests=int(record.get("requests", 0)),
            successes=int(record.get("successes", 0)),
            failures=int(record.get("failures", 0)),
            quality_ema=record.get("quality_ema"),
            latency_ema_ms=record.get("latency_ema_ms"),
            feedback_count=int(record.get("feedback_count", 0)),
            updated_at=float(record.get("updated_at", time.time())),
        )

    def effective_quality(self, prior: float, stats: ModelRuntimeStats) -> float:
        """Blend the configured prior with observed quality.

        Confidence grows with the number of samples, so a model with two
        ratings barely moves while one with hundreds dominates its prior.
        """
        cfg = self._config
        if not cfg.enabled or stats.quality_ema is None:
            return prior
        samples = max(stats.feedback_count, 0) + stats.requests * 0.25
        confidence = min(1.0, samples / max(cfg.min_samples, 1))
        blend = cfg.learning_rate * confidence
        quality = prior * (1 - blend) + stats.quality_ema * blend

        # A model failing outright is unfit regardless of its ratings.
        if stats.requests >= cfg.min_samples and stats.success_rate < cfg.degraded_success_rate:
            penalty = (cfg.degraded_success_rate - stats.success_rate) / max(
                cfg.degraded_success_rate, 1e-6
            )
            quality *= max(0.0, 1.0 - penalty)
        return max(0.0, min(1.0, quality))

    def implicit_score(
        self,
        *,
        success: bool,
        attempts: int,
        truncated: bool,
        latency_ms: float | None,
        latency_prior_ms: float,
    ) -> float | None:
        """Derive a weak quality signal from the request lifecycle alone."""
        cfg = self._config
        if not cfg.enabled:
            return None
        if not success:
            return 0.0

        score = 0.8
        if cfg.treat_retry_as_negative and attempts > 1:
            score -= 0.15 * min(attempts - 1, 3)
        if cfg.treat_truncation_as_negative and truncated:
            score -= 0.2
        if latency_ms is not None and latency_prior_ms > 0:
            # Being much slower than expected degrades perceived quality.
            ratio = latency_ms / latency_prior_ms
            if ratio > 2.0:
                score -= min(0.2, 0.05 * (ratio - 2.0))
        return max(0.0, min(1.0, score))
