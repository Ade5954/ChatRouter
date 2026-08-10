"""Model load tracking and overflow scheduling.

Every routable model has an independent capacity envelope (RPM, TPM, and
concurrency). The load tracker turns those limits into a normalised utilisation
figure that the router uses to (a) rank candidates and (b) decide when traffic
must overflow onto another model.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from ..config.models import ModelConfig, OverflowConfig
from ..storage.base import Storage


@dataclass(slots=True)
class LoadSnapshot:
    """Current utilisation of a single model."""

    model_id: str
    rpm_used: int = 0
    rpm_limit: int | None = None
    tpm_used: int = 0
    tpm_limit: int | None = None
    inflight: int = 0
    concurrency_limit: int | None = None

    @property
    def utilisation(self) -> float:
        """Highest utilisation across all configured dimensions, in [0, 1+]."""
        ratios: list[float] = []
        if self.rpm_limit:
            ratios.append(self.rpm_used / self.rpm_limit)
        if self.tpm_limit:
            ratios.append(self.tpm_used / self.tpm_limit)
        if self.concurrency_limit:
            ratios.append(self.inflight / self.concurrency_limit)
        return max(ratios) if ratios else 0.0

    def has_headroom(self, projected_tokens: int = 0) -> bool:
        """Whether one more request of this size fits inside the envelope."""
        if self.rpm_limit and self.rpm_used + 1 > self.rpm_limit:
            return False
        if self.tpm_limit and self.tpm_used + projected_tokens > self.tpm_limit:
            return False
        if self.concurrency_limit and self.inflight >= self.concurrency_limit:
            return False
        return True

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model_id,
            "rpm_used": self.rpm_used,
            "rpm_limit": self.rpm_limit,
            "tpm_used": self.tpm_used,
            "tpm_limit": self.tpm_limit,
            "inflight": self.inflight,
            "concurrency_limit": self.concurrency_limit,
            "utilisation": round(self.utilisation, 4),
        }


class ModelLoadTracker:
    """Tracks per-model consumption against the configured capacity."""

    def __init__(self, storage: Storage, overflow: OverflowConfig) -> None:
        self._storage = storage
        self._overflow = overflow
        # Local mirror of in-flight counts, used to release reliably.
        self._local_inflight: dict[str, int] = {}

    @staticmethod
    def _rpm_key(model_id: str) -> str:
        return f"model:rpm:{model_id}"

    @staticmethod
    def _tpm_key(model_id: str) -> str:
        return f"model:tpm:{model_id}"

    @staticmethod
    def _inflight_key(model_id: str) -> str:
        return f"model:inflight:{model_id}"

    async def snapshot(self, model: ModelConfig) -> LoadSnapshot:
        """Read the current utilisation of one model."""
        rpm_used = await self._storage.get_window(self._rpm_key(model.id), 60) if model.max_rpm else 0
        tpm_used = await self._storage.get_window(self._tpm_key(model.id), 60) if model.max_tpm else 0
        inflight = await self._storage.get_gauge(self._inflight_key(model.id))
        return LoadSnapshot(
            model_id=model.id,
            rpm_used=rpm_used,
            rpm_limit=model.max_rpm,
            tpm_used=tpm_used,
            tpm_limit=model.max_tpm,
            inflight=inflight,
            concurrency_limit=model.max_concurrency,
        )

    async def snapshot_many(self, models: list[ModelConfig]) -> dict[str, LoadSnapshot]:
        """Read utilisation for many models concurrently."""
        results = await asyncio.gather(*(self.snapshot(m) for m in models))
        return {snap.model_id: snap for snap in results}

    def is_saturated(self, snapshot: LoadSnapshot, projected_tokens: int = 0) -> bool:
        """Whether the model should be treated as full for scheduling."""
        if not snapshot.has_headroom(projected_tokens):
            return True
        if not self._overflow.enabled:
            return False
        return snapshot.utilisation >= self._overflow.saturation_threshold

    async def reserve(self, model: ModelConfig, projected_tokens: int) -> None:
        """Account for a request that is about to be dispatched."""
        if model.max_rpm:
            await self._storage.incr_window(self._rpm_key(model.id), 1, 60)
        if model.max_tpm and projected_tokens:
            await self._storage.incr_window(self._tpm_key(model.id), projected_tokens, 60)
        await self._storage.incr_gauge(self._inflight_key(model.id), 1)
        self._local_inflight[model.id] = self._local_inflight.get(model.id, 0) + 1

    async def release(self, model: ModelConfig, actual_tokens: int, projected_tokens: int) -> None:
        """Release concurrency and reconcile the token estimate with reality."""
        await self._storage.incr_gauge(self._inflight_key(model.id), -1)
        self._local_inflight[model.id] = max(0, self._local_inflight.get(model.id, 1) - 1)
        if model.max_tpm:
            delta = actual_tokens - projected_tokens
            if delta:
                # Correct the reservation so the window reflects real usage.
                await self._storage.incr_window(self._tpm_key(model.id), delta, 60)

    async def wait_for_capacity(
        self, model: ModelConfig, projected_tokens: int, deadline_seconds: float
    ) -> bool:
        """Briefly queue for capacity instead of failing immediately.

        Returns ``True`` if capacity became available within the deadline.
        """
        if not self._overflow.queue_enabled or deadline_seconds <= 0:
            return False
        deadline = time.monotonic() + deadline_seconds
        delay = 0.05
        while time.monotonic() < deadline:
            snapshot = await self.snapshot(model)
            if snapshot.has_headroom(projected_tokens):
                return True
            await asyncio.sleep(min(delay, max(0.0, deadline - time.monotonic())))
            delay = min(delay * 1.6, 0.5)
        return False
