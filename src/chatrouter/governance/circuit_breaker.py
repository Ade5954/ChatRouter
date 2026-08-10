"""Per-model circuit breakers.

A breaker isolates a failing upstream so the gateway stops spending latency
budget on calls that are almost certain to fail, and gives the provider room to
recover. State is intentionally process-local: reacting to what *this* replica
observes is both faster and sufficient, since every replica sees the same
outage.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from ..config.models import CircuitBreakerConfig


class BreakerState(str, Enum):
    CLOSED = "closed"      # normal operation
    OPEN = "open"          # failing; calls are rejected outright
    HALF_OPEN = "half_open"  # probing whether the upstream recovered


@dataclass
class _BreakerRecord:
    state: BreakerState = BreakerState.CLOSED
    opened_at: float = 0.0
    half_open_calls: int = 0
    half_open_successes: int = 0
    consecutive_failures: int = 0
    # (timestamp, was_failure) samples inside the rolling window.
    events: deque[tuple[float, bool]] = field(default_factory=deque)


class CircuitBreakerRegistry:
    """Holds one breaker per model id."""

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._breakers: dict[str, _BreakerRecord] = {}

    def _record(self, model_id: str) -> _BreakerRecord:
        record = self._breakers.get(model_id)
        if record is None:
            record = _BreakerRecord()
            self._breakers[model_id] = record
        return record

    def _prune(self, record: _BreakerRecord, now: float) -> None:
        cutoff = now - self._config.window_seconds
        while record.events and record.events[0][0] < cutoff:
            record.events.popleft()

    def allows(self, model_id: str) -> bool:
        """Whether a call to this model may be attempted right now."""
        if not self._config.enabled:
            return True
        record = self._record(model_id)
        now = time.monotonic()

        if record.state is BreakerState.OPEN:
            if now - record.opened_at >= self._config.open_seconds:
                # Cool-down elapsed: allow a limited number of probe calls.
                record.state = BreakerState.HALF_OPEN
                record.half_open_calls = 0
                record.half_open_successes = 0
            else:
                return False

        if record.state is BreakerState.HALF_OPEN:
            if record.half_open_calls >= self._config.half_open_max_calls:
                return False
            record.half_open_calls += 1

        return True

    def record_success(self, model_id: str) -> None:
        if not self._config.enabled:
            return
        record = self._record(model_id)
        now = time.monotonic()
        record.events.append((now, False))
        self._prune(record, now)
        record.consecutive_failures = 0

        if record.state is BreakerState.HALF_OPEN:
            record.half_open_successes += 1
            if record.half_open_successes >= self._config.half_open_max_calls:
                # The upstream proved itself; resume normal operation.
                record.state = BreakerState.CLOSED
                record.events.clear()

    def record_failure(self, model_id: str) -> None:
        if not self._config.enabled:
            return
        record = self._record(model_id)
        now = time.monotonic()
        record.events.append((now, True))
        self._prune(record, now)
        record.consecutive_failures += 1

        if record.state is BreakerState.HALF_OPEN:
            # A probe failed: go straight back to open with a fresh cool-down.
            self._open(record, now)
            return

        cfg = self._config
        if record.consecutive_failures >= cfg.failure_threshold:
            self._open(record, now)
            return

        total = len(record.events)
        if total >= cfg.min_requests:
            failures = sum(1 for _, failed in record.events if failed)
            if failures / total >= cfg.failure_rate_threshold:
                self._open(record, now)

    @staticmethod
    def _open(record: _BreakerRecord, now: float) -> None:
        record.state = BreakerState.OPEN
        record.opened_at = now
        record.half_open_calls = 0
        record.half_open_successes = 0

    def state(self, model_id: str) -> BreakerState:
        return self._record(model_id).state

    def health_penalty(self, model_id: str) -> float:
        """A [0, 1] penalty reflecting recent failures, used in scoring.

        Lets the router prefer healthier models *before* the breaker trips.
        """
        if not self._config.enabled:
            return 0.0
        record = self._record(model_id)
        now = time.monotonic()
        self._prune(record, now)
        if record.state is BreakerState.OPEN:
            return 1.0
        if record.state is BreakerState.HALF_OPEN:
            return 0.6
        total = len(record.events)
        if total < 3:
            return 0.0
        failures = sum(1 for _, failed in record.events if failed)
        return min(1.0, failures / total)

    def snapshot(self) -> dict[str, dict[str, object]]:
        """Diagnostics for the admin endpoint."""
        now = time.monotonic()
        result: dict[str, dict[str, object]] = {}
        for model_id, record in self._breakers.items():
            self._prune(record, now)
            total = len(record.events)
            failures = sum(1 for _, failed in record.events if failed)
            result[model_id] = {
                "state": record.state.value,
                "window_requests": total,
                "window_failures": failures,
                "consecutive_failures": record.consecutive_failures,
                "reopen_in_seconds": (
                    max(0.0, self._config.open_seconds - (now - record.opened_at))
                    if record.state is BreakerState.OPEN
                    else 0.0
                ),
            }
        return result
