"""Collapse heterogeneous explicit feedback into a single normalised score.

A client can express satisfaction many ways — a 0–1 score, a 1–5 star rating,
a thumb, an accept/reject flag, or the behavioural signals of regenerating or
editing the answer. The routing loop only understands one number; this module
is the single place that performs the collapse.

Centralising it has three payoffs:

* **Explainability** — the operator can see, for any submission, *which* signal
  produced the recorded score via the returned ``source``.
* **Tunability** — the mapping lives in :class:`FeedbackNormalizationConfig`
  instead of being buried in request-schema arithmetic.
* **Testability** — the mapping is pure and side-effect free, so every shape is
  unit-tested without standing up a gateway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..config.models import FeedbackConfig, FeedbackNormalizationConfig
from ..core.schemas import FeedbackRequest

# Order in which signals are consulted. Earlier entries win when several are
# present on one request, because they are the more deliberate expressions of
# satisfaction. A 1–5 rating is a considered judgement; "I regenerated" is a
# weak, possibly incidental signal, so it is consulted last.
SignalSource = Literal["score", "rating", "thumb", "accepted", "regenerated", "edited"]


@dataclass(slots=True)
class NormalizedFeedback:
    """A normalised quality score plus the signal that produced it."""

    score: float
    source: SignalSource


class FeedbackNormalizer:
    """Maps a :class:`FeedbackRequest` onto a normalised [0, 1] score."""

    def __init__(self, config: FeedbackNormalizationConfig | None = None) -> None:
        self._cfg = config or FeedbackNormalizationConfig()

    @classmethod
    def from_feedback_config(cls, config: FeedbackConfig) -> FeedbackNormalizer:
        return cls(config.normalization)

    def normalize(self, feedback: FeedbackRequest) -> NormalizedFeedback | None:
        """Return the normalised score, or ``None`` if no signal was supplied.

        ``None`` means the submission carries no usable evidence and should be
        rejected by the caller (it is not a valid feedback payload).
        """
        if feedback.score is not None:
            return NormalizedFeedback(score=feedback.score, source="score")

        if feedback.rating is not None:
            # 1 → 0.0, 5 → 1.0; linear across the star range.
            return NormalizedFeedback(score=(feedback.rating - 1) / 4, source="rating")

        if feedback.thumb is not None:
            score = (
                self._cfg.thumb_up_score
                if feedback.thumb == "up"
                else self._cfg.thumb_down_score
            )
            return NormalizedFeedback(score=score, source="thumb")

        if feedback.accepted is not None:
            score = self._cfg.accept_score if feedback.accepted else self._cfg.reject_score
            return NormalizedFeedback(score=score, source="accepted")

        if feedback.regenerated:
            return NormalizedFeedback(score=self._cfg.regenerated_score, source="regenerated")

        if feedback.edited:
            return NormalizedFeedback(score=self._cfg.edited_score, source="edited")

        return None
