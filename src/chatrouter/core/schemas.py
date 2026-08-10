"""OpenAI-compatible request/response schemas.

The gateway stays permissive: unknown fields are preserved and forwarded
upstream so that new provider features keep working without a gateway release.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "developer", "user", "assistant", "tool", "function"]


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    model_config = ConfigDict(extra="allow")

    role: Role
    # Content may be a plain string or a list of multimodal parts.
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def text(self) -> str:
        """Flatten the content into plain text for analysis purposes."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for part in self.content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in (None, "text", "input_text") and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)

    def has_image(self) -> bool:
        if not isinstance(self.content, list):
            return False
        return any(
            isinstance(p, dict) and p.get("type") in ("image_url", "input_image") for p in self.content
        )


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    include_usage: bool | None = None


class ChatCompletionRequest(BaseModel):
    """Incoming /v1/chat/completions payload."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    stream_options: StreamOptions | None = None

    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    seed: int | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None

    # ChatRouter extension: per-request routing hints.
    chatrouter: RoutingHints | None = None

    def upstream_payload(self, upstream_model: str) -> dict[str, Any]:
        """Serialise for the upstream API, replacing the model name."""
        payload = self.model_dump(exclude_none=True, exclude={"chatrouter"}, by_alias=True)
        payload["model"] = upstream_model
        return payload

    @property
    def requested_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens


class RoutingHints(BaseModel):
    """Optional client-supplied hints under the ``chatrouter`` request field."""

    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(default=None, description="Groups turns of one conversation.")
    min_tier: str | None = None
    max_tier: str | None = None
    prefer_models: list[str] = Field(default_factory=list)
    exclude_models: list[str] = Field(default_factory=list)
    quality_bias: float | None = Field(default=None, ge=0.0, le=1.0)
    # Skip routing entirely and pin the request to one model.
    pin_model: str | None = None


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelCard(BaseModel):
    """Entry of the /v1/models listing."""

    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "chatrouter"
    # Non-standard but useful metadata for clients choosing a virtual model.
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    """Explicit quality feedback for a previously served request."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    # Normalised satisfaction in [0, 1]; or use ``rating`` / ``thumb``.
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    rating: int | None = Field(default=None, ge=1, le=5)
    thumb: Literal["up", "down"] | None = None
    # Behavioural signals the caller can report.
    regenerated: bool = False
    edited: bool = False
    accepted: bool | None = None
    comment: str | None = Field(default=None, max_length=2000)
    tags: list[str] = Field(default_factory=list)

    def normalised_score(self) -> float | None:
        """Collapse the different feedback shapes into a [0, 1] score.

        Delegates to :class:`FeedbackNormalizer` so the mapping is driven by
        configuration and stays consistent with the service layer. Kept as a
        convenience method; the service uses :meth:`FeedbackNormalizer.normalize`
        to also recover the originating signal.
        """
        from ..config.models import FeedbackNormalizationConfig
        from ..routing.feedback_normalizer import FeedbackNormalizer

        result = FeedbackNormalizer(FeedbackNormalizationConfig()).normalize(self)
        return result.score if result else None


class FeedbackResponse(BaseModel):
    accepted: bool
    request_id: str
    model: str | None = None
    applied_score: float | None = None
    # Which signal produced ``applied_score`` (score / rating / thumb / accepted
    # / regenerated / edited). ``None`` when the submission was discarded.
    source: str | None = None
    detail: str | None = None


class ErrorDetail(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


def error_payload(
    message: str,
    error_type: str = "invalid_request_error",
    code: str | None = None,
    param: str | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-shaped error body."""
    return ErrorResponse(
        error=ErrorDetail(message=message, type=error_type, code=code, param=param)
    ).model_dump(exclude_none=True)


def new_request_id() -> str:
    return f"chatcmpl-rt-{uuid.uuid4().hex[:24]}"


ChatCompletionRequest.model_rebuild()
