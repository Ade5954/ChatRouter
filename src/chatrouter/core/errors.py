"""Gateway error hierarchy mapped onto OpenAI-compatible HTTP responses."""

from __future__ import annotations

from typing import Any


class ChatRouterError(Exception):
    """Base class for all gateway errors."""

    status_code: int = 500
    error_type: str = "server_error"
    code: str | None = None

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        param: str | None = None,
        headers: dict[str, str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.param = param
        self.headers = headers or {}
        self.context = context or {}

    def to_payload(self) -> dict[str, Any]:
        from .schemas import error_payload

        return error_payload(self.message, self.error_type, self.code, self.param)


class AuthenticationError(ChatRouterError):
    status_code = 401
    error_type = "invalid_request_error"
    code = "invalid_api_key"


class PermissionError_(ChatRouterError):
    status_code = 403
    error_type = "invalid_request_error"
    code = "model_not_allowed"


class InvalidRequestError(ChatRouterError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "invalid_request"


class ModelNotFoundError(ChatRouterError):
    status_code = 404
    error_type = "invalid_request_error"
    code = "model_not_found"


class RateLimitError(ChatRouterError):
    status_code = 429
    error_type = "rate_limit_error"
    code = "rate_limit_exceeded"

    def __init__(self, message: str, *, retry_after: float | None = None, **kwargs: Any) -> None:
        headers = kwargs.pop("headers", {}) or {}
        if retry_after is not None:
            headers.setdefault("Retry-After", str(max(1, int(retry_after + 0.999))))
        super().__init__(message, headers=headers, **kwargs)
        self.retry_after = retry_after


class QuotaExceededError(ChatRouterError):
    status_code = 429
    error_type = "insufficient_quota"
    code = "quota_exceeded"


class NoCapacityError(ChatRouterError):
    """Every candidate model is saturated or open-circuited."""

    status_code = 503
    error_type = "server_error"
    code = "no_capacity"


class NoCandidateError(ChatRouterError):
    """Routing produced an empty candidate set."""

    status_code = 503
    error_type = "server_error"
    code = "no_route"


class UpstreamError(ChatRouterError):
    """An upstream provider returned an error or was unreachable."""

    status_code = 502
    error_type = "server_error"
    code = "upstream_error"

    def __init__(
        self,
        message: str,
        *,
        model_id: str | None = None,
        upstream_status: int | None = None,
        retryable: bool = True,
        payload: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.model_id = model_id
        self.upstream_status = upstream_status
        self.retryable = retryable
        self.payload = payload
        if upstream_status is not None and 400 <= upstream_status < 500:
            # Client-side upstream errors are surfaced verbatim.
            self.status_code = upstream_status

    def to_payload(self) -> dict[str, Any]:
        if self.payload and "error" in self.payload:
            return self.payload
        return super().to_payload()


class UpstreamTimeoutError(UpstreamError):
    status_code = 504
    code = "upstream_timeout"


class AllUpstreamsFailedError(ChatRouterError):
    """Every attempt in the fallback chain failed."""

    status_code = 502
    error_type = "server_error"
    code = "all_upstreams_failed"
