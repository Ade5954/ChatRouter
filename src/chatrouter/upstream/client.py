"""HTTP client pool for upstream OpenAI-compatible providers.

One ``httpx.AsyncClient`` is kept per provider so connection pools, timeouts
and credentials stay isolated. Streaming responses are exposed as an async
iterator of raw SSE lines, which the dispatcher relays to the caller unchanged.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from ..config.loader import resolve_api_key
from ..config.models import ProviderConfig
from ..core.errors import UpstreamError, UpstreamTimeoutError


class UpstreamResponse:
    """A completed non-streaming upstream response."""

    __slots__ = ("status_code", "payload", "headers")

    def __init__(self, status_code: int, payload: dict[str, Any], headers: dict[str, str]) -> None:
        self.status_code = status_code
        self.payload = payload
        self.headers = headers


class ProviderClient:
    """Wraps one provider endpoint."""

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._api_key = resolve_api_key(config.api_key, config.api_key_env)
        limits = httpx.Limits(
            max_connections=config.max_connections,
            max_keepalive_connections=max(10, config.max_connections // 4),
        )
        timeout = httpx.Timeout(
            config.timeout_seconds,
            connect=config.connect_timeout_seconds,
            # Streaming responses may pause between tokens; the read timeout
            # applies per chunk, not to the whole response.
            read=config.timeout_seconds,
        )
        self._client = httpx.AsyncClient(
            base_url=config.base_url, limits=limits, timeout=timeout, follow_redirects=True
        )

    @property
    def name(self) -> str:
        return self._config.name

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"content-type": "application/json", **self._config.extra_headers}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        if extra:
            headers.update(extra)
        return headers

    async def close(self) -> None:
        await self._client.aclose()

    async def chat_completion(
        self, payload: dict[str, Any], extra_headers: dict[str, str] | None = None
    ) -> UpstreamResponse:
        """Perform a non-streaming chat completion call."""
        try:
            response = await self._client.post(
                "/chat/completions", json=payload, headers=self._headers(extra_headers)
            )
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(
                f"provider '{self.name}' timed out: {exc}", retryable=True
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(
                f"provider '{self.name}' is unreachable: {exc}", retryable=True
            ) from exc

        return self._parse(response)

    @asynccontextmanager
    async def chat_completion_stream(
        self, payload: dict[str, Any], extra_headers: dict[str, str] | None = None
    ) -> AsyncIterator[AsyncIterator[bytes]]:
        """Open a streaming chat completion call.

        Yields an iterator over raw SSE lines. Errors raised *before* the first
        chunk are retryable; once bytes are forwarded the stream is committed.
        """
        request = self._client.build_request(
            "POST", "/chat/completions", json=payload, headers=self._headers(extra_headers)
        )
        try:
            response = await self._client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(
                f"provider '{self.name}' timed out: {exc}", retryable=True
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(
                f"provider '{self.name}' is unreachable: {exc}", retryable=True
            ) from exc

        if response.status_code >= 400:
            body = await response.aread()
            await response.aclose()
            raise self._error_from_body(response.status_code, body)

        try:
            yield response.aiter_lines()
        finally:
            await response.aclose()

    async def list_models(self) -> list[dict[str, Any]]:
        """Fetch the provider's model catalogue (best effort)."""
        try:
            response = await self._client.get("/models", headers=self._headers())
            if response.status_code >= 400:
                return []
            data = response.json()
            return data.get("data", []) if isinstance(data, dict) else []
        except (httpx.HTTPError, json.JSONDecodeError, ValueError):
            return []

    def _parse(self, response: httpx.Response) -> UpstreamResponse:
        if response.status_code >= 400:
            raise self._error_from_body(response.status_code, response.content)
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise UpstreamError(
                f"provider '{self.name}' returned malformed JSON", retryable=True
            ) from exc
        return UpstreamResponse(response.status_code, payload, dict(response.headers))

    def _error_from_body(self, status: int, body: bytes) -> UpstreamError:
        """Translate an upstream error body into a gateway error."""
        payload: dict[str, Any] | None = None
        message = f"provider '{self.name}' returned HTTP {status}"
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                payload = parsed
                error = parsed.get("error")
                if isinstance(error, dict) and error.get("message"):
                    message = str(error["message"])
                elif isinstance(error, str):
                    message = error
        except (json.JSONDecodeError, UnicodeDecodeError):
            snippet = body[:200].decode("utf-8", errors="replace")
            if snippet:
                message = f"{message}: {snippet}"

        # 429 and 5xx are transient; 4xx generally indicates a bad request.
        retryable = status == 429 or status >= 500 or status in (408, 409, 425)
        error_cls = UpstreamTimeoutError if status == 504 else UpstreamError
        return error_cls(
            message,
            upstream_status=status,
            retryable=retryable,
            payload=payload,
        )


class ProviderPool:
    """Holds one client per configured provider."""

    def __init__(self, providers: list[ProviderConfig]) -> None:
        self._clients: dict[str, ProviderClient] = {p.name: ProviderClient(p) for p in providers}

    def get(self, name: str) -> ProviderClient:
        client = self._clients.get(name)
        if client is None:
            raise UpstreamError(f"provider '{name}' is not configured", retryable=False)
        return client

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()

    def names(self) -> list[str]:
        return list(self._clients)
