"""Structured logging.

Uses ``structlog`` when available and degrades to the standard library with a
compatible keyword-argument API, so call sites never need to branch.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_configured = False


class _StdlibLoggerAdapter:
    """Minimal shim providing structlog's keyword-argument interface."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @staticmethod
    def _render(event: str, fields: dict[str, Any]) -> str:
        if not fields:
            return event
        rendered = " ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"{event} {rendered}"

    def debug(self, event: str, **fields: Any) -> None:
        self._logger.debug(self._render(event, fields))

    def info(self, event: str, **fields: Any) -> None:
        self._logger.info(self._render(event, fields))

    def warning(self, event: str, **fields: Any) -> None:
        self._logger.warning(self._render(event, fields))

    def error(self, event: str, **fields: Any) -> None:
        self._logger.error(self._render(event, fields))

    def exception(self, event: str, **fields: Any) -> None:
        self._logger.exception(self._render(event, fields))


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure process-wide logging exactly once."""
    global _configured
    if _configured:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s", stream=sys.stdout, level=log_level, force=True
    )
    # Uvicorn's access log duplicates our request log line.
    logging.getLogger("uvicorn.access").disabled = True

    try:
        import structlog
    except ImportError:
        _configured = True
        return

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json_output
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> Any:
    """Return a logger exposing ``logger.info("event", key=value)``."""
    try:
        import structlog

        return structlog.get_logger(name)
    except ImportError:
        return _StdlibLoggerAdapter(logging.getLogger(name))


def bind_request_context(**fields: Any) -> None:
    """Attach fields to every log line emitted for the current request."""
    try:
        import structlog

        structlog.contextvars.bind_contextvars(**fields)
    except ImportError:
        return


def clear_request_context() -> None:
    try:
        import structlog

        structlog.contextvars.clear_contextvars()
    except ImportError:
        return
