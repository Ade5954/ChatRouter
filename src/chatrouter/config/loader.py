"""Configuration loading with environment variable expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .models import AppConfig

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

DEFAULT_CONFIG_ENV = "CHATROUTER_CONFIG"
DEFAULT_CONFIG_PATH = "config/config.yaml"


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or is invalid."""


def _expand_env(value: Any) -> Any:
    """Recursively substitute ``${VAR}`` / ``${VAR:-default}`` in strings."""
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            resolved = os.environ.get(name)
            if resolved is None:
                if default is None:
                    raise ConfigError(f"environment variable '{name}' is referenced but not set")
                return default
            return resolved

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def resolve_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Pick the configuration file: explicit arg > env var > default path."""
    candidate = path or os.environ.get(DEFAULT_CONFIG_ENV) or DEFAULT_CONFIG_PATH
    return Path(candidate).expanduser().resolve()


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    """Load and validate the application configuration from YAML."""
    config_path = resolve_config_path(path)
    if not config_path.is_file():
        raise ConfigError(f"configuration file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"configuration root must be a mapping, got {type(raw).__name__}")

    try:
        return AppConfig.model_validate(_expand_env(raw))
    except ConfigError:
        raise
    except Exception as exc:  # pydantic ValidationError and friends
        raise ConfigError(f"invalid configuration in {config_path}: {exc}") from exc


def resolve_api_key(api_key: str | None, api_key_env: str | None) -> str | None:
    """Resolve a provider credential, preferring the environment variable."""
    if api_key_env:
        from_env = os.environ.get(api_key_env)
        if from_env:
            return from_env
    return api_key
