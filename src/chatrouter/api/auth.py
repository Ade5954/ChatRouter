"""API key authentication and tenant resolution."""

from __future__ import annotations

import hmac

from ..config.models import AppConfig, TenantConfig
from ..core.errors import AuthenticationError

_ANONYMOUS = TenantConfig(id="anonymous", name="Anonymous")


class TenantRegistry:
    """Resolves an inbound API key to its tenant in constant time."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._by_key: dict[str, TenantConfig] = {}
        for tenant in config.tenants:
            if not tenant.enabled:
                continue
            for key in tenant.api_keys:
                if key:
                    self._by_key[key] = tenant
        self._by_id = {t.id: t for t in config.tenants}

    @staticmethod
    def _extract_key(authorization: str | None, api_key_header: str | None) -> str | None:
        if authorization:
            scheme, _, value = authorization.partition(" ")
            if scheme.lower() == "bearer" and value.strip():
                return value.strip()
            if authorization.strip() and not _:
                return authorization.strip()
        if api_key_header:
            return api_key_header.strip()
        return None

    def resolve(self, authorization: str | None, api_key_header: str | None = None) -> TenantConfig:
        """Return the tenant owning the presented key."""
        if not self._config.server.require_auth:
            key = self._extract_key(authorization, api_key_header)
            return self._by_key.get(key or "", _ANONYMOUS)

        key = self._extract_key(authorization, api_key_header)
        if not key:
            raise AuthenticationError(
                "missing API key; provide it as 'Authorization: Bearer <key>'",
                code="missing_api_key",
            )

        tenant = self._by_key.get(key)
        if tenant is None:
            # Compare against every key anyway to avoid a timing oracle that
            # would reveal which prefixes are valid.
            for known in self._by_key:
                hmac.compare_digest(known, key)
            raise AuthenticationError("invalid API key provided")
        if not tenant.enabled:
            raise AuthenticationError(f"tenant '{tenant.id}' is disabled", code="tenant_disabled")
        return tenant

    def by_id(self, tenant_id: str) -> TenantConfig | None:
        return self._by_id.get(tenant_id)

    def all(self) -> list[TenantConfig]:
        return list(self._config.tenants)


def verify_admin_key(config: AppConfig, presented: str | None) -> None:
    """Guard the admin API surface."""
    import os

    expected = config.server.admin_api_key or os.environ.get(config.server.admin_api_key_env)
    if not expected:
        raise AuthenticationError(
            "admin API is disabled because no admin key is configured", code="admin_disabled"
        )
    if not presented or not hmac.compare_digest(expected, presented):
        raise AuthenticationError("invalid admin key", code="invalid_admin_key")
