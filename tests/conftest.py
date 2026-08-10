"""Shared test fixtures."""

from __future__ import annotations

import pytest

from chatrouter.config.models import (
    AppConfig,
    ModelConfig,
    ModelTier,
    ProviderConfig,
    QuotaConfig,
    RateLimitConfig,
    RoutingConfig,
    ServerConfig,
    TenantConfig,
)
from chatrouter.core.schemas import ChatCompletionRequest, ChatMessage
from chatrouter.service import GatewayService


def make_config(**overrides) -> AppConfig:
    """Build a small but representative configuration for tests."""
    base = {
        "server": ServerConfig(require_auth=True, admin_api_key="admin-secret"),
        "routing": RoutingConfig(default_model="mid"),
        "providers": [
            ProviderConfig(name="p1", base_url="https://p1.test/v1", api_key="k1"),
        ],
        "models": [
            ModelConfig(
                id="cheap",
                provider="p1",
                upstream_model="cheap-1",
                tier=ModelTier.ECONOMY,
                input_cost_per_1k=0.0001,
                output_cost_per_1k=0.0002,
                context_window=32000,
                quality_prior=0.4,
                latency_prior_ms=500,
            ),
            ModelConfig(
                id="mid",
                provider="p1",
                upstream_model="mid-1",
                tier=ModelTier.STANDARD,
                input_cost_per_1k=0.001,
                output_cost_per_1k=0.002,
                context_window=128000,
                quality_prior=0.65,
                latency_prior_ms=1200,
            ),
            ModelConfig(
                id="strong",
                provider="p1",
                upstream_model="strong-1",
                tier=ModelTier.PREMIUM,
                input_cost_per_1k=0.005,
                output_cost_per_1k=0.015,
                context_window=128000,
                quality_prior=0.85,
                latency_prior_ms=2500,
                supports_vision=True,
            ),
            ModelConfig(
                id="reasoner",
                provider="p1",
                upstream_model="reasoner-1",
                tier=ModelTier.REASONING,
                input_cost_per_1k=0.003,
                output_cost_per_1k=0.01,
                context_window=64000,
                quality_prior=0.9,
                latency_prior_ms=8000,
            ),
        ],
        "tenants": [
            TenantConfig(
                id="acme",
                api_keys=["sk-test-acme"],
                rate_limit=RateLimitConfig(rpm=100, tpm=200000, max_concurrency=10),
                quota=QuotaConfig(period="day", max_requests=1000, max_cost_usd=50),
            )
        ],
    }
    base.update(overrides)
    return AppConfig(**base)


@pytest.fixture
def config() -> AppConfig:
    return make_config()


@pytest.fixture
async def service(config: AppConfig):
    svc = GatewayService(config)
    await svc.start()
    try:
        yield svc
    finally:
        await svc.close()


@pytest.fixture
def tenant(config: AppConfig) -> TenantConfig:
    return config.tenants[0]


def user(text: str) -> ChatMessage:
    return ChatMessage(role="user", content=text)


def assistant(text: str) -> ChatMessage:
    return ChatMessage(role="assistant", content=text)


def system(text: str) -> ChatMessage:
    return ChatMessage(role="system", content=text)


def make_request(messages: list[ChatMessage], model: str = "auto", **kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=messages, **kwargs)
