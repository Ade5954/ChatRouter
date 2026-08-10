"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest

from chatrouter.config.loader import ConfigError, load_config
from chatrouter.config.models import AppConfig, ModelTier, TierThresholds

VALID_YAML = """
providers:
  - name: p1
    base_url: https://p1.test/v1/
    api_key_env: TEST_KEY
models:
  - id: m1
    provider: p1
    upstream_model: up-1
    tier: standard
tenants:
  - id: t1
    api_keys: [k1]
"""


def write(tmp_path, content: str):
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


class TestLoading:
    def test_valid_config_loads(self, tmp_path):
        config = load_config(write(tmp_path, VALID_YAML))
        assert len(config.models) == 1
        assert config.tenants[0].id == "t1"

    def test_base_url_trailing_slash_normalised(self, tmp_path):
        config = load_config(write(tmp_path, VALID_YAML))
        assert config.providers[0].base_url == "https://p1.test/v1"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "absent.yaml")

    def test_invalid_yaml_raises(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config(write(tmp_path, "providers: [unclosed"))

    def test_env_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_URL", "https://env.test/v1")
        yaml = VALID_YAML.replace("https://p1.test/v1/", "${MY_URL}")
        config = load_config(write(tmp_path, yaml))
        assert config.providers[0].base_url == "https://env.test/v1"

    def test_env_default_value(self, tmp_path):
        yaml = VALID_YAML.replace("https://p1.test/v1/", "${UNSET_VAR:-https://fallback.test/v1}")
        config = load_config(write(tmp_path, yaml))
        assert config.providers[0].base_url == "https://fallback.test/v1"

    def test_missing_env_without_default_raises(self, tmp_path):
        yaml = VALID_YAML.replace("https://p1.test/v1/", "${DEFINITELY_UNSET_VAR_XYZ}")
        with pytest.raises(ConfigError, match="not set"):
            load_config(write(tmp_path, yaml))


class TestValidation:
    def test_unknown_provider_reference_rejected(self, tmp_path):
        yaml = VALID_YAML.replace("provider: p1", "provider: ghost")
        with pytest.raises(ConfigError, match="unknown provider"):
            load_config(write(tmp_path, yaml))

    def test_duplicate_model_ids_rejected(self):
        with pytest.raises(Exception, match="duplicate model ids"):
            AppConfig.model_validate(
                {
                    "providers": [{"name": "p", "base_url": "https://x.test"}],
                    "models": [
                        {"id": "dup", "provider": "p", "upstream_model": "a"},
                        {"id": "dup", "provider": "p", "upstream_model": "b"},
                    ],
                }
            )

    def test_unknown_default_model_rejected(self):
        with pytest.raises(Exception, match="default_model"):
            AppConfig.model_validate(
                {
                    "providers": [{"name": "p", "base_url": "https://x.test"}],
                    "models": [{"id": "m", "provider": "p", "upstream_model": "a"}],
                    "routing": {"default_model": "ghost"},
                }
            )

    def test_tenant_unknown_model_rejected(self):
        with pytest.raises(Exception, match="unknown model"):
            AppConfig.model_validate(
                {
                    "providers": [{"name": "p", "base_url": "https://x.test"}],
                    "models": [{"id": "m", "provider": "p", "upstream_model": "a"}],
                    "tenants": [{"id": "t", "allowed_models": ["ghost"]}],
                }
            )

    def test_non_monotonic_thresholds_rejected(self):
        with pytest.raises(Exception, match="strictly increasing"):
            TierThresholds(economy_max=0.8, standard_max=0.5, premium_max=0.9)


class TestTierOrdering:
    def test_ranks_are_ordered(self):
        assert (
            ModelTier.ECONOMY.rank
            < ModelTier.STANDARD.rank
            < ModelTier.PREMIUM.rank
            < ModelTier.REASONING.rank
        )

    def test_threshold_mapping(self):
        thresholds = TierThresholds()
        assert thresholds.tier_for(0.0) is ModelTier.ECONOMY
        assert thresholds.tier_for(0.4) is ModelTier.STANDARD
        assert thresholds.tier_for(0.7) is ModelTier.PREMIUM
        assert thresholds.tier_for(0.95) is ModelTier.REASONING


class TestExampleConfig:
    def test_shipped_example_is_valid(self, monkeypatch):
        """The example must stay loadable, it is the onboarding path."""
        from pathlib import Path

        monkeypatch.setenv("OPENAI_API_KEY", "x")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
        monkeypatch.setenv("VLLM_API_KEY", "x")
        example = Path(__file__).resolve().parents[1] / "config" / "config.example.yaml"
        config = load_config(example)
        assert config.models
        assert config.tenants
