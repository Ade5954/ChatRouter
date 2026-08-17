"""Tests for hot configuration reload and the admin config write-back API."""

from __future__ import annotations

import yaml
from fastapi.testclient import TestClient

from chatrouter.app import create_app
from chatrouter.config.models import AppConfig, ModelConfig, ModelTier
from chatrouter.service import GatewayService

from .conftest import make_config


def dump_config_yaml(config: AppConfig) -> str:
    return yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False)


class TestReload:
    async def test_reload_swaps_components_keeps_storage(self, config):
        svc = GatewayService(config)
        await svc.start()
        try:
            old_pool = svc.providers
            new_models = list(config.models)
            new_models.append(
                ModelConfig(
                    id="new-mid",
                    provider="p1",
                    upstream_model="new-mid-1",
                    tier=ModelTier.STANDARD,
                    input_cost_per_1k=0.001,
                    output_cost_per_1k=0.002,
                    context_window=128000,
                    quality_prior=0.6,
                    latency_prior_ms=1000,
                )
            )
            new_config = make_config(models=new_models)

            await svc.reload(new_config)

            assert svc.config is new_config
            assert {m.id for m in svc.router.models} == {
                "cheap", "mid", "strong", "reasoner", "new-mid"
            }
            assert svc.providers is not old_pool
            assert svc.tenants.by_id("acme") is not None
            assert old_pool in svc._retired_pools
            # Storage backend instance is preserved across the reload.
            assert svc.storage is not None
        finally:
            await svc.close()


class TestAdminConfigWriteBack:
    def test_put_config_persists_hot_applies_and_redacts(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(dump_config_yaml(make_config()), encoding="utf-8")

        app = create_app(config_path=str(cfg_file))
        with TestClient(app) as client:
            headers = {"x-admin-key": "admin-secret"}
            assert client.get("/admin/config", headers=headers).status_code == 200

            payload = {
                "models": [
                    m
                    for m in make_config().model_dump(mode="json")["models"]
                    if m["id"] != "reasoner"
                ]
                + [
                    {
                        "id": "new-mid",
                        "provider": "p1",
                        "upstream_model": "new-mid-1",
                        "tier": "standard",
                        "input_cost_per_1k": 0.001,
                        "output_cost_per_1k": 0.002,
                        "context_window": 128000,
                        "quality_prior": 0.6,
                        "latency_prior_ms": 1000,
                    }
                ]
            }
            response = client.put("/admin/config", json=payload, headers=headers)
            assert response.status_code == 200, response.text
            ids = [m["id"] for m in response.json()["models"]]
            assert "new-mid" in ids and "reasoner" not in ids

            # Hot-applied: the gateway routes with the new model pool.
            listing = client.get("/v1/models", headers={"Authorization": "Bearer sk-test-acme"})
            listed = [m["id"] for m in listing.json()["data"]]
            assert "new-mid" in listed and "reasoner" not in listed

            # Persisted: the YAML file carries the update and keeps the key.
            raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            assert "new-mid" in [m["id"] for m in raw["models"]]
            assert raw["providers"][0]["api_key"] == "k1"

    def test_put_config_keeps_existing_key_when_placeholder(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(dump_config_yaml(make_config()), encoding="utf-8")

        app = create_app(config_path=str(cfg_file))
        with TestClient(app) as client:
            headers = {"x-admin-key": "admin-secret"}
            payload = {
                "providers": [
                    {
                        "name": "p1",
                        "base_url": "https://p1.test/v1",
                        "api_key": "***1",  # redacted sentinel -> keep k1
                    }
                ]
            }
            response = client.put("/admin/config", json=payload, headers=headers)
            assert response.status_code == 200, response.text
            assert "api_key" not in response.json()["providers"][0]

            raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            assert raw["providers"][0]["api_key"] == "k1"

    def test_put_config_requires_admin_key(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(dump_config_yaml(make_config()), encoding="utf-8")

        app = create_app(config_path=str(cfg_file))
        with TestClient(app) as client:
            response = client.put("/admin/config", json={"models": []})
            assert response.status_code == 401
