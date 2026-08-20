"""Tests for hot configuration reload and the admin config write-back API."""

from __future__ import annotations

import asyncio

import yaml
from fastapi.testclient import TestClient

from chatrouter.app import create_app
from chatrouter.config.models import AppConfig, ModelConfig, ModelTier
from chatrouter.service import GatewayService
from chatrouter.storage.memory import MemoryStorage

from .conftest import make_config


def dump_config_yaml(config: AppConfig) -> str:
    return yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False)


def _stored_config(client: TestClient) -> dict:
    """Read the authoritative configuration document the replica is using.

    The admin PUT persists into the shared storage backend; this helper reads
    it back synchronously so a synchronous TestClient test can assert on it.
    MemoryStorage (the default in tests) has no event-loop affinity, so a
    fresh loop via ``asyncio.run`` is safe here.
    """
    storage = client.app.state.service.storage
    return asyncio.run(storage.get_config())


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

            # Persisted to the shared store: the document carries the update
            # and keeps the provider secret (which the round-trip never
            # surfaces in cleartext via GET).
            stored = _stored_config(client)
            assert "new-mid" in [m["id"] for m in stored["models"]]
            assert stored["providers"][0]["api_key"] == "k1"

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

            # The placeholder round-trip kept the secret in the store.
            stored = _stored_config(client)
            assert stored["providers"][0]["api_key"] == "k1"

    def test_put_config_requires_admin_key(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(dump_config_yaml(make_config()), encoding="utf-8")

        app = create_app(config_path=str(cfg_file))
        with TestClient(app) as client:
            response = client.put("/admin/config", json={"models": []})
            assert response.status_code == 401


class TestCrossReplicaReload:
    """Phase 2: a write on one replica must hot-apply on every other replica.

    These tests build two ``GatewayService`` instances sharing one
    ``MemoryStorage``: replica A writes via ``apply_config_update`` (the
    same path the admin endpoint uses), and replica B's background
    subscription task must observe the published version and reload from
    storage. The publisher itself must not double-reload from its own
    notification.
    """

    @staticmethod
    def _config_with_extra_model() -> dict:
        base = make_config().model_dump(mode="json")
        base["models"] = [
            m for m in base["models"] if m["id"] != "reasoner"
        ] + [
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
        return base

    async def test_write_on_one_replica_reloads_other_replica(self, config):
        # Two services, one shared in-memory store.
        shared_storage = MemoryStorage()
        await shared_storage.start()
        # Seed the store so both replicas start from the same authoritative
        # configuration (mirrors the bootstrap path in app.py).
        await shared_storage.set_config(config.model_dump(mode="json"))

        replica_a = GatewayService(config, storage=shared_storage)
        replica_b = GatewayService(config, storage=shared_storage)
        await replica_a.start()
        await replica_b.start()
        try:
            assert "new-mid" not in {m.id for m in replica_b.router.models}

            new_config_dict = self._config_with_extra_model()
            await replica_a.apply_config_update(new_config_dict)

            # replica_a applied immediately.
            assert "new-mid" in {m.id for m in replica_a.router.models}

            # replica_b must pick up the change via the pub/sub notification.
            # The watcher pushes the new version onto an asyncio.Queue and the
            # consumer reloads synchronously, so a short bounded wait is
            # enough; we poll so the test fails fast rather than after the
            # full timeout when the watcher is broken.
            for _ in range(100):
                if "new-mid" in {m.id for m in replica_b.router.models}:
                    break
                await asyncio.sleep(0.01)
            assert "new-mid" in {m.id for m in replica_b.router.models}
            assert "reasoner" not in {m.id for m in replica_b.router.models}
            # The watcher bumped replica_b's applied version to match.
            assert replica_b._applied_config_version == replica_a._applied_config_version
        finally:
            await replica_a.close()
            await replica_b.close()
            await shared_storage.close()

    async def test_publisher_does_not_double_reload_from_own_notification(
        self, config
    ):
        shared_storage = MemoryStorage()
        await shared_storage.start()
        await shared_storage.set_config(config.model_dump(mode="json"))

        svc = GatewayService(config, storage=shared_storage)
        await svc.start()
        try:
            initial_pool = svc.providers
            initial_version = svc._applied_config_version

            await svc.apply_config_update(self._config_with_extra_model())

            # The provider pool was swapped exactly once (by the local reload
            # inside apply_config_update); the self-published notification
            # must not trigger a second reload.
            assert svc.providers is not initial_pool
            assert initial_pool in svc._retired_pools
            assert svc._applied_config_version == initial_version + 1

            # Give the watcher a moment to process any pending notification
            # (the self-publish is delivered asynchronously via the queue).
            await asyncio.sleep(0.05)
            # Still exactly one retired pool — no second reload happened.
            assert len(svc._retired_pools) == 1
        finally:
            await svc.close()
            await shared_storage.close()

    async def test_stale_notification_is_ignored(self, config):
        shared_storage = MemoryStorage()
        await shared_storage.start()
        await shared_storage.set_config(config.model_dump(mode="json"))

        svc = GatewayService(config, storage=shared_storage)
        await svc.start()
        try:
            initial_version = svc._applied_config_version
            initial_pool = svc.providers

            # Apply a real update (version N+1).
            await svc.apply_config_update(self._config_with_extra_model())
            assert svc._applied_config_version == initial_version + 1

            # Manually publish a stale version (<= applied). The watcher
            # must ignore it: no reload, provider pool unchanged.
            await shared_storage.publish_config_reload(initial_version)
            await asyncio.sleep(0.05)
            assert svc.providers is not initial_pool  # still the reloaded one
            assert len(svc._retired_pools) == 1  # still exactly one retired pool
        finally:
            await svc.close()
            await shared_storage.close()


class TestConfigReconciliation:
    """Belt-and-suspenders: the periodic reconciler catches a lost notification.

    Simulates a notification being lost (we write the new config + bump the
    version directly in storage, bypassing ``publish_config_reload``) and
    verifies the reconciler detects the version drift and reloads.
    """

    @staticmethod
    def _config_with_extra_model() -> dict:
        base = make_config().model_dump(mode="json")
        base["models"] = [
            m for m in base["models"] if m["id"] != "reasoner"
        ] + [
            {
                "id": "reconciled-mid",
                "provider": "p1",
                "upstream_model": "reconciled-1",
                "tier": "standard",
                "input_cost_per_1k": 0.001,
                "output_cost_per_1k": 0.002,
                "context_window": 128000,
                "quality_prior": 0.6,
                "latency_prior_ms": 1000,
            }
        ]
        return base

    async def test_reconciler_catches_lost_notification(self, config):
        shared_storage = MemoryStorage()
        await shared_storage.start()
        await shared_storage.set_config(config.model_dump(mode="json"))

        svc = GatewayService(config, storage=shared_storage)
        # Tighten the reconcile interval so the test runs fast.
        svc._reconcile_interval_seconds = 0.05
        await svc.start()
        try:
            assert "reconciled-mid" not in {m.id for m in svc.router.models}

            # Simulate a lost notification: another replica wrote a new config
            # to storage (bumping the version) but the publish() call was
            # lost — so our watcher never hears about it.
            await shared_storage.set_config(self._config_with_extra_model())

            # The watcher has not been notified, so the service still has the
            # old model pool. Poll until the reconciler fires and reloads.
            for _ in range(100):
                if "reconciled-mid" in {m.id for m in svc.router.models}:
                    break
                await asyncio.sleep(0.01)
            assert "reconciled-mid" in {m.id for m in svc.router.models}
            assert svc._applied_config_version == await shared_storage.get_config_version()
        finally:
            await svc.close()
            await shared_storage.close()
