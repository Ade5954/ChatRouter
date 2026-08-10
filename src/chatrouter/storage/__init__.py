"""Storage backends."""

from ..config.models import StorageConfig
from .base import QuotaUsage, RateLimitVerdict, Storage
from .memory import MemoryStorage
from .redis_store import RedisStorage

__all__ = [
    "MemoryStorage",
    "QuotaUsage",
    "RateLimitVerdict",
    "RedisStorage",
    "Storage",
    "build_storage",
]


def build_storage(config: StorageConfig) -> Storage:
    """Instantiate the configured storage backend."""
    if config.backend == "redis":
        return RedisStorage(config.redis_url, config.key_prefix)
    return MemoryStorage(config.key_prefix)
