from typing import Literal

from redis.asyncio import Redis

from app.core.config import Settings

Namespace = Literal["state", "cache", "channel"]


class RedisClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: Redis = Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password.get_secret_value(),
            max_connections=settings.redis_max_connections,
            socket_connect_timeout=settings.dependency_timeout_seconds,
            socket_timeout=settings.dependency_timeout_seconds,
            decode_responses=True,
        )

    def key(self, namespace: Namespace, scope: str) -> str:
        if not scope:
            raise ValueError("A nonempty Redis key scope is required")
        return f"{self.settings.redis_namespace}:{namespace}:{scope}"

    async def get_cache(self, scope: str) -> str | None:
        return await self.client.get(self.key("cache", scope))

    async def set_cache(self, scope: str, value: str, *, ttl: int | None = None) -> None:
        lifetime = self.settings.cache_ttl_seconds if ttl is None else ttl
        if not 60 <= lifetime <= 300:
            raise ValueError("Cache TTL must be between 60 and 300 seconds")
        await self.client.set(self.key("cache", scope), value, ex=lifetime)

    async def get_state(self, scope: str) -> str | None:
        return await self.client.get(self.key("state", scope))

    async def set_state(self, scope: str, value: str) -> None:
        await self.client.set(self.key("state", scope), value, ex=self.settings.state_ttl_seconds)

    async def health(self) -> None:
        await self.client.ping()

    async def close(self) -> None:
        await self.client.aclose()
