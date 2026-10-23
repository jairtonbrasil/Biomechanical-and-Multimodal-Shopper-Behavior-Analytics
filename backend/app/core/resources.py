import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager

from app.core.config import Settings
from app.core.database import Database
from app.core.kafka_client import KafkaHealth
from app.core.redis_client import RedisClient


class Resources:
    def __init__(self, settings: Settings) -> None:
        self.timeout = settings.dependency_timeout_seconds
        self.database = Database(settings)
        self.redis = RedisClient(settings)
        self.kafka = KafkaHealth(settings)

    async def ready(self) -> bool:
        async def check(operation: Callable[[], Awaitable[None]]) -> bool:
            try:
                async with asyncio.timeout(self.timeout):
                    await operation()
                return True
            except Exception:
                return False

        return all(
            await asyncio.gather(
                check(self.database.health), check(self.redis.health), check(self.kafka.health)
            )
        )


@asynccontextmanager
async def open_resources(settings: Settings) -> AsyncIterator[Resources]:
    resources = Resources(settings)
    async with AsyncExitStack() as stack:
        stack.push_async_callback(resources.database.close)
        stack.push_async_callback(resources.redis.close)
        stack.push_async_callback(resources.kafka.close)
        async with asyncio.timeout(settings.kafka_request_timeout_ms / 1000):
            await resources.kafka.start()
        yield resources
