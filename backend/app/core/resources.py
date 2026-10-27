import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager

from app.core.config import Settings
from app.core.database import Database
from app.core.kafka_client import KafkaHealth
from app.core.media import MediaMTXControl
from app.core.redis_client import RedisClient
from app.core.references import ReferenceManager
from app.services.configuration import ConfigurationSynchronizer


class Resources:
    def __init__(self, settings: Settings) -> None:
        self.timeout = settings.dependency_timeout_seconds
        self.database = Database(settings)
        self.redis = RedisClient(settings)
        self.kafka = KafkaHealth(settings)
        self.references = ReferenceManager(settings, self.redis)
        self.media = (
            MediaMTXControl(settings) if settings.mediamtx_api_password.get_secret_value() else None
        )
        self.configuration = (
            ConfigurationSynchronizer(
                settings, self.database, self.redis, self.references, self.media
            )
            if self.media is not None
            else None
        )

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
        if resources.media is not None:
            stack.push_async_callback(resources.media.close)
        async with asyncio.timeout(settings.kafka_request_timeout_ms / 1000):
            await resources.kafka.start()
        task = (
            asyncio.create_task(resources.configuration.run()) if resources.configuration else None
        )
        if task is not None:

            async def stop_configuration() -> None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            stack.push_async_callback(stop_configuration)
        yield resources
