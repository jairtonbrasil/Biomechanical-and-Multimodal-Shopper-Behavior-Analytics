"""Consistent desired-state snapshots and single-owner camera leases."""

import asyncio
import json
from dataclasses import dataclass
from uuid import UUID, uuid4

from sqlalchemy import select

from app.core.database import Database
from app.core.redis_client import RedisClient
from app.core.references import ReferenceManager
from app.domain.configuration import camera_revision
from app.models import Camera, Shelf
from app.repositories.assets import ShelfRepository
from app.repositories.errors import EntityNotFoundError
from app.schemas.geometry import GeometrySnapshot
from app.schemas.runtime import RuntimeConfiguration
from app.services.configuration import configuration_snapshot


class ConfigurationUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class VisionConfiguration:
    runtime: RuntimeConfiguration
    geometry: tuple[GeometrySnapshot, ...]
    camera_revision: UUID


async def load_configuration(
    camera_id: UUID, database: Database, redis: RedisClient, references: ReferenceManager
) -> VisionConfiguration:
    async with database.transaction() as session:
        camera = await session.scalar(
            select(Camera).where(Camera.id == camera_id).with_for_update(read=True)
        )
        if camera is None:
            raise EntityNotFoundError("Camera does not exist")
        shelves = list(
            await session.scalars(
                select(Shelf).where(Shelf.camera_id == camera_id).order_by(Shelf.id)
            )
        )
        runtime = await configuration_snapshot(camera, shelves, references)
        if not runtime.is_active or not runtime.reference_revision or not runtime.shelves:
            raise ConfigurationUnavailable("Activate aligned shelves and capture a reference first")
        if len(runtime.shelves) > 32:
            raise ConfigurationUnavailable("A version 2.0 observation supports at most 32 shelves")
        value = await redis.get_state(f"configuration:{camera_id}")
        try:
            ack = json.loads(value or "null")
        except ValueError:
            ack = None
        if not (
            isinstance(ack, dict)
            and ack.get("status") == "applied"
            and ack.get("configuration_revision") == str(runtime.configuration_revision)
            and ack.get("source_revision") == str(runtime.source_revision)
            and ack.get("camera_revision") == str(camera_revision(camera))
        ):
            raise ConfigurationUnavailable("Wait for matching proxy configuration")
        active = {item.id for item in runtime.shelves}
        geometry = tuple(ShelfRepository.snapshot(item) for item in shelves if item.id in active)
        return VisionConfiguration(runtime, geometry, camera_revision(camera))


class CameraLease:
    RELEASE = """
    if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end
    return 0
    """
    RENEW = """
    if redis.call('GET', KEYS[1]) == ARGV[1] then
      return redis.call('EXPIRE', KEYS[1], ARGV[2])
    end
    return 0
    """

    def __init__(self, redis: RedisClient, camera_id: UUID, *, ttl: int = 30) -> None:
        self.redis, self.ttl = redis, ttl
        self.key = redis.key("state", f"vision-owner:{camera_id}")
        self.owner = str(uuid4())
        self.lost = False
        self.task: asyncio.Task | None = None

    async def __aenter__(self):
        if not await self.redis.client.set(self.key, self.owner, nx=True, ex=self.ttl):
            raise ConfigurationUnavailable("A vision worker already owns this camera")
        self.task = asyncio.create_task(self._renew())
        return self

    async def _renew(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.ttl / 3)
                if not await self.redis.client.eval(self.RENEW, 1, self.key, self.owner, self.ttl):
                    self.lost = True
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            self.lost = True

    async def __aexit__(self, *args) -> None:
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        try:
            await self.redis.client.eval(self.RELEASE, 1, self.key, self.owner)
        except Exception:
            pass
