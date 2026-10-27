"""Reconcile committed database configuration; Redis is only an expiring acknowledgement."""

import asyncio
import json
import logging
from datetime import UTC, datetime
from uuid import uuid5

from redis.exceptions import RedisError
from sqlalchemy import select, text

from app.core.config import Settings
from app.core.database import Database
from app.core.media import MediaMTXControl
from app.core.redis_client import RedisClient
from app.core.references import ReferenceManager
from app.domain.configuration import (
    CONFIGURATION_NAMESPACE,
    camera_revision,
    proxy_path,
    source_revision,
)
from app.models import Camera, Shelf
from app.schemas.geometry import StoredProductROI
from app.schemas.runtime import RuntimeConfiguration, RuntimeShelf

logger = logging.getLogger(__name__)


async def configuration_snapshot(
    camera: Camera, shelves: list[Shelf], references: ReferenceManager
) -> RuntimeConfiguration:
    reference = await references.current(camera)
    configured = []
    for shelf in shelves:
        product = StoredProductROI.model_validate(shelf.product_roi)
        if (
            shelf.is_active
            and reference is not None
            and product.reference_revision == reference.revision
        ):
            configured.append(
                RuntimeShelf(
                    id=shelf.id,
                    name=shelf.name,
                    category=shelf.category,
                    roi_version=shelf.roi_version,
                    reference_revision=reference.revision,
                    product_roi=product.model_dump(include={"type", "coordinate_system", "points"}),
                    presence_roi=shelf.presence_roi,
                )
            )
    revision = uuid5(
        CONFIGURATION_NAMESPACE,
        json.dumps(
            [
                str(camera_revision(camera)),
                str(reference.revision) if reference else None,
                [(str(shelf.id), shelf.updated_at.isoformat()) for shelf in shelves],
            ],
            separators=(",", ":"),
        ),
    )
    return RuntimeConfiguration(
        camera_id=camera.id,
        configuration_revision=revision,
        source_revision=source_revision(camera),
        is_active=camera.is_active,
        proxy_path=proxy_path(camera.id) if camera.is_active else None,
        reference_revision=reference.revision if reference else None,
        shelves=tuple(configured) if camera.is_active else (),
    )


class ConfigurationSynchronizer:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        redis: RedisClient,
        references: ReferenceManager,
        media: MediaMTXControl,
    ) -> None:
        self.settings, self.database, self.redis = settings, database, redis
        self.references, self.media = references, media

    async def reconcile_once(self) -> None:
        async with self.database.transaction() as session:
            if not await session.scalar(text("SELECT pg_try_advisory_xact_lock(724191003)")):
                return
            camera_ids = set(await session.scalars(select(Camera.id)))
            try:
                await self.media.remove_orphans(camera_ids)
            except Exception:
                if camera_ids:
                    await self.redis.client.delete(
                        *(
                            self.redis.key("state", f"configuration:{identifier}")
                            for identifier in camera_ids
                        )
                    )
                raise
            for identifier in sorted(camera_ids):
                acknowledgement_key = self.redis.key("state", f"configuration:{identifier}")
                # A failed or superseded reconciliation must not retain a prior applied state.
                await self.redis.client.delete(acknowledgement_key)
                try:
                    camera = await session.get(Camera, identifier)
                    if camera is None:
                        continue
                    shelves = list(
                        await session.scalars(
                            select(Shelf).where(Shelf.camera_id == identifier).order_by(Shelf.id)
                        )
                    )
                    desired = await configuration_snapshot(camera, shelves, self.references)
                    await self.media.apply(camera)
                    # Never acknowledge a revision superseded during media control I/O.
                    await session.refresh(camera)
                    shelves = list(
                        await session.scalars(
                            select(Shelf)
                            .where(Shelf.camera_id == identifier)
                            .order_by(Shelf.id)
                            .execution_options(populate_existing=True)
                        )
                    )
                    latest = await configuration_snapshot(camera, shelves, self.references)
                    if latest.configuration_revision != desired.configuration_revision:
                        continue
                    acknowledgement = {
                        "status": "applied",
                        "camera_revision": str(camera_revision(camera)),
                        "source_revision": str(source_revision(camera)),
                        "configuration_revision": str(desired.configuration_revision),
                        "applied_at": datetime.now(UTC).isoformat(),
                    }
                    await self.redis.client.set(
                        acknowledgement_key,
                        json.dumps(acknowledgement),
                        ex=max(15, int(self.settings.configuration_poll_seconds * 4)),
                    )
                except Exception:
                    await self.redis.client.delete(acknowledgement_key)
                    logger.warning(
                        "Camera configuration remains pending", extra={"camera_id": str(identifier)}
                    )

    async def run(self) -> None:
        while True:
            try:
                await self.reconcile_once()
            except Exception as error:
                logger.warning(
                    "Configuration reconciliation unavailable",
                    extra={"error_type": type(error).__name__},
                )
            await asyncio.sleep(self.settings.configuration_poll_seconds)

    async def status(self, configuration: RuntimeConfiguration) -> RuntimeConfiguration:
        try:
            value = await self.redis.client.get(
                self.redis.key("state", f"configuration:{configuration.camera_id}")
            )
            acknowledgement = json.loads(value) if value else {}
            if (
                isinstance(acknowledgement, dict)
                and acknowledgement.get("status") == "applied"
                and acknowledgement.get("configuration_revision")
                == str(configuration.configuration_revision)
                and acknowledgement.get("source_revision") == str(configuration.source_revision)
            ):
                applied_at = datetime.fromisoformat(acknowledgement["applied_at"])
                if applied_at.utcoffset() is None:
                    return configuration
                return configuration.model_copy(
                    update={
                        "proxy_state": "APPLIED",
                        "applied_at": applied_at,
                    }
                )
        except (ValueError, TypeError, KeyError, OSError, RedisError):
            pass
        return configuration
