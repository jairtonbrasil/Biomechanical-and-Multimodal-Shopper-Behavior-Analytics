import asyncio
import json
import signal
from datetime import UTC, datetime

from pydantic import ValidationError
from sqlalchemy import select

from app.core.config import Settings
from app.core.database import Database
from app.core.redis_client import RedisClient
from app.core.references import ReferenceStore, capture_jpeg, reference_pending_scope
from app.domain.configuration import camera_revision
from app.domain.configuration import source_revision as camera_source_revision
from app.models.camera import Camera
from app.models.shelf import Shelf
from app.models.shelf_interaction import ShelfInteraction
from app.schemas.references import ReferenceCaptureJob

FINISH_CAPTURE = """
redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
if redis.call('GET', KEYS[2]) == ARGV[3] then redis.call('DEL', KEYS[2]) end
return 1
"""


async def process_job(
    job: ReferenceCaptureJob,
    settings: Settings,
    database: Database,
    store: ReferenceStore,
    redis: RedisClient,
) -> dict[str, str]:
    age = (datetime.now(UTC) - job.requested_at).total_seconds()
    if not 0 <= age <= settings.reference_job_ttl_seconds:
        return {"status": "unavailable"}
    async with database.transaction() as session:
        camera = await session.get(Camera, job.camera_id)
        if (
            camera is None
            or not camera.is_active
            or camera_source_revision(camera) != job.source_revision
            or camera_revision(camera) != job.configuration_revision
        ):
            return {"status": "conflict"}
    raw_acknowledgement = await redis.get_state(f"configuration:{job.camera_id}")
    try:
        acknowledgement = json.loads(raw_acknowledgement or "null")
        applied = (
            isinstance(acknowledgement, dict)
            and acknowledgement.get("camera_revision") == str(job.configuration_revision)
            and acknowledgement.get("source_revision") == str(job.source_revision)
            and acknowledgement.get("status") == "applied"
        )
    except (TypeError, ValueError):
        applied = False
    if not applied:
        return {"status": "unavailable"}
    content = await capture_jpeg(settings, job.camera_id)
    async with database.transaction() as session:
        camera = await session.scalar(
            select(Camera).where(Camera.id == job.camera_id).with_for_update()
        )
        if (
            camera is None
            or not camera.is_active
            or camera_source_revision(camera) != job.source_revision
            or camera_revision(camera) != job.configuration_revision
        ):
            return {"status": "conflict"}
        active_shelf = await session.scalar(
            select(Shelf.id).where(Shelf.camera_id == camera.id, Shelf.is_active).limit(1)
        )
        active_interaction = await session.scalar(
            select(ShelfInteraction.id)
            .join(Shelf, Shelf.id == ShelfInteraction.shelf_id)
            .where(Shelf.camera_id == camera.id, ShelfInteraction.status == "ACTIVE")
            .limit(1)
        )
        if active_shelf is not None or active_interaction is not None:
            return {"status": "conflict"}
        # The row lock fences publication against camera updates after a slow capture.
        publication = asyncio.create_task(
            asyncio.to_thread(store.publish, camera.id, job.source_revision, content)
        )
        cancelled = False
        while True:
            try:
                frame = await asyncio.shield(publication)
                break
            except asyncio.CancelledError:
                # A filesystem thread cannot be cancelled; retain the lock until it finishes.
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError
    return {"status": "ready", "revision": str(frame.revision)}


async def run(settings: Settings) -> None:
    task = asyncio.current_task()
    assert task is not None
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    database = Database(settings)
    redis = RedisClient(settings)
    store = ReferenceStore(settings)
    try:
        while True:
            message = await redis.client.blpop(redis.key("channel", "reference:jobs"), timeout=1)
            if message is None:
                continue
            if len(message[1]) > 4096:
                continue
            try:
                job = ReferenceCaptureJob.model_validate_json(message[1])
            except (ValidationError, ValueError):
                continue
            try:
                result = await process_job(job, settings, database, store, redis)
            except Exception:
                result = {"status": "unavailable"}
            await redis.client.eval(
                FINISH_CAPTURE,
                2,
                redis.key("channel", f"reference:result:{job.id}"),
                redis.key("channel", reference_pending_scope(job)),
                json.dumps(result),
                settings.reference_job_ttl_seconds,
                str(job.id),
            )
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
        await redis.close()
        await database.close()


def main() -> None:
    try:
        asyncio.run(run(Settings()))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception:
        raise SystemExit("Reference capture worker stopped; dependencies are unavailable") from None


if __name__ == "__main__":
    main()
