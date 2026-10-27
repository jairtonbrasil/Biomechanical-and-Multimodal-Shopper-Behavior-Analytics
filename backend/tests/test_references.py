import asyncio
import json
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core.references import (
    ReferenceManager,
    ReferenceStore,
    ReferenceUnavailableError,
    capture_jpeg,
    jpeg_dimensions,
)
from app.domain.configuration import camera_revision, source_revision
from app.models.camera import Camera
from app.repositories.errors import VersionConflictError
from app.schemas.references import ReferenceCaptureJob
from workers.reference_capture.main import process_job


def jpeg(width: int = 640, height: int = 360) -> bytes:
    return (
        b"\xff\xd8\xff\xc0\x00\x0b\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x01\x01\x11\x00\xff\xd9"
    )


@pytest.fixture
def reference_settings(settings, tmp_path):
    return settings.model_copy(
        update={
            "reference_directory": tmp_path,
            "reference_max_bytes": 4096,
            "reference_storage_limit_bytes": 1024 * 1024,
            "reference_max_width": 1920,
            "reference_timeout_seconds": 0.05,
            "reference_queue_limit": 4,
            "reference_job_ttl_seconds": 60,
        }
    )


@pytest.fixture
def camera():
    return Camera(
        id=uuid4(),
        name="Reference camera",
        stream_url="rtsp://camera.internal/live",
        is_active=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def test_reference_storage_preserves_history_and_fences_sources(reference_settings):
    store = ReferenceStore(reference_settings)
    camera_id, source = uuid4(), uuid4()
    first = store.publish(camera_id, source, jpeg())
    assert store.read(camera_id, source) == first
    assert store.read(camera_id, uuid4()) is None
    second = store.publish(camera_id, source, jpeg(800, 600))
    assert second.revision != first.revision
    assert store.read(camera_id, source) == second
    assert store.read(camera_id, source, first.revision) == first
    assert (reference_settings.reference_directory / str(camera_id)).stat().st_mode & 0o777 == 0o700
    for artifact in (reference_settings.reference_directory / str(camera_id)).iterdir():
        assert artifact.stat().st_mode & 0o777 == 0o600


def test_reference_storage_detects_missing_invalid_and_tampered_files(reference_settings):
    store = ReferenceStore(reference_settings)
    camera_id, source = uuid4(), uuid4()
    assert store.read(camera_id, source) is None
    frame = store.publish(camera_id, source, jpeg())
    directory = reference_settings.reference_directory / str(camera_id)
    (directory / f"{frame.revision}.jpg").write_bytes(jpeg(800, 600))
    assert store.read(camera_id, source) is None
    (directory / "current.json").write_text("not valid metadata")
    assert store.read(camera_id, source) is None


def test_reference_storage_rejects_symlink_files(reference_settings, tmp_path):
    store = ReferenceStore(reference_settings)
    camera_id, source = uuid4(), uuid4()
    frame = store.publish(camera_id, source, jpeg())
    image = reference_settings.reference_directory / str(camera_id) / f"{frame.revision}.jpg"
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(image.read_bytes())
    image.unlink()
    image.symlink_to(outside)
    assert store.read(camera_id, source) is None


def test_reference_storage_quota_preserves_the_previous_manifest(reference_settings):
    store = ReferenceStore(reference_settings)
    camera_id, source = uuid4(), uuid4()
    first = store.publish(camera_id, source, jpeg())
    store.storage_limit = 1
    with pytest.raises(ReferenceUnavailableError, match="quota"):
        store.publish(camera_id, source, jpeg(800, 600))
    assert store.read(camera_id, source) == first


@pytest.mark.parametrize("content", [b"", b"not an image", b"\xff\xd8\xff\xda\xff\xd9"])
def test_jpeg_validation_rejects_missing_frame_dimensions(content):
    with pytest.raises(ReferenceUnavailableError):
        jpeg_dimensions(content)


@pytest.mark.parametrize("content", [jpeg(1921), jpeg(640, 1921), jpeg() + b"x" * 4096])
def test_reference_storage_enforces_dimensions_and_bytes(reference_settings, content):
    with pytest.raises(ReferenceUnavailableError):
        ReferenceStore(reference_settings).publish(uuid4(), uuid4(), content)


def fake_redis():
    client = SimpleNamespace(eval=AsyncMock(), get=AsyncMock())
    return SimpleNamespace(client=client, key=lambda namespace, value: f"test:{namespace}:{value}")


def acknowledged_redis(camera):
    return SimpleNamespace(
        get_state=AsyncMock(
            return_value=json.dumps(
                {
                    "status": "applied",
                    "camera_revision": str(camera_revision(camera)),
                    "source_revision": str(source_revision(camera)),
                }
            )
        )
    )


async def test_reference_cache_is_stable_without_enqueueing(reference_settings, camera):
    redis = fake_redis()
    manager = ReferenceManager(reference_settings, redis)
    first = manager.store.publish(camera.id, source_revision(camera), jpeg())
    assert await manager.get(camera) == first
    assert await manager.get(camera) == first
    redis.client.eval.assert_not_awaited()
    camera.name = "Changed display name"
    camera.updated_at = datetime.now(UTC)
    assert await manager.current(camera) == first
    camera.stream_url = "rtsp://another-camera.internal/live"
    assert await manager.current(camera) is None


async def test_inactive_cameras_cannot_capture_or_read(reference_settings, camera):
    redis = fake_redis()
    manager = ReferenceManager(reference_settings, redis)
    camera.is_active = False
    assert await manager.current(camera) is None
    with pytest.raises(ReferenceUnavailableError, match="Inactive"):
        await manager.get(camera)
    redis.client.eval.assert_not_awaited()


async def test_reference_capture_queue_is_bounded(reference_settings, camera):
    redis = fake_redis()
    redis.client.eval.return_value = ""
    with pytest.raises(ReferenceUnavailableError, match="queue is full"):
        await ReferenceManager(reference_settings, redis).get(camera)


async def test_reference_capture_refresh_reads_immutable_result(reference_settings, camera):
    redis = fake_redis()
    manager = ReferenceManager(reference_settings, redis)
    first = manager.store.publish(camera.id, source_revision(camera), jpeg())
    second = manager.store.publish(camera.id, source_revision(camera), jpeg(800, 600))
    redis.client.eval.return_value = str(uuid4())
    redis.client.get.return_value = json.dumps({"status": "ready", "revision": str(first.revision)})
    assert await manager.get(camera, refresh=True) == first
    assert await manager.current(camera) == second
    arguments = redis.client.eval.call_args.args
    job = ReferenceCaptureJob.model_validate_json(arguments[5])
    assert job.refresh
    assert camera.stream_url not in str(arguments)


async def test_pending_capture_keys_distinguish_configuration_edits(reference_settings, camera):
    redis = fake_redis()
    manager = ReferenceManager(reference_settings, redis)
    frame = manager.store.publish(camera.id, source_revision(camera), jpeg())
    redis.client.eval.return_value = str(uuid4())
    redis.client.get.return_value = json.dumps({"status": "ready", "revision": str(frame.revision)})
    await manager.get(camera, refresh=True)
    previous_revision = camera_revision(camera)
    camera.updated_at += timedelta(seconds=1)
    await manager.get(camera, refresh=True)
    previous_key, current_key = [call.args[2] for call in redis.client.eval.call_args_list]
    assert previous_key != current_key
    assert str(previous_revision) in previous_key
    assert str(camera_revision(camera)) in current_key
    assert str(source_revision(camera)) in previous_key
    assert str(source_revision(camera)) in current_key


async def test_reference_capture_reports_revision_conflict(reference_settings, camera):
    redis = fake_redis()
    redis.client.eval.return_value = str(uuid4())
    redis.client.get.return_value = '{"status":"conflict"}'
    with pytest.raises(VersionConflictError):
        await ReferenceManager(reference_settings, redis).get(camera)


async def test_reference_capture_broker_failure_has_no_credentials(reference_settings, camera):
    redis = fake_redis()
    redis.client.eval.side_effect = RuntimeError("rtsp://user:secret@camera.internal/live")
    with pytest.raises(ReferenceUnavailableError) as result:
        await ReferenceManager(reference_settings, redis).get(camera)
    assert "secret" not in str(result.value)


class FakeProcess:
    def __init__(self, stdout, code=0):
        self.stdout = SimpleNamespace(read=stdout)
        self.code = code
        self.returncode = None
        self.killed = False

    async def wait(self):
        self.returncode = self.code
        return self.code

    def kill(self):
        self.killed = True


async def test_ffmpeg_capture_uses_proxy_and_checks_exit_status(reference_settings, monkeypatch):
    process = FakeProcess(AsyncMock(side_effect=[jpeg(), b""]))
    create = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    camera_id = uuid4()
    assert await capture_jpeg(reference_settings, camera_id) == jpeg()
    arguments = create.call_args.args
    assert f"/cameras/{camera_id}" in arguments[arguments.index("-i") + 1]
    assert "camera.internal" not in str(arguments)
    assert create.call_args.kwargs["stderr"] == asyncio.subprocess.DEVNULL
    process = FakeProcess(AsyncMock(side_effect=[jpeg(), b""]), code=1)
    create.return_value = process
    with pytest.raises(ReferenceUnavailableError):
        await capture_jpeg(reference_settings, camera_id)


async def test_ffmpeg_capture_timeout_kills_and_reaps_process(reference_settings, monkeypatch):
    async def blocked_read(count):
        await asyncio.Event().wait()

    process = FakeProcess(blocked_read)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    with pytest.raises(ReferenceUnavailableError):
        await capture_jpeg(reference_settings, uuid4())
    assert process.killed
    assert process.returncode == 0


async def test_ffmpeg_capture_cancellation_kills_and_reaps_process(reference_settings, monkeypatch):
    started = asyncio.Event()

    async def blocked_read(count):
        started.set()
        await asyncio.Event().wait()

    process = FakeProcess(blocked_read)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    task = asyncio.create_task(capture_jpeg(reference_settings, uuid4()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.killed
    assert process.returncode == 0


async def test_ffmpeg_capture_rejects_oversized_output(reference_settings, monkeypatch):
    process = FakeProcess(AsyncMock(return_value=b"x" * 4097))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    with pytest.raises(ReferenceUnavailableError):
        await capture_jpeg(reference_settings, uuid4())
    assert process.killed


async def test_worker_rechecks_source_revision_after_capture(
    reference_settings, camera, monkeypatch
):
    original_revision = source_revision(camera)
    session = SimpleNamespace(
        get=AsyncMock(return_value=camera), scalar=AsyncMock(return_value=camera)
    )

    @asynccontextmanager
    async def transaction():
        yield session

    async def change_during_capture(settings, camera_id):
        camera.stream_url = "rtsp://changed.internal/live"
        return jpeg()

    monkeypatch.setattr("workers.reference_capture.main.capture_jpeg", change_during_capture)
    store = ReferenceStore(reference_settings)
    result = await process_job(
        ReferenceCaptureJob(
            id=uuid4(),
            camera_id=camera.id,
            source_revision=original_revision,
            configuration_revision=camera_revision(camera),
            requested_at=datetime.now(UTC),
        ),
        reference_settings,
        SimpleNamespace(transaction=transaction),
        store,
        acknowledged_redis(camera),
    )
    assert result == {"status": "conflict"}
    assert store.read(camera.id, original_revision) is None


@pytest.mark.parametrize("has_shelf,has_interaction", [(True, False), (False, True)])
async def test_worker_cannot_replace_reference_under_active_geometry(
    reference_settings, camera, monkeypatch, has_shelf, has_interaction
):
    session = SimpleNamespace(
        get=AsyncMock(return_value=camera),
        scalar=AsyncMock(
            side_effect=[
                camera,
                uuid4() if has_shelf else None,
                uuid4() if has_interaction else None,
            ]
        ),
    )

    @asynccontextmanager
    async def transaction():
        yield session

    monkeypatch.setattr(
        "workers.reference_capture.main.capture_jpeg", AsyncMock(return_value=jpeg())
    )
    store = ReferenceStore(reference_settings)
    first = store.publish(camera.id, source_revision(camera), jpeg())
    result = await process_job(
        ReferenceCaptureJob(
            id=uuid4(),
            camera_id=camera.id,
            source_revision=source_revision(camera),
            configuration_revision=camera_revision(camera),
            requested_at=datetime.now(UTC),
            refresh=True,
        ),
        reference_settings,
        SimpleNamespace(transaction=transaction),
        store,
        acknowledged_redis(camera),
    )
    assert result == {"status": "conflict"}
    assert store.read(camera.id, source_revision(camera)) == first


async def test_worker_publishes_successful_capture(reference_settings, camera, monkeypatch):
    session = SimpleNamespace(
        get=AsyncMock(return_value=camera), scalar=AsyncMock(side_effect=[camera, None, None])
    )

    @asynccontextmanager
    async def transaction():
        yield session

    monkeypatch.setattr(
        "workers.reference_capture.main.capture_jpeg", AsyncMock(return_value=jpeg())
    )
    store = ReferenceStore(reference_settings)
    result = await process_job(
        ReferenceCaptureJob(
            id=uuid4(),
            camera_id=camera.id,
            source_revision=source_revision(camera),
            configuration_revision=camera_revision(camera),
            requested_at=datetime.now(UTC),
        ),
        reference_settings,
        SimpleNamespace(transaction=transaction),
        store,
        acknowledged_redis(camera),
    )
    assert result["status"] == "ready"
    assert str(store.read(camera.id, source_revision(camera)).revision) == result["revision"]


async def test_worker_drops_expired_jobs_before_connecting(reference_settings):
    database = SimpleNamespace(transaction=AsyncMock())
    result = await process_job(
        ReferenceCaptureJob(
            id=uuid4(),
            camera_id=uuid4(),
            source_revision=uuid4(),
            configuration_revision=uuid4(),
            requested_at=datetime.now(UTC) - timedelta(minutes=2),
        ),
        reference_settings,
        database,
        ReferenceStore(reference_settings),
        fake_redis(),
    )
    assert result == {"status": "unavailable"}
    database.transaction.assert_not_called()


@pytest.mark.parametrize("acknowledgement", [None, "{}", "null", "invalid", "older", "pending"])
async def test_worker_rejects_missing_pending_or_stale_proxy_acknowledgement(
    reference_settings, camera, monkeypatch, acknowledgement
):
    session = SimpleNamespace(get=AsyncMock(return_value=camera))

    @asynccontextmanager
    async def transaction():
        yield session

    if acknowledgement in ("older", "pending"):
        acknowledgement = json.dumps(
            {
                "status": "pending" if acknowledgement == "pending" else "applied",
                "camera_revision": (
                    str(uuid4()) if acknowledgement == "older" else str(camera_revision(camera))
                ),
                "source_revision": str(source_revision(camera)),
            }
        )
    capture = AsyncMock(return_value=jpeg())
    monkeypatch.setattr("workers.reference_capture.main.capture_jpeg", capture)
    store = ReferenceStore(reference_settings)
    result = await process_job(
        ReferenceCaptureJob(
            id=uuid4(),
            camera_id=camera.id,
            source_revision=source_revision(camera),
            configuration_revision=camera_revision(camera),
            requested_at=datetime.now(UTC),
        ),
        reference_settings,
        SimpleNamespace(transaction=transaction),
        store,
        SimpleNamespace(get_state=AsyncMock(return_value=acknowledgement)),
    )
    assert result == {"status": "unavailable"}
    capture.assert_not_awaited()
    assert store.read(camera.id, source_revision(camera)) is None


async def test_worker_retains_row_lock_until_cancelled_publication_finishes(
    reference_settings, camera, monkeypatch
):
    session = SimpleNamespace(
        get=AsyncMock(return_value=camera), scalar=AsyncMock(side_effect=[camera, None, None])
    )
    locked = False
    entered = threading.Event()
    release = threading.Event()
    transaction_count = 0

    @asynccontextmanager
    async def transaction():
        nonlocal transaction_count, locked
        transaction_count += 1
        if transaction_count == 2:
            locked = True
        try:
            yield session
        finally:
            locked = False

    def publish(camera_id, source, content):
        entered.set()
        assert release.wait(timeout=5)
        assert locked
        return SimpleNamespace(revision=uuid4())

    monkeypatch.setattr(
        "workers.reference_capture.main.capture_jpeg", AsyncMock(return_value=jpeg())
    )
    task = asyncio.create_task(
        process_job(
            ReferenceCaptureJob(
                id=uuid4(),
                camera_id=camera.id,
                source_revision=source_revision(camera),
                configuration_revision=camera_revision(camera),
                requested_at=datetime.now(UTC),
            ),
            reference_settings,
            SimpleNamespace(transaction=transaction),
            SimpleNamespace(publish=publish),
            acknowledged_redis(camera),
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert locked
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not locked
