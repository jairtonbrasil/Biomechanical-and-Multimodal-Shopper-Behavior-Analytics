"""Bounded, revision-fenced reference images outside the API process."""

import asyncio
import fcntl
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import ValidationError

from app.core.config import Settings
from app.core.redis_client import RedisClient
from app.domain.configuration import camera_revision
from app.domain.configuration import source_revision as camera_source_revision
from app.models.camera import Camera
from app.repositories.errors import VersionConflictError
from app.schemas.references import ReferenceCaptureJob, ReferenceFrame, ReferenceMetadata


class ReferenceUnavailableError(RuntimeError):
    """The reference cannot be served without violating its capture contract."""


def jpeg_dimensions(content: bytes) -> tuple[int, int]:
    """Read JPEG frame dimensions without loading an image-processing runtime."""
    if not content.startswith(b"\xff\xd8") or not content.endswith(b"\xff\xd9"):
        raise ReferenceUnavailableError("Reference capture did not return a JPEG")
    offset = 2
    while offset + 3 < len(content):
        if content[offset] != 0xFF:
            break
        while offset < len(content) and content[offset] == 0xFF:
            offset += 1
        if offset >= len(content):
            break
        marker = content[offset]
        offset += 1
        if marker in (0xD8, 0xD9, 0xDA):
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(content):
            break
        length = int.from_bytes(content[offset : offset + 2], "big")
        if length < 2 or offset + length > len(content):
            break
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if length < 8:
                break
            height = int.from_bytes(content[offset + 3 : offset + 5], "big")
            width = int.from_bytes(content[offset + 5 : offset + 7], "big")
            if width > 0 and height > 0:
                return width, height
            break
        offset += length
    raise ReferenceUnavailableError("Reference JPEG dimensions are unavailable")


class ReferenceStore:
    def __init__(self, settings: Settings) -> None:
        self.directory = settings.reference_directory
        self.max_bytes = settings.reference_max_bytes
        self.storage_limit = settings.reference_storage_limit_bytes
        self.max_dimension = settings.reference_max_width

    @staticmethod
    def _read(path: Path, maximum: int) -> bytes:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as source:
            content = source.read(maximum + 1)
        if len(content) > maximum:
            raise ReferenceUnavailableError("Reference exceeds its configured size limit")
        return content

    def read(
        self, camera_id: UUID, source_revision: UUID, revision: UUID | None = None
    ) -> ReferenceFrame | None:
        try:
            directory = self.directory / str(camera_id)
            if directory.is_symlink():
                return None
            name = "current.json" if revision is None else f"{revision}.json"
            metadata = ReferenceMetadata.model_validate_json(self._read(directory / name, 4096))
            if (
                metadata.camera_id != camera_id
                or metadata.source_revision != source_revision
                or (revision is not None and metadata.revision != revision)
            ):
                return None
            content = self._read(directory / f"{metadata.revision}.jpg", self.max_bytes)
            if (
                len(content) != metadata.content_length
                or hashlib.sha256(content).hexdigest() != metadata.sha256
                or jpeg_dimensions(content) != (metadata.width, metadata.height)
                or max(metadata.width, metadata.height) > self.max_dimension
            ):
                return None
            return ReferenceFrame(
                content=content,
                revision=metadata.revision,
                width=metadata.width,
                height=metadata.height,
                captured_at=metadata.captured_at,
                source_revision=metadata.source_revision,
            )
        except (OSError, ValidationError, ValueError, ReferenceUnavailableError):
            return None

    @staticmethod
    def _atomic_write(directory: Path, destination: Path, content: bytes) -> None:
        descriptor, temporary = tempfile.mkstemp(prefix=".reference-", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def publish(self, camera_id: UUID, source_revision: UUID, content: bytes) -> ReferenceFrame:
        if len(content) > self.max_bytes:
            raise ReferenceUnavailableError("Reference exceeds its configured size limit")
        width, height = jpeg_dimensions(content)
        if max(width, height) > self.max_dimension:
            raise ReferenceUnavailableError("Reference dimensions exceed their configured limit")
        metadata = ReferenceMetadata(
            camera_id=camera_id,
            revision=uuid4(),
            source_revision=source_revision,
            captured_at=datetime.now(UTC),
            width=width,
            height=height,
            content_length=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        encoded = metadata.model_dump_json().encode()
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            if self.directory.is_symlink():
                raise ReferenceUnavailableError("Reference storage is unavailable")
            lock = os.open(
                self.directory / ".storage.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(lock, "wb") as guard:
                # Shared by API and capture processes; the quota check and publication are atomic.
                fcntl.flock(guard.fileno(), fcntl.LOCK_EX)
                directory = self.directory / str(camera_id)
                directory.mkdir(mode=0o700, exist_ok=True)
                if directory.is_symlink():
                    raise ReferenceUnavailableError("Reference storage is unavailable")
                used = sum(
                    entry.stat().st_size
                    for entry in self.directory.rglob("*")
                    if entry.is_file() and not entry.is_symlink()
                )
                if used + len(content) + 2 * len(encoded) > self.storage_limit:
                    raise ReferenceUnavailableError("Reference storage quota is exhausted")
                self._atomic_write(directory, directory / f"{metadata.revision}.jpg", content)
                self._atomic_write(directory, directory / f"{metadata.revision}.json", encoded)
                self._atomic_write(directory, directory / "current.json", encoded)
                directory_descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            return ReferenceFrame(
                content, metadata.revision, width, height, metadata.captured_at, source_revision
            )
        except OSError:
            raise ReferenceUnavailableError("Reference storage is unavailable") from None


ENQUEUE_CAPTURE = """
local pending = redis.call('GET', KEYS[1])
if pending then return pending end
if redis.call('LLEN', KEYS[2]) >= tonumber(ARGV[3]) then return '' end
redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[4])
redis.call('RPUSH', KEYS[2], ARGV[2])
return ARGV[1]
"""


def reference_pending_scope(job: ReferenceCaptureJob) -> str:
    return f"reference:pending:{job.camera_id}:{job.source_revision}:{job.configuration_revision}"


class ReferenceManager:
    def __init__(self, settings: Settings, redis: RedisClient) -> None:
        self.settings = settings
        self.redis = redis
        self.store = ReferenceStore(settings)

    async def current(self, camera: Camera) -> ReferenceFrame | None:
        if not camera.is_active:
            return None
        return await asyncio.to_thread(self.store.read, camera.id, camera_source_revision(camera))

    async def get(self, camera: Camera, *, refresh: bool = False) -> ReferenceFrame:
        if not camera.is_active:
            raise ReferenceUnavailableError("Inactive cameras do not provide reference images")
        if not refresh:
            existing = await self.current(camera)
            if existing is not None:
                return existing
        source_revision = camera_source_revision(camera)
        job = ReferenceCaptureJob(
            id=uuid4(),
            camera_id=camera.id,
            configuration_revision=camera_revision(camera),
            source_revision=source_revision,
            requested_at=datetime.now(UTC),
            refresh=refresh,
        )
        try:
            job_id = await self.redis.client.eval(
                ENQUEUE_CAPTURE,
                2,
                self.redis.key("channel", reference_pending_scope(job)),
                self.redis.key("channel", "reference:jobs"),
                str(job.id),
                job.model_dump_json(),
                self.settings.reference_queue_limit,
                self.settings.reference_job_ttl_seconds,
            )
            if not job_id:
                raise ReferenceUnavailableError("Reference capture queue is full")
            job_id = UUID(job_id)
            async with asyncio.timeout(self.settings.reference_timeout_seconds + 2):
                while True:
                    raw = await self.redis.client.get(
                        self.redis.key("channel", f"reference:result:{job_id}")
                    )
                    if raw is not None:
                        result = json.loads(raw)
                        if result.get("status") == "conflict":
                            raise VersionConflictError("Camera changed during reference capture")
                        if result.get("status") != "ready":
                            raise ReferenceUnavailableError("Reference capture is unavailable")
                        frame = await asyncio.to_thread(
                            self.store.read, camera.id, source_revision, UUID(result["revision"])
                        )
                        if frame is None:
                            raise ReferenceUnavailableError("Captured reference is unavailable")
                        return frame
                    await asyncio.sleep(0.1)
        except (ReferenceUnavailableError, VersionConflictError):
            raise
        except Exception:
            raise ReferenceUnavailableError("Reference capture is unavailable") from None


async def capture_jpeg(settings: Settings, camera_id: UUID) -> bytes:
    """Capture one bounded frame from the authenticated internal proxy only."""
    from urllib.parse import quote

    password = quote(settings.mediamtx_read_password.get_secret_value(), safe="")
    proxy = (
        f"rtsp://media-reader:{password}@{settings.mediamtx_rtsp_host}:"
        f"{settings.mediamtx_rtsp_port}/cameras/{camera_id}"
    )
    limit = settings.reference_max_width
    process = None
    try:
        async with asyncio.timeout(settings.reference_timeout_seconds):
            process = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-rtsp_transport",
                "tcp",
                "-i",
                proxy,
                "-frames:v",
                "1",
                "-an",
                "-vf",
                f"scale=w='min({limit},iw)':h='min({limit},ih)':force_original_aspect_ratio=decrease",
                "-threads",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "pipe:1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            content = bytearray()
            while True:
                chunk = await process.stdout.read(min(65536, settings.reference_max_bytes + 1))
                if not chunk:
                    break
                content.extend(chunk)
                if len(content) > settings.reference_max_bytes:
                    raise ReferenceUnavailableError("Reference exceeds its configured size limit")
            if await process.wait() != 0:
                raise ReferenceUnavailableError("Reference capture failed")
            jpeg_dimensions(bytes(content))
            return bytes(content)
    except asyncio.CancelledError:
        raise
    except Exception:
        raise ReferenceUnavailableError("Reference capture failed") from None
    finally:
        if process is not None and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
