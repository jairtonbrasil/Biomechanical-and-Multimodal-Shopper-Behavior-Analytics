"""Bounded RTSP decoding with a single latest-frame slot and explicit source epochs."""

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from typing import Any
from urllib.parse import quote
from uuid import UUID, uuid4

from app.core.config import Settings

logger = logging.getLogger(__name__)


class CaptureUnavailableError(RuntimeError):
    """The source cannot produce a bounded frame within its capture policy."""


@dataclass(frozen=True, slots=True)
class RawFrame:
    image: Any
    frame_seq: int
    source_epoch: UUID
    observed_at: datetime
    received_monotonic: float
    source_fps: float | None
    timestamp_source: str = "CAPTURE_CLOCK"


def fit_dimensions(
    width: int, height: int, maximum_width: int, maximum_height: int
) -> tuple[int, int]:
    if (
        not 1 <= width <= 8192
        or not 1 <= height <= 8192
        or width * height > 33554432
        or not 2 <= maximum_width <= 1920
        or not 2 <= maximum_height <= 1920
    ):
        raise CaptureUnavailableError("Source dimensions exceed the capture policy")
    scale = min(1.0, maximum_width / width, maximum_height / height)
    return max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)


def decode_frame(content: bytes, width: int, height: int) -> Any:
    import numpy as np

    return np.frombuffer(content, dtype=np.uint8).reshape((height, width, 3))


async def stop_process(process: Any) -> None:
    if process is None:
        return
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass

    async def drain_stdout() -> None:
        stream = getattr(process, "stdout", None)
        if stream is not None and hasattr(stream, "read"):
            while await stream.read(65536):
                pass

    # Reaping can wait forever when a killed decoder has a paused, unread stdout pipe.
    async with asyncio.timeout(5):
        await asyncio.gather(drain_stdout(), process.wait())


class LatestFrameCapture:
    def __init__(self, settings: Settings, camera_id: UUID) -> None:
        self.settings, self.camera_id = settings, camera_id
        self.counters = {
            "frames_received": 0,
            "sampling_skips": 0,
            "overload_drops": 0,
            "reconnects": 0,
        }
        self._queue: asyncio.Queue[RawFrame | CaptureUnavailableError] = asyncio.Queue(maxsize=1)
        self._task: asyncio.Task | None = None
        self._process: Any = None
        self._terminal: CaptureUnavailableError | None = None
        self._closed = False

    @property
    def timestamp_source(self) -> str:
        return "CAPTURE_CLOCK"

    def _source(self) -> str:
        secret = quote(self.settings.mediamtx_read_password.get_secret_value(), safe="")
        return (
            f"rtsp://media-reader:{secret}@{self.settings.mediamtx_rtsp_host}:"
            f"{self.settings.mediamtx_rtsp_port}/cameras/{self.camera_id}"
        )

    async def _probe(self) -> tuple[int, int, float | None]:
        process = None
        try:
            async with asyncio.timeout(self.settings.vision_capture_timeout_seconds):
                process = await asyncio.create_subprocess_exec(
                    "ffprobe",
                    "-v",
                    "error",
                    "-rtsp_transport",
                    "tcp",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate",
                    "-of",
                    "json",
                    self._source(),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                content = bytearray()
                while True:
                    chunk = await process.stdout.read(4097 - len(content))
                    if not chunk:
                        break
                    content.extend(chunk)
                    if len(content) > 4096:
                        raise CaptureUnavailableError("Source metadata is unavailable")
                if await process.wait() != 0:
                    raise CaptureUnavailableError("Source metadata is unavailable")
                document = json.loads(content)
                streams = document.get("streams")
                if not isinstance(streams, list) or len(streams) != 1:
                    raise CaptureUnavailableError("Source requires one selected video stream")
                stream = streams[0]
                if type(stream.get("width")) is not int or type(stream.get("height")) is not int:
                    raise CaptureUnavailableError("Source dimensions are unavailable")
                width, height = fit_dimensions(
                    stream["width"],
                    stream["height"],
                    self.settings.vision_frame_width,
                    self.settings.vision_frame_height,
                )
                try:
                    fps = float(Fraction(stream.get("avg_frame_rate", "0/1")))
                    fps = fps if math.isfinite(fps) and 0 < fps <= 240 else None
                except (ValueError, ZeroDivisionError, TypeError):
                    fps = None
                return width, height, fps
        except asyncio.CancelledError:
            raise
        except Exception:
            raise CaptureUnavailableError("Source metadata is unavailable") from None
        finally:
            await stop_process(process)

    def _offer(self, item: RawFrame | CaptureUnavailableError) -> None:
        if self._queue.full():
            previous = self._queue.get_nowait()
            if isinstance(previous, RawFrame):
                self.counters["overload_drops"] += 1
        self._queue.put_nowait(item)

    def _discard_pending(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()

    async def _decode(self) -> None:
        width, height, source_fps = await self._probe()
        epoch, frame_seq = uuid4(), 0
        frame_bytes = width * height * 3
        self._process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-i",
            self._source(),
            "-an",
            "-vf",
            f"scale={width}:{height}",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=65536,
        )
        logger.info(
            "capture_connected",
            extra={
                "camera_id": str(self.camera_id),
                "source_epoch": str(epoch),
                "width": width,
                "height": height,
                "source_fps": source_fps,
                "timestamp_source": self.timestamp_source,
            },
        )
        next_sample = 0.0
        try:
            while True:
                async with asyncio.timeout(self.settings.vision_capture_timeout_seconds):
                    content = await self._process.stdout.readexactly(frame_bytes)
                received = time.monotonic()
                observed_at = datetime.now(UTC)
                self.counters["frames_received"] += 1
                current_seq, frame_seq = frame_seq, frame_seq + 1
                if received < next_sample:
                    self.counters["sampling_skips"] += 1
                    continue
                next_sample = received + 1 / self.settings.sampling_fps
                self._offer(
                    RawFrame(
                        decode_frame(content, width, height),
                        current_seq,
                        epoch,
                        observed_at,
                        received,
                        source_fps,
                    )
                )
        finally:
            await stop_process(self._process)
            self._process = None

    async def _run(self) -> None:
        failures = 0
        try:
            while not self._closed:
                received_before = self.counters["frames_received"]
                try:
                    await self._decode()
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self._discard_pending()
                    failures = (
                        1 if self.counters["frames_received"] > received_before else failures + 1
                    )
                    if failures >= self.settings.vision_reconnect_attempts:
                        self._terminal = CaptureUnavailableError(
                            "Source reconnect attempts exhausted"
                        )
                        self._offer(self._terminal)
                        return
                    self.counters["reconnects"] += 1
                    logger.warning(
                        "capture_reconnecting",
                        extra={
                            "camera_id": str(self.camera_id),
                            "attempt": failures,
                            "error_type": type(error).__name__,
                        },
                    )
                    await asyncio.sleep(
                        min(2 ** (failures - 1), self.settings.vision_reconnect_max_seconds)
                    )
        finally:
            await stop_process(self._process)
            self._process = None

    async def start(self) -> None:
        if self._closed:
            raise CaptureUnavailableError("Capture is closed")
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def next_frame(self, timeout: float | None = None) -> RawFrame:  # noqa: ASYNC109
        if self._closed:
            raise CaptureUnavailableError("Capture is closed")
        if self._terminal is not None:
            raise self._terminal
        await self.start()
        try:
            async with asyncio.timeout(timeout):
                frame = await self._queue.get()
        except TimeoutError:
            raise CaptureUnavailableError("Frame wait deadline exceeded") from None
        if isinstance(frame, CaptureUnavailableError):
            raise frame
        return frame

    async def close(self) -> None:
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._discard_pending()
        self._terminal = CaptureUnavailableError("Capture is closed")
        self._offer(self._terminal)
