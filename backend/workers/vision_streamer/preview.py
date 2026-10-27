"""Bounded local diagnostic video; image data never enters logs or Kafka payloads."""

import asyncio
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from app.schemas.geometry import GeometrySnapshot
from app.schemas.observation import ObservationEnvelope

SKELETON_EDGES = (
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


class PreviewUnavailableError(RuntimeError):
    pass


def _render_frame(
    frame: Any,
    observations: list[ObservationEnvelope],
    geometry: tuple[GeometrySnapshot, ...],
    frame_seq: int,
    inference_ms: float,
    capture_age_ms: float,
    width: int,
    height: int,
) -> bytes:
    import cv2

    if frame.shape != (height, width, 3) or str(frame.dtype) != "uint8":
        raise PreviewUnavailableError("Diagnostic frame shape or pixel format is invalid")
    canvas = frame.copy()

    def xy(x: float, y: float) -> tuple[int, int]:
        return min(width - 1, max(0, round(x * width))), min(height - 1, max(0, round(y * height)))

    snapshots = observations[0].observation.attributes.geometry if observations else geometry
    for snapshot in snapshots[:32]:
        for polygon, color in (
            (snapshot.product_roi, (255, 190, 40)),
            (snapshot.presence_roi, (80, 220, 100)),
        ):
            points = [xy(x, y) for x, y in polygon.points]
            for index, origin in enumerate(points):
                cv2.line(canvas, origin, points[(index + 1) % len(points)], color, 2)
    for envelope in observations[:64]:
        observation = envelope.observation
        attributes = observation.attributes
        for first, second in SKELETON_EDGES:
            a, b = getattr(attributes.pose, first), getattr(attributes.pose, second)
            if a is not None and b is not None and min(a.confidence, b.confidence) >= 0.3:
                cv2.line(canvas, xy(a.x, a.y), xy(b.x, b.y), (60, 220, 255), 2)
        for point in attributes.pose.model_dump().values():
            if point is not None and point["confidence"] >= 0.3:
                cv2.circle(canvas, xy(point["x"], point["y"]), 3, (30, 80, 255), -1)
        spatial = attributes.spatial
        if spatial.head_origin is not None and spatial.gaze_target is not None:
            cv2.arrowedLine(
                canvas,
                xy(spatial.head_origin.x, spatial.head_origin.y),
                xy(spatial.gaze_target.x, spatial.gaze_target.y),
                (70, 240, 90),
                2,
                tipLength=0.15,
            )
        if attributes.bbox is not None:
            box = attributes.bbox
            origin = xy(box.x1, box.y1)
            cv2.rectangle(canvas, origin, xy(box.x2, box.y2), (255, 180, 40), 2)
        else:
            origin = (8, 75)
        candidates = [
            name
            for name, supported in (
                ("REACH", attributes.kinematics.reach_candidate),
                ("INSPECT", attributes.kinematics.inspect_candidate),
                ("HESITATION", attributes.kinematics.hesitation_candidate),
            )
            if supported is True
        ]
        label = (
            f"Track {observation.local_track_id[:16]} | {attributes.movement.state or 'UNKNOWN'}"
        )
        if candidates:
            label += " | candidate: " + ",".join(candidates)
        if attributes.face.expression_detected is not None:
            label += f" | face: {attributes.face.expression_detected}"
        cv2.putText(
            canvas,
            label,
            (origin[0], max(68, origin[1] - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (0, 0), (width - 1, min(height - 1, 52)), (15, 20, 25), -1)
    cv2.putText(
        canvas,
        f"Frame {frame_seq} | inference {inference_ms:.1f} ms"
        f" | capture age {capture_age_ms:.1f} ms",
        (8, min(height - 1, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Diagnostic candidates: not confirmed actions, emotions or purchases",
        (8, min(height - 1, 42)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 220, 100),
        1,
        cv2.LINE_AA,
    )
    return canvas.tobytes()


class DiagnosticPreview:
    def __init__(
        self,
        output: Path,
        fps: float,
        max_seconds: int,
        width: int,
        height: int,
        *,
        geometry: tuple[GeometrySnapshot, ...] = (),
        write_timeout_seconds: float = 5,
    ) -> None:
        if not math.isfinite(fps) or not 0 < fps <= 60 or not 1 <= max_seconds <= 3600:
            raise ValueError("Diagnostic preview duration and cadence must be bounded")
        if not 1 <= width <= 16384 or not 1 <= height <= 16384 or write_timeout_seconds <= 0:
            raise ValueError("Diagnostic preview dimensions and deadlines must be positive")
        self.output, self.fps, self.max_seconds = output, fps, max_seconds
        self.width, self.height = width, height
        self.geometry = geometry
        self.write_timeout_seconds = write_timeout_seconds
        self.frames_written = 0
        self._maximum_frames = math.ceil(fps * max_seconds)
        self._process: asyncio.subprocess.Process | None = None
        self._started_at: datetime | None = None
        self._last_frame: bytes | None = None
        self._closed = False

    @property
    def complete(self) -> bool:
        return self.frames_written >= self._maximum_frames

    async def start(self) -> None:
        if self._process is not None:
            return
        if self._closed:
            raise PreviewUnavailableError("Diagnostic preview is closed")
        await asyncio.to_thread(self.output.parent.mkdir, parents=True, exist_ok=True)
        try:
            self._process = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
                f"{self.width}x{self.height}",
                "-framerate",
                str(self.fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "23",
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-movflags",
                "+faststart",
                str(self.output),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError:
            raise PreviewUnavailableError("Diagnostic encoder is unavailable") from None

    async def _append(self, content: bytes) -> None:
        if self.complete:
            return
        if (
            self._process is None
            or self._process.stdin is None
            or self._process.returncode is not None
        ):
            raise PreviewUnavailableError("Diagnostic encoder is unavailable")
        self._process.stdin.write(content)
        await self._process.stdin.drain()
        self.frames_written += 1

    async def write(
        self,
        frame: Any,
        observations: list[ObservationEnvelope],
        frame_seq: int,
        inference_ms: float,
        capture_age_ms: float,
        observed_at: datetime,
    ) -> None:
        if self._closed or self.complete:
            return
        if observed_at.utcoffset() is None or frame_seq < 0:
            raise ValueError("Diagnostic frames need aware event time and nonnegative sequence")
        if not all(math.isfinite(value) and value >= 0 for value in (inference_ms, capture_age_ms)):
            raise ValueError("Diagnostic durations must be finite and nonnegative")
        if self._process is None:
            await self.start()
        if self._started_at is None:
            self._started_at = observed_at
        elapsed = (observed_at - self._started_at).total_seconds()
        if elapsed < 0:
            return
        target_index = min(self._maximum_frames, round(elapsed * self.fps))
        try:
            async with asyncio.timeout(self.write_timeout_seconds):
                rendered = await asyncio.to_thread(
                    _render_frame,
                    frame,
                    observations,
                    self.geometry,
                    frame_seq,
                    inference_ms,
                    capture_age_ms,
                    self.width,
                    self.height,
                )
                if self._last_frame is not None:
                    while self.frames_written < target_index:
                        await self._append(self._last_frame)
                if not self.complete and target_index >= self.frames_written:
                    await self._append(rendered)
                self._last_frame = rendered
        except asyncio.CancelledError:
            await self._terminate()
            raise
        except Exception:
            await self._terminate()
            raise PreviewUnavailableError("Diagnostic preview encoding failed") from None

    async def _terminate(self) -> None:
        process = self._process
        self._closed = True
        if process is not None and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is None:
            return
        try:
            async with asyncio.timeout(self.write_timeout_seconds):
                if process.stdin is not None:
                    process.stdin.close()
                    await process.stdin.wait_closed()
                result = await process.wait()
                if result != 0:
                    raise PreviewUnavailableError("Diagnostic encoder did not finish successfully")
        except asyncio.CancelledError:
            await self._terminate()
            raise
        except Exception:
            await self._terminate()
            raise PreviewUnavailableError("Diagnostic preview finalization failed") from None
