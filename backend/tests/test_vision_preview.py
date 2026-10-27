import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.factories import example_envelope
from workers.vision_streamer import preview
from workers.vision_streamer.preview import DiagnosticPreview, PreviewUnavailableError

NOW = datetime(2026, 10, 27, tzinfo=UTC)


class FakeInput:
    def __init__(self, *, blocked=False):
        self.frames = []
        self.blocked = blocked
        self.closed = False

    def write(self, content):
        self.frames.append(content)

    async def drain(self):
        if self.blocked:
            await asyncio.Event().wait()

    def close(self):
        self.closed = True

    async def wait_closed(self):
        return None


class FakeProcess:
    def __init__(self, *, blocked=False):
        self.stdin = FakeInput(blocked=blocked)
        self.returncode = None
        self.killed = False
        self.waited = False

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        self.waited = True
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


@pytest.fixture
def encoder(monkeypatch):
    process = FakeProcess()
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(preview.asyncio, "create_subprocess_exec", spawn)

    async def inline(function, *arguments, **keywords):
        return function(*arguments, **keywords)

    monkeypatch.setattr(preview.asyncio, "to_thread", inline)
    monkeypatch.setattr(preview, "_render_frame", lambda frame, *arguments: frame)
    return process, spawn


async def test_preview_uses_bounded_h264_encoder_and_reaps_successfully(tmp_path, encoder):
    process, spawn = encoder
    video = DiagnosticPreview(tmp_path / "output" / "annotated.mp4", 10, 2, 100, 80)
    await video.start()
    await video.start()
    command = spawn.call_args.args
    assert command[0] == "ffmpeg"
    assert command[command.index("-pixel_format") + 1] == "bgr24"
    assert command[command.index("-video_size") + 1] == "100x80"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert spawn.call_args.kwargs["stderr"] == asyncio.subprocess.DEVNULL
    assert spawn.await_count == 1
    await video.write(b"frame", [], 1, 12, 2, NOW)
    await video.close()
    await video.close()
    assert process.stdin.frames == [b"frame"]
    assert process.stdin.closed and process.waited and not process.killed


async def test_preview_preserves_elapsed_time_by_repeating_prior_frame(tmp_path, encoder):
    process, _ = encoder
    video = DiagnosticPreview(tmp_path / "annotated.mp4", 2, 2, 100, 80)
    await video.write(b"first", [], 1, 10, 1, NOW)
    await video.write(b"next", [], 4, 10, 1, NOW + timedelta(seconds=1.5))
    assert process.stdin.frames == [b"first", b"first", b"first", b"next"]
    assert video.frames_written == 4 and video.complete
    await video.write(b"ignored", [], 9, 10, 1, NOW + timedelta(seconds=10))
    assert len(process.stdin.frames) == 4
    await video.close()


async def test_preview_caps_long_gaps_at_configured_duration(tmp_path, encoder):
    process, _ = encoder
    video = DiagnosticPreview(tmp_path / "annotated.mp4", 2, 2, 100, 80)
    await video.write(b"first", [], 1, 10, 1, NOW)
    await video.write(b"late", [], 100, 10, 1, NOW + timedelta(seconds=100))
    assert process.stdin.frames == [b"first"] * 4
    assert video.complete
    await video.close()


async def test_preview_does_not_duplicate_high_rate_or_backwards_samples(tmp_path, encoder):
    process, _ = encoder
    video = DiagnosticPreview(tmp_path / "annotated.mp4", 2, 2, 100, 80)
    await video.write(b"first", [], 1, 10, 1, NOW)
    await video.write(b"latest", [], 2, 10, 1, NOW + timedelta(seconds=0.1))
    await video.write(b"backwards", [], 0, 10, 1, NOW - timedelta(seconds=1))
    await video.write(b"next", [], 3, 10, 1, NOW + timedelta(seconds=1))
    assert process.stdin.frames == [b"first", b"latest", b"next"]
    await video.close()


async def test_preview_slow_encoder_is_killed_and_reaped(tmp_path, encoder):
    process, _ = encoder
    process.stdin.blocked = True
    video = DiagnosticPreview(
        tmp_path / "annotated.mp4", 10, 2, 100, 80, write_timeout_seconds=0.01
    )
    with pytest.raises(PreviewUnavailableError, match="encoding failed"):
        await video.write(b"frame", [], 1, 10, 1, NOW)
    assert process.killed and process.waited


async def test_preview_cancellation_kills_and_reaps_encoder(tmp_path, encoder):
    process, _ = encoder
    process.stdin.blocked = True
    video = DiagnosticPreview(tmp_path / "annotated.mp4", 10, 2, 100, 80)
    task = asyncio.create_task(video.write(b"frame", [], 1, 10, 1, NOW))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.killed and process.waited


async def test_preview_reports_failed_encoder_without_stderr(tmp_path, encoder):
    process, _ = encoder
    video = DiagnosticPreview(tmp_path / "annotated.mp4", 10, 2, 100, 80)
    await video.start()
    process.returncode = 1
    with pytest.raises(PreviewUnavailableError, match="finalization failed"):
        await video.close()
    assert process.waited


async def test_preview_can_close_without_starting_and_cannot_restart(tmp_path, encoder):
    _, spawn = encoder
    video = DiagnosticPreview(tmp_path / "annotated.mp4", 10, 2, 100, 80)
    await video.close()
    assert not spawn.called
    with pytest.raises(PreviewUnavailableError, match="closed"):
        await video.start()


@pytest.mark.parametrize(
    "changes", [{"fps": 0}, {"fps": float("nan")}, {"max_seconds": 0}, {"width": 0}, {"height": 0}]
)
def test_preview_bounds_are_validated(tmp_path, changes):
    values = {
        "output": tmp_path / "annotated.mp4",
        "fps": 10,
        "max_seconds": 2,
        "width": 100,
        "height": 80,
        **changes,
    }
    with pytest.raises(ValueError):
        DiagnosticPreview(**values)


class FakeFrame:
    shape = (80, 100, 3)
    dtype = "uint8"

    def copy(self):
        return FakeFrame()

    def tobytes(self):
        return b"frame-bytes"


@pytest.fixture
def drawing(monkeypatch):
    import sys

    calls = []

    def record(name):
        def capture(*arguments, **keywords):
            calls.append((name, arguments, keywords))

        return capture

    module = SimpleNamespace(FONT_HERSHEY_SIMPLEX=0, LINE_AA=0)
    for name in ("line", "circle", "arrowedLine", "rectangle", "putText"):
        setattr(module, name, record(name))
    monkeypatch.setitem(sys.modules, "cv2", module)
    return calls


def test_preview_renders_regions_with_zero_people_and_marks_diagnostics(drawing):
    geometry = example_envelope().observation.attributes.geometry
    content = preview._render_frame(FakeFrame(), [], geometry, 12, 3.5, 2.5, 100, 80)
    assert content == b"frame-bytes"
    assert len([call for call in drawing if call[0] == "line"]) == 8
    labels = [call[1][1] for call in drawing if call[0] == "putText"]
    assert any("Frame 12" in label and "3.5 ms" in label for label in labels)
    assert any("not confirmed" in label for label in labels)


def test_preview_labels_candidates_and_ignores_low_confidence_landmarks(drawing):
    from app.schemas.observation import ObservationEnvelope

    payload = example_envelope().model_dump()
    attributes = payload["observation"]["attributes"]
    attributes["pose"]["left_shoulder"] = {"x": 0.4, "y": 0.4, "confidence": 0.9}
    attributes["pose"]["left_elbow"] = {"x": 0.3, "y": 0.5, "confidence": 0.1}
    attributes["kinematics"]["shoulder_width_pixels"] = 100
    attributes["kinematics"]["reach_candidate"] = True
    envelope = ObservationEnvelope.model_validate(payload)
    preview._render_frame(FakeFrame(), [envelope], (), 12, 3.5, 2.5, 100, 80)
    labels = [call[1][1] for call in drawing if call[0] == "putText"]
    assert any("candidate: REACH" in label for label in labels)
    assert not any("PICKUP" in label for label in labels)
    assert len([call for call in drawing if call[0] == "circle"]) == 1


def test_preview_rejects_unexpected_frame_dimensions(drawing):
    with pytest.raises(PreviewUnavailableError, match="shape"):
        preview._render_frame(FakeFrame(), [], (), 1, 1, 1, 200, 80)
