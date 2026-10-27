import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import SecretStr

from workers.vision_streamer import capture
from workers.vision_streamer.capture import CaptureUnavailableError, LatestFrameCapture, RawFrame


def configuration(**changes):
    return SimpleNamespace(
        mediamtx_read_password=SecretStr("private-media-password"),
        mediamtx_rtsp_host="mediamtx",
        mediamtx_rtsp_port=8554,
        vision_frame_width=640,
        vision_frame_height=360,
        vision_capture_timeout_seconds=0.1,
        vision_reconnect_attempts=1,
        vision_reconnect_max_seconds=0.001,
        sampling_fps=10,
        **changes,
    )


class ProbeStream:
    def __init__(self, content):
        self.content = content

    async def read(self, maximum):
        result, self.content = self.content[:maximum], self.content[maximum:]
        return result


class VideoStream:
    def __init__(self):
        self.content = asyncio.Queue()

    async def readexactly(self, size):
        content = await self.content.get()
        if isinstance(content, Exception):
            raise content
        assert len(content) == size
        return content


class Process:
    def __init__(self, stdout, *, exit_code=None):
        self.stdout = stdout
        self.returncode = exit_code
        self.killed = False
        self.waited = False

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        self.waited = True
        if isinstance(self.stdout, ProbeStream) and self.returncode is None:
            self.returncode = 0
        return self.returncode


def subprocesses(monkeypatch, videos, *, metadata=None):
    commands = []
    probes = []
    document = (
        metadata
        if metadata is not None
        else {"streams": [{"width": 2, "height": 2, "avg_frame_rate": "30/1"}]}
    )

    async def spawn(*arguments, **kwargs):
        commands.append(arguments)
        assert kwargs["stderr"] == asyncio.subprocess.DEVNULL
        if arguments[0] == "ffprobe":
            probe = Process(ProbeStream(json.dumps(document).encode()))
            probes.append(probe)
            return probe
        return videos.pop(0)

    monkeypatch.setattr(capture.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(capture, "decode_frame", lambda content, width, height: content)
    return commands, probes


@pytest.mark.parametrize(
    "source,bounds,expected",
    [
        ((1920, 1080), (1280, 720), (1280, 720)),
        ((1080, 1920), (1280, 720), (404, 720)),
        ((640, 360), (1280, 720), (640, 360)),
    ],
)
def test_capture_dimensions_preserve_aspect_without_upscaling(source, bounds, expected):
    assert capture.fit_dimensions(*source, *bounds) == expected


@pytest.mark.parametrize("dimensions", [(0, 10), (8193, 2), (8192, 8192), (-1, 1)])
def test_capture_rejects_unbounded_source_dimensions(dimensions):
    with pytest.raises(CaptureUnavailableError):
        capture.fit_dimensions(*dimensions, 1280, 720)


async def test_latest_slot_discards_overload_instead_of_accumulating_frames():
    reader = LatestFrameCapture(configuration(), uuid4())
    epoch = uuid4()
    for sequence in range(100):
        reader._offer(RawFrame(b"frame", sequence, epoch, datetime.now(UTC), 1.0, 30.0))
    assert reader._queue.qsize() == 1
    assert reader.counters["overload_drops"] == 99
    assert reader._queue.get_nowait().frame_seq == 99
    await reader.close()


async def test_capture_sampling_reports_skips_separately_from_overload(monkeypatch):
    stream = VideoStream()
    process = Process(stream)
    commands, probes = subprocesses(monkeypatch, [process])
    ticks = iter([1.0, 1.01, 1.2])
    monkeypatch.setattr(capture, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    reader = LatestFrameCapture(configuration(), uuid4())
    for value in (b"a" * 12, b"b" * 12, b"c" * 12):
        stream.content.put_nowait(value)
    await reader.start()
    for _ in range(50):
        if reader.counters["frames_received"] == 3:
            break
        await asyncio.sleep(0)
    frame = await reader.next_frame(timeout=1)
    assert frame.frame_seq == 2 and frame.image == b"c" * 12
    assert frame.timestamp_source == "CAPTURE_CLOCK" and frame.observed_at.utcoffset() is not None
    assert frame.source_fps == 30.0
    assert reader.counters["sampling_skips"] == 1
    assert reader.counters["overload_drops"] == 1
    assert str(reader.camera_id) in commands[0][-1]
    assert commands[1][0] == "ffmpeg"
    await reader.close()
    assert process.killed and process.waited and probes[0].waited


async def test_reconnect_changes_epoch_and_resets_frame_sequence(monkeypatch):
    first, second = VideoStream(), VideoStream()
    first_process, second_process = Process(first), Process(second)
    subprocesses(monkeypatch, [first_process, second_process])
    settings = configuration()
    settings.vision_reconnect_attempts = 2
    reader = LatestFrameCapture(settings, uuid4())
    first.content.put_nowait(b"a" * 12)
    one = await reader.next_frame(timeout=1)
    first.content.put_nowait(asyncio.IncompleteReadError(b"", 12))
    second.content.put_nowait(b"b" * 12)
    two = await reader.next_frame(timeout=1)
    assert one.source_epoch != two.source_epoch
    assert one.frame_seq == two.frame_seq == 0
    assert reader.counters["reconnects"] == 1
    assert first_process.killed and first_process.waited
    await reader.close()
    assert second_process.killed and second_process.waited


async def test_capture_timeout_reaps_process_and_fails_with_bounded_retries(monkeypatch, caplog):
    process = Process(VideoStream())
    subprocesses(monkeypatch, [process])
    reader = LatestFrameCapture(configuration(), uuid4())
    with pytest.raises(CaptureUnavailableError, match="reconnect attempts exhausted"):
        await reader.next_frame(timeout=1)
    assert process.killed and process.waited
    assert "private-media-password" not in caplog.text
    with pytest.raises(CaptureUnavailableError):
        await reader.next_frame(timeout=1)
    await reader.close()


async def test_capture_cancellation_kills_child_and_closed_capture_cannot_restart(monkeypatch):
    process = Process(VideoStream())
    subprocesses(monkeypatch, [process])
    reader = LatestFrameCapture(configuration(), uuid4())
    await reader.start()
    for _ in range(20):
        if reader._process is not None:
            break
        await asyncio.sleep(0)
    await reader.close()
    assert process.killed and process.waited
    with pytest.raises(CaptureUnavailableError, match="closed"):
        await reader.start()


@pytest.mark.parametrize(
    "metadata", [{}, {"streams": []}, {"streams": [{"width": "huge", "height": 2}]}]
)
async def test_invalid_probe_metadata_never_starts_decoder(monkeypatch, metadata):
    commands, probes = subprocesses(monkeypatch, [], metadata=metadata)
    reader = LatestFrameCapture(configuration(), uuid4())
    with pytest.raises(CaptureUnavailableError):
        await reader.next_frame(timeout=1)
    assert len(commands) == 1 and commands[0][0] == "ffprobe"
    assert probes[0].waited
    await reader.close()


async def test_wait_timeout_does_not_cancel_background_capture(monkeypatch):
    process = Process(VideoStream())
    subprocesses(monkeypatch, [process])
    reader = LatestFrameCapture(configuration(), uuid4())
    with pytest.raises(CaptureUnavailableError, match="wait deadline"):
        await reader.next_frame(timeout=0.001)
    assert not reader._task.done()
    await reader.close()


async def test_close_wakes_a_reader_waiting_for_the_next_frame(monkeypatch):
    process = Process(VideoStream())
    subprocesses(monkeypatch, [process])
    reader = LatestFrameCapture(configuration(), uuid4())
    waiter = asyncio.create_task(reader.next_frame())
    for _ in range(20):
        if reader._process is not None:
            break
        await asyncio.sleep(0)
    await reader.close()
    with pytest.raises(CaptureUnavailableError, match="closed"):
        await asyncio.wait_for(waiter, 1)


async def test_probe_reads_fragmented_json_until_eof_before_decoding(monkeypatch):
    class FragmentedProbe(ProbeStream):
        async def read(self, maximum):
            return await super().read(min(maximum, 3))

    content = json.dumps(
        {"streams": [{"width": 1280, "height": 720, "avg_frame_rate": "30/1"}]}
    ).encode()
    process = Process(FragmentedProbe(content))

    async def spawn(*arguments, **kwargs):
        return process

    monkeypatch.setattr(capture.asyncio, "create_subprocess_exec", spawn)
    reader = LatestFrameCapture(configuration(), uuid4())
    assert await reader._probe() == (640, 360, 30.0)
    assert process.waited


async def test_decoder_shutdown_drains_buffered_stdout_before_waiting_for_reap():
    drained = asyncio.Event()

    class BufferedStream:
        def __init__(self):
            self.remaining = 3

        async def read(self, size):
            if self.remaining:
                self.remaining -= 1
                return b"buffered frame data"
            drained.set()
            return b""

    class BufferedProcess(Process):
        async def wait(self):
            await drained.wait()
            self.waited = True
            return self.returncode

    process = BufferedProcess(BufferedStream())
    await asyncio.wait_for(capture.stop_process(process), timeout=1)
    assert process.killed and process.waited and drained.is_set()
