import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from aiokafka import TopicPartition
from aiokafka.structs import OffsetAndMetadata

from app.schemas.common import ModelKind
from app.schemas.geometry import GeometrySnapshot
from app.schemas.observation import ObservationEnvelope
from tests.factories import POLYGON, observation
from workers.vision_streamer import inspect as inspector

TOPIC = "retail.frame-observations"


def envelope(*, camera_id=None, frame=0, observed_at=None):
    instant = observed_at or datetime.now(UTC)
    snapshot = GeometrySnapshot(
        camera_id=camera_id or uuid4(),
        shelf_id=uuid4(),
        roi_version=1,
        reference_revision=uuid4(),
        product_roi=POLYGON,
        presence_roi=POLYGON,
        valid_from=instant - timedelta(seconds=1),
        valid_until=None,
    )
    data = observation(
        snapshot, {ModelKind.POSE: uuid4(), ModelKind.TEMPORAL_RULES: uuid4()}, instant, frame=frame
    )
    return ObservationEnvelope(schema_version="2.0", observation=data)


def record(data, offset=0, partition=0):
    return SimpleNamespace(
        topic=TOPIC,
        partition=partition,
        offset=offset,
        key=str(data.observation.camera_id).encode(),
        value=data.model_dump_json().encode(),
    )


class FakeConsumer:
    def __init__(self, records, *, start_error=False):
        self.records = records
        self.start_error = start_error
        self.stopped = False
        self.assigned = []
        self.seek = None
        self.commits = 0

    async def start(self):
        if self.start_error:
            raise RuntimeError("kafka://private-password@internal.invalid")

    async def topics(self):
        return {TOPIC}

    def partitions_for_topic(self, topic):
        return {0, 1}

    def assign(self, partitions):
        self.assigned = partitions

    async def seek_to_end(self, *partitions):
        self.seek = "TAIL"

    async def seek_to_beginning(self, *partitions):
        self.seek = "BEGINNING"

    async def getmany(self, **kwargs):
        if self.records:
            records, self.records = (
                self.records[: kwargs["max_records"]],
                self.records[kwargs["max_records"] :],
            )
            return {TopicPartition(TOPIC, 0): records}
        await asyncio.sleep(0.01)
        return {}

    async def end_offsets(self, partitions):
        return {partition: 20 + partition.partition for partition in partitions}

    async def commit(self, *args, **kwargs):
        self.commits += 1
        raise AssertionError("Diagnostic inspection must never commit offsets")

    async def stop(self):
        self.stopped = True


class FakeAdmin:
    def __init__(self, offsets=None, *, unavailable=False):
        self.offsets = offsets or {}
        self.unavailable = unavailable
        self.closed = False
        self.groups = []

    async def start(self):
        pass

    async def describe_topics(self, topics):
        return [
            {
                "topic": topics[0],
                "error_code": 0,
                "partitions": [
                    {"partition": 0, "error_code": 0},
                    {"partition": 1, "error_code": 0},
                ],
            }
        ]

    async def list_consumer_group_offsets(self, group, partitions):
        if self.unavailable:
            raise RuntimeError("private-admin-password")
        self.groups.append((group, partitions))
        return self.offsets

    async def close(self):
        self.closed = True


def install_clients(monkeypatch, records, *, offsets=None, start_error=False, admin_error=False):
    consumer = FakeConsumer(records, start_error=start_error)
    admin = FakeAdmin(offsets, unavailable=admin_error)
    arguments = []

    def construct(**kwargs):
        arguments.append(kwargs)
        return consumer

    monkeypatch.setattr(inspector, "AIOKafkaConsumer", construct)
    monkeypatch.setattr(inspector, "AIOKafkaAdminClient", lambda **kwargs: admin)
    return consumer, admin, arguments


def events(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


async def test_inspector_tails_without_joining_or_committing_production_group(
    settings, monkeypatch, tmp_path, capsys
):
    one, two = envelope(frame=1), envelope(frame=2)
    consumer, admin, arguments = install_clients(monkeypatch, [record(one, 5), record(two, 6)])
    path = tmp_path / "records.jsonl"
    summary = await inspector.inspect_observations(settings, limit=2, output=path)
    assert arguments[0]["group_id"] is None and arguments[0]["enable_auto_commit"] is False
    assert consumer.seek == "TAIL" and len(consumer.assigned) == 2
    assert consumer.commits == 0 and consumer.stopped and admin.closed
    assert summary["count"] == 2 and summary["diagnostic_only"]
    assert summary["production_offsets_modified"] is False
    assert summary["offset_ranges"][0]["min_offset"] == 5
    assert summary["offset_ranges"][0]["max_offset"] == 6
    assert all(
        item["lag_offsets"] is None and item["status"] == "UNCOMMITTED"
        for item in summary["consumer_group_lag"]
    )
    assert len(path.read_text().splitlines()) == 2
    assert json.loads(path.with_suffix(".jsonl.summary.json").read_text()) == summary
    lines = events(capsys)
    assert [line["event"] for line in lines] == [
        "inspection_started",
        "observation_received",
        "observation_received",
        "inspection_summary",
    ]
    assert lines[1]["observed_age_ms"] >= 0 and lines[1]["physical_exposure_latency_ms"] is None


async def test_camera_filter_and_beginning_seek_preserve_validated_output(
    settings, monkeypatch, tmp_path
):
    chosen, other = uuid4(), uuid4()
    install_clients(
        monkeypatch, [record(envelope(camera_id=other)), record(envelope(camera_id=chosen), 2)]
    )
    result = await inspector.inspect_observations(
        settings, limit=1, camera_id=chosen, from_beginning=True, output=tmp_path / "records.jsonl"
    )
    assert result["count"] == 1 and result["filtered_messages"] == 1
    data = ObservationEnvelope.model_validate_json((tmp_path / "records.jsonl").read_text())
    assert data.observation.camera_id == chosen


@pytest.mark.parametrize("tampering", ["image", "identity", "key", "oversize"])
async def test_inspector_rejects_imaging_or_inconsistent_records_without_logging_payload(
    settings, monkeypatch, capsys, tampering
):
    data = envelope()
    invalid = record(data)
    if tampering == "image":
        body = data.model_dump(mode="json")
        body["image"] = "private-image-content"
        invalid.value = json.dumps(body).encode()
    elif tampering == "identity":
        body = data.model_dump(mode="json")
        body["observation"]["id"] = str(uuid4())
        invalid.value = json.dumps(body).encode()
    elif tampering == "key":
        invalid.key = b"private-image-content"
    else:
        invalid.value = b"private-image-content" * settings.kafka_max_message_bytes
    install_clients(monkeypatch, [invalid, record(data, 2)])
    result = await inspector.inspect_observations(settings, limit=1)
    assert result["invalid_messages"] == 1 and result["count"] == 1
    lines = capsys.readouterr().out
    assert "private-image-content" not in lines
    assert "INVALID_OBSERVATION_CONTRACT" in lines


async def test_negative_wall_clock_age_is_unknown_instead_of_false_zero(
    settings, monkeypatch, capsys
):
    install_clients(
        monkeypatch, [record(envelope(observed_at=datetime.now(UTC) + timedelta(days=1)))]
    )
    await inspector.inspect_observations(settings, limit=1)
    line = next(item for item in events(capsys) if item["event"] == "observation_received")
    assert line["observed_age_ms"] is None
    assert line["clock_assumption"] == "SYNCHRONIZED_WALL_CLOCKS"


async def test_committed_group_offsets_are_read_only_and_lag_is_per_partition(
    settings, monkeypatch
):
    offsets = {
        TopicPartition(TOPIC, 0): OffsetAndMetadata(12, ""),
        TopicPartition(TOPIC, 1): OffsetAndMetadata(-1, ""),
    }
    consumer, admin, _ = install_clients(monkeypatch, [record(envelope())], offsets=offsets)
    summary = await inspector.inspect_observations(settings, limit=1, group="business-group")
    first, second = summary["consumer_group_lag"]
    assert first["end_offset"] == 20 and first["committed_offset"] == 12
    assert first["lag_offsets"] == 8 and first["status"] == "AVAILABLE"
    assert second["committed_offset"] is None and second["lag_offsets"] is None
    assert admin.groups[0][0] == "business-group" and consumer.commits == 0


async def test_group_lag_failure_does_not_turn_missing_telemetry_into_zero(settings, monkeypatch):
    consumer, admin, _ = install_clients(monkeypatch, [record(envelope())], admin_error=True)
    summary = await inspector.inspect_observations(settings, limit=1)
    assert summary["count"] == 1
    assert all(
        item["lag_offsets"] is None and item["status"] == "UNAVAILABLE"
        for item in summary["consumer_group_lag"]
    )
    assert consumer.stopped and admin.closed


async def test_broker_failure_is_bounded_closed_and_redacted(settings, monkeypatch, capsys):
    consumer, _, _ = install_clients(monkeypatch, [], start_error=True)
    summary = await inspector.inspect_observations(settings, wait_seconds=0.1)
    assert summary["status"] == "UNAVAILABLE" and consumer.stopped
    text = capsys.readouterr().out
    assert "private-password" not in text and "internal.invalid" not in text


async def test_empty_topic_deadline_returns_summary_and_closes_clients(settings, monkeypatch):
    consumer, admin, _ = install_clients(monkeypatch, [])
    summary = await inspector.inspect_observations(settings, wait_seconds=0.01)
    assert summary["status"] == "DEADLINE_REACHED" and summary["count"] == 0
    assert consumer.stopped and admin.closed


async def test_inspection_cancellation_closes_consumer(settings, monkeypatch):
    consumer, _, _ = install_clients(monkeypatch, [])
    task = asyncio.create_task(inspector.inspect_observations(settings))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert consumer.stopped


@pytest.mark.parametrize(
    "changes", [{"limit": 0}, {"limit": 1001}, {"wait_seconds": 0}, {"group": "bad/private"}]
)
async def test_inspection_rejects_unbounded_options_before_connecting(settings, changes):
    with pytest.raises(ValueError):
        await inspector.inspect_observations(settings, **changes)


async def test_partition_discovery_does_not_require_consumer_metadata_cache(settings, monkeypatch):
    consumer, _, _ = install_clients(monkeypatch, [record(envelope())])
    consumer.partitions_for_topic = lambda topic: None
    result = await inspector.inspect_observations(settings, limit=1)
    assert result["status"] == "COMPLETE"
    assert consumer.assigned == [TopicPartition(TOPIC, 0), TopicPartition(TOPIC, 1)]


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        [{"topic": TOPIC, "error_code": 3}],
        [{"topic": TOPIC, "partitions": []}],
        [{"topic": TOPIC, "partitions": [{"partition": 0}, {"partition": 0}]}],
    ],
)
async def test_invalid_admin_partition_metadata_fails_before_assignment(
    settings, monkeypatch, metadata
):
    consumer, admin, _ = install_clients(monkeypatch, [])

    async def describe(topics):
        return metadata

    admin.describe_topics = describe
    result = await inspector.inspect_observations(settings, wait_seconds=0.1)
    assert result["status"] == "UNAVAILABLE" and consumer.assigned == []
    assert consumer.stopped and admin.closed
