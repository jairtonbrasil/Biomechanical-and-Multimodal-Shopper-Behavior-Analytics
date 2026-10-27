import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.schemas.common import ModelKind
from app.schemas.geometry import GeometrySnapshot
from app.schemas.observation import BoundingBox, Face, ObservationEnvelope, Pose
from app.schemas.runtime import RuntimeConfiguration, RuntimeShelf
from tests.factories import POLYGON, example_envelope
from workers.vision_streamer.configuration import (
    CameraLease,
    ConfigurationUnavailable,
    VisionConfiguration,
)
from workers.vision_streamer.diagnostics import Diagnostics
from workers.vision_streamer.gaze import GazeEstimator
from workers.vision_streamer.kinematics import KinematicsEngine
from workers.vision_streamer.observations import build_observation, unsupported_product_evidence
from workers.vision_streamer.publisher import PublicationUnavailable, publish_envelope
from workers.vision_streamer.registry import local_artifacts
from workers.vision_streamer.types import Detection


def assembled_observation():
    camera, shelf, reference = uuid4(), uuid4(), uuid4()
    polygon = {**POLYGON, "points": [[0, 0.2], [0.3, 0.2], [0.3, 0.6], [0, 0.6]]}
    geometry = GeometrySnapshot(
        camera_id=camera,
        shelf_id=shelf,
        roi_version=1,
        reference_revision=reference,
        product_roi=polygon,
        presence_roi=POLYGON,
        valid_from=datetime(2026, 1, 1, tzinfo=UTC),
        valid_until=None,
    )
    runtime = RuntimeConfiguration(
        camera_id=camera,
        configuration_revision=uuid4(),
        source_revision=uuid4(),
        is_active=True,
        proxy_path=f"cameras/{camera}",
        reference_revision=reference,
        shelves=(
            RuntimeShelf(
                id=shelf,
                name="Shelf",
                category="Drinks",
                roi_version=1,
                reference_revision=reference,
                product_roi=polygon,
                presence_roi=POLYGON,
            ),
        ),
    )
    pose = dict.fromkeys(Pose.model_fields)
    for name, x, y in [
        ("left_shoulder", 0.4, 0.4),
        ("right_shoulder", 0.6, 0.4),
        ("left_hip", 0.4, 0.7),
        ("left_elbow", 0.2, 0.4),
        ("left_wrist", 0, 0.4),
    ]:
        pose[name] = {"x": x, "y": y, "confidence": 0.9}
    detection = Detection(
        "1", BoundingBox(x1=0, y1=0.2, x2=0.8, y2=0.9, confidence=0.9), Pose(**pose)
    )
    options = dict(
        configuration=VisionConfiguration(runtime, (geometry,), uuid4()),
        model_ids={
            kind: uuid4() for kind in (ModelKind.POSE, ModelKind.FACE, ModelKind.TEMPORAL_RULES)
        },
        session_id=uuid4(),
        source_epoch=uuid4(),
        frame_seq=7,
        observed_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        source_fps=30,
        sampled_fps=10,
        width=1000,
        height=1000,
        face=Face(visible=None, expression_detected=None, confidence=None),
    )
    return detection, options


def test_pipeline_stable_observation_and_explicit_unsupported_evidence():
    detection, options = assembled_observation()
    first = build_observation(
        detection, kinematics=KinematicsEngine(), gaze=GazeEstimator(), **options
    )
    second = build_observation(
        detection, kinematics=KinematicsEngine(), gaze=GazeEstimator(), **options
    )
    assert first == second
    observation = ObservationEnvelope.model_validate_json(first.model_dump_json()).observation
    assert observation.reach_shelf_id == options["configuration"].runtime.shelves[0].id
    assert observation.attributes.hands.left.in_product_roi is True
    assert observation.attributes.kinematics.reach_candidate is True
    assert observation.attributes.presence[0].in_roi is None
    assert observation.attributes.movement.state is None
    assert observation.attributes.face.expression_detected is None
    assert observation.attributes.evidence == unsupported_product_evidence()
    assert len(first.model_dump_json().encode()) < 262144 - 512


def test_wrist_entry_without_qualifying_arm_has_no_reach_target():
    detection, options = assembled_observation()
    detection = replace(detection, pose=detection.pose.model_copy(update={"left_elbow": None}))
    envelope = build_observation(
        detection, kinematics=KinematicsEngine(), gaze=GazeEstimator(), **options
    )
    assert envelope.observation.attributes.hands.left.in_product_roi is True
    assert envelope.observation.reach_shelf_id is None
    assert envelope.observation.attributes.evidence.object_left_shelf is None


async def test_publisher_retries_identical_payload_and_reports_ack(settings, capsys):
    settings = settings.model_copy(update={"vision_publish_attempts": 2})
    metadata = SimpleNamespace(topic=settings.kafka_observations_topic, partition=1, offset=9)
    producer = SimpleNamespace(
        send_and_wait=AsyncMock(side_effect=[RuntimeError("private-url"), metadata])
    )
    diagnostics = Diagnostics()
    assert await publish_envelope(producer, settings, example_envelope(), diagnostics) is metadata
    calls = producer.send_and_wait.call_args_list
    assert calls[0] == calls[1]
    assert diagnostics.counts["producer_retries"] == 1
    assert diagnostics.counts["observations_published"] == 1
    output = capsys.readouterr().out
    assert "private-url" not in output
    assert '"partition":1' in output and '"offset":9' in output


async def test_publisher_bounded_retry_does_not_claim_failed_publication(settings):
    settings = settings.model_copy(update={"vision_publish_attempts": 1})
    producer = SimpleNamespace(send_and_wait=AsyncMock(side_effect=OSError("private-url")))
    diagnostics = Diagnostics()
    with pytest.raises(PublicationUnavailable):
        await publish_envelope(producer, settings, example_envelope(), diagnostics)
    assert producer.send_and_wait.await_count == 1
    assert diagnostics.counts["observations_published"] == 0


async def test_publisher_cancellation_is_not_retried(settings):
    producer = SimpleNamespace(send_and_wait=AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await publish_envelope(producer, settings, example_envelope(), Diagnostics())
    assert producer.send_and_wait.await_count == 1


def test_timing_samples_are_bounded_and_invalid_measurements_rejected():
    diagnostics = Diagnostics(sample_limit=3)
    for value in range(10):
        diagnostics.record("inference", float(value))
    assert len(diagnostics.timings["inference"]) == 3
    assert diagnostics.snapshot()["physical_camera_latency_ms"] is None
    with pytest.raises(ValueError):
        diagnostics.record("inference", float("nan"))


def test_model_bundle_checksum_validation_fails_closed(tmp_path):
    manifest = {
        "schema_version": "1.0",
        "pose": {"filename": "pose.pt", "sha256": "a" * 64},
        "face": {"filename": "face.onnx", "sha256": "b" * 64},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "pose.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        local_artifacts(tmp_path / "manifest.json")
    manifest["pose"]["filename"] = "../pose.pt"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="filenames"):
        local_artifacts(tmp_path / "manifest.json")


async def test_camera_lease_rejects_duplicate_owner_and_releases_own_token():
    redis = SimpleNamespace(
        client=SimpleNamespace(set=AsyncMock(return_value=True), eval=AsyncMock(return_value=1)),
        key=lambda category, name: f"test:{category}:{name}",
    )
    identifier = uuid4()
    async with CameraLease(redis, identifier) as lease:
        assert not lease.lost
        redis.client.set.return_value = False
        with pytest.raises(ConfigurationUnavailable):
            async with CameraLease(redis, identifier):
                pytest.fail("Duplicate owner was accepted")
    assert redis.client.eval.call_args.args[-1] == lease.owner


async def test_publication_error_survives_producer_cleanup_failure(settings, monkeypatch):
    from app.core import kafka_client

    producer = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(side_effect=TimeoutError))
    monkeypatch.setattr(kafka_client, "AIOKafkaProducer", lambda **kwargs: producer)
    with pytest.raises(PublicationUnavailable):
        async with kafka_client.observation_producer(settings):
            raise PublicationUnavailable("Bounded publication failed")
    producer.stop.assert_awaited_once()


async def test_producer_cleanup_failure_is_visible_after_success(settings, monkeypatch):
    from app.core import kafka_client

    producer = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(side_effect=TimeoutError))
    monkeypatch.setattr(kafka_client, "AIOKafkaProducer", lambda **kwargs: producer)
    with pytest.raises(TimeoutError):
        async with kafka_client.observation_producer(settings):
            pass
