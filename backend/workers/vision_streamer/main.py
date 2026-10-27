"""Run bounded RTSP inference and publish versioned observations to Kafka."""

import argparse
import asyncio
import json
import os
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from app.core.config import Settings
from app.core.database import Database
from app.core.kafka_client import observation_producer
from app.core.redis_client import RedisClient
from app.core.references import ReferenceManager
from workers.vision_streamer.capture import LatestFrameCapture
from workers.vision_streamer.configuration import (
    CameraLease,
    ConfigurationUnavailable,
    load_configuration,
)
from workers.vision_streamer.diagnostics import Diagnostics, configure_worker_logging, emit
from workers.vision_streamer.face import FaceEstimator
from workers.vision_streamer.gaze import GazeEstimator
from workers.vision_streamer.kinematics import KinematicsEngine
from workers.vision_streamer.observations import build_observation
from workers.vision_streamer.pose import PoseEstimator
from workers.vision_streamer.preview import DiagnosticPreview
from workers.vision_streamer.publisher import publish_envelope
from workers.vision_streamer.registry import local_artifacts, register_models


async def wait_configuration(camera_id, database, redis, references, *, attempts=30):
    for attempt in range(attempts):
        try:
            return await load_configuration(camera_id, database, redis, references)
        except ConfigurationUnavailable:
            if attempt + 1 == attempts:
                raise
            await asyncio.sleep(0.5)


async def run(settings: Settings, camera_id: UUID) -> dict:
    database, redis = Database(settings), RedisClient(settings)
    references = ReferenceManager(settings, redis)
    capture = LatestFrameCapture(settings, camera_id)
    diagnostics = Diagnostics()
    identifier = f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    directory = settings.vision_output_directory / identifier
    directory.mkdir(parents=True, exist_ok=False)
    preview = None
    status, failure_type = "completed", None
    sample_count, sample = 0, None
    model_ids = {}
    configuration = None
    session_id = None
    source_epoch = None
    offsets: dict[int, int] = {}
    started = time.monotonic()
    emit("vision_starting", camera_id=camera_id, run_id=identifier)
    try:
        async with CameraLease(redis, camera_id) as lease:
            configuration = await wait_configuration(camera_id, database, redis, references)
            artifacts = await asyncio.to_thread(local_artifacts, settings.vision_model_manifest)
            os.environ["YOLO_AUTOINSTALL"] = "false"
            os.environ["YOLO_VERBOSE"] = "false"
            import torch

            torch.set_num_threads(settings.vision_cpu_threads)
            model_started = time.monotonic()
            pose = await asyncio.to_thread(
                PoseEstimator,
                artifacts["pose"].path,
                settings.vision_confidence,
                settings.vision_image_size,
                settings.vision_device,
                settings.vision_max_people,
                keypoint_confidence=settings.vision_confidence,
            )
            face = await asyncio.to_thread(
                FaceEstimator, artifacts["face"].path, threads=settings.vision_cpu_threads
            )
            model_ids = await register_models(
                settings, database, artifacts, pose.parameters, face.parameters
            )
            emit(
                "vision_models_ready",
                camera_id=camera_id,
                model_ids=model_ids,
                model_initialization_ms=round((time.monotonic() - model_started) * 1000, 3),
            )
            kinematics = KinematicsEngine(settings.vision_confidence, settings.sampling_fps)
            gaze = GazeEstimator(settings.vision_confidence)
            sample = (directory / "observations.jsonl").open("w")
            active_started = time.monotonic()
            last_summary = active_started
            await capture.start()
            async with observation_producer(settings) as producer:
                while time.monotonic() - active_started < settings.vision_max_seconds:
                    if (
                        settings.vision_max_frames
                        and diagnostics.counts["frames_processed"] >= settings.vision_max_frames
                    ):
                        break
                    if lease.lost or await redis.client.get(lease.key) != lease.owner:
                        raise ConfigurationUnavailable("Vision camera ownership was lost")
                    latest = await wait_configuration(
                        camera_id, database, redis, references, attempts=30
                    )
                    if (
                        latest.runtime.configuration_revision
                        != configuration.runtime.configuration_revision
                    ):
                        emit(
                            "vision_configuration_cutover",
                            camera_id=camera_id,
                            previous_revision=configuration.runtime.configuration_revision,
                            configuration_revision=latest.runtime.configuration_revision,
                        )
                        source_changed = (
                            latest.runtime.source_revision != configuration.runtime.source_revision
                        )
                        configuration = latest
                        pose.reset()
                        kinematics.reset()
                        gaze.reset()
                        session_id = None
                        if source_changed:
                            await capture.close()
                            capture = LatestFrameCapture(settings, camera_id)
                            await capture.start()
                    remaining = settings.vision_max_seconds - (time.monotonic() - active_started)
                    if remaining <= 0:
                        break
                    try:
                        frame = await asyncio.wait_for(
                            capture.next_frame(),
                            timeout=remaining,
                        )
                    except TimeoutError:
                        if time.monotonic() - active_started >= settings.vision_max_seconds:
                            break
                        raise
                    if frame.source_epoch != source_epoch or session_id is None:
                        source_epoch, session_id = frame.source_epoch, uuid4()
                        pose.reset()
                        kinematics.reset()
                        gaze.reset()
                        emit(
                            "vision_tracking_epoch",
                            camera_id=camera_id,
                            source_epoch=source_epoch,
                            tracking_session_id=session_id,
                            configuration_revision=configuration.runtime.configuration_revision,
                        )
                    if any(frame.observed_at < item.valid_from for item in configuration.geometry):
                        diagnostics.counts["configuration_frames_discarded"] += 1
                        continue
                    inference_started = time.monotonic()
                    capture_age_ms = max(0.0, (inference_started - frame.received_monotonic) * 1000)
                    detections = await asyncio.to_thread(pose.infer, frame.image)
                    envelopes = []
                    height, width = frame.image.shape[:2]
                    for detection in detections:
                        facial = await asyncio.to_thread(face.infer, frame.image, detection.pose)
                        envelope = build_observation(
                            detection,
                            configuration=configuration,
                            model_ids=model_ids,
                            session_id=session_id,
                            source_epoch=source_epoch,
                            frame_seq=frame.frame_seq,
                            observed_at=frame.observed_at,
                            source_fps=frame.source_fps,
                            sampled_fps=settings.sampling_fps,
                            width=width,
                            height=height,
                            kinematics=kinematics,
                            gaze=gaze,
                            face=facial,
                            confidence_threshold=settings.vision_confidence,
                        )
                        envelopes.append(envelope)
                    inference_ms = (time.monotonic() - inference_started) * 1000
                    after_inference = await wait_configuration(
                        camera_id, database, redis, references, attempts=30
                    )
                    if (
                        after_inference.runtime.configuration_revision
                        != configuration.runtime.configuration_revision
                    ):
                        diagnostics.counts["configuration_frames_discarded"] += 1
                        continue
                    if lease.lost or await redis.client.get(lease.key) != lease.owner:
                        raise ConfigurationUnavailable("Vision camera ownership was lost")
                    diagnostics.counts["frames_processed"] += 1
                    diagnostics.counts["people_observed"] += len(envelopes)
                    diagnostics.record("capture_to_inference_age", capture_age_ms)
                    diagnostics.record("inference", inference_ms)
                    for envelope in envelopes:
                        metadata = await publish_envelope(producer, settings, envelope, diagnostics)
                        offsets[metadata.partition] = metadata.offset
                        diagnostics.record(
                            "capture_to_publish_ack",
                            max(0.0, (time.monotonic() - frame.received_monotonic) * 1000),
                        )
                        if sample_count < settings.vision_observation_sample_limit:
                            sample.write(envelope.model_dump_json() + "\n")
                            sample.flush()
                            sample_count += 1
                    if settings.vision_preview_enabled:
                        if preview is None:
                            preview = DiagnosticPreview(
                                directory / "annotated.mp4",
                                min(settings.sampling_fps, 10),
                                settings.vision_preview_max_seconds,
                                width,
                                height,
                                geometry=configuration.geometry,
                            )
                            await preview.start()
                        preview.geometry = configuration.geometry
                        await preview.write(
                            frame.image,
                            envelopes,
                            frame.frame_seq,
                            inference_ms,
                            capture_age_ms,
                            frame.observed_at,
                        )
                    if time.monotonic() - last_summary >= settings.vision_summary_seconds:
                        emit(
                            "vision_summary",
                            camera_id=camera_id,
                            capture=capture.counters,
                            **diagnostics.snapshot(),
                        )
                        last_summary = time.monotonic()
    except asyncio.CancelledError:
        status, failure_type = "cancelled", "CancelledError"
        raise
    except Exception as error:
        status, failure_type = "failed", type(error).__name__
        emit("vision_failed", camera_id=camera_id, error_type=failure_type)
        raise
    finally:
        await capture.close()
        if preview is not None:
            try:
                await preview.close()
            except Exception as error:
                status, failure_type = "failed", type(error).__name__
                emit("preview_failed", error_type=failure_type)
        if sample:
            sample.close()
        await database.close()
        await redis.close()
        report = {
            "run_id": identifier,
            "camera_id": str(camera_id),
            "status": status,
            "failure_type": failure_type,
            "model_ids": {str(k): str(v) for k, v in model_ids.items()},
            "configuration_revision": str(configuration.runtime.configuration_revision)
            if configuration
            else None,
            "source_revision": str(configuration.runtime.source_revision)
            if configuration
            else None,
            "capture": capture.counters,
            "offsets": offsets,
            "sampled_observations": sample_count,
            "sampling_target_fps": settings.sampling_fps,
            "wall_seconds": round(time.monotonic() - started, 3),
            "diagnostic_clip": "annotated.mp4" if (directory / "annotated.mp4").is_file() else None,
            "consumer_business_processing": "NOT_IMPLEMENTED_STAGE_5",
            "scientific_acceptance": "PENDING_LABELED_EVALUATION",
            "product_capabilities": {"pickup": False, "return": False, "touch": False},
            **diagnostics.snapshot(),
        }
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (settings.vision_output_directory / "latest.json").write_text(
            json.dumps({"run_directory": identifier, "report": report}, indent=2) + "\n"
        )
        emit(
            "vision_stopped",
            camera_id=camera_id,
            status=status,
            report=f"{identifier}/report.json",
            **diagnostics.snapshot(),
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-id", type=UUID)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    configure_worker_logging()
    settings = Settings()
    camera_id = args.camera_id or (
        UUID(settings.vision_camera_id) if settings.vision_camera_id else None
    )
    if camera_id is None:
        raise SystemExit(
            "Provide --camera-id or VISION_CAMERA_ID; demo_stage4.py selects the demo camera"
        )
    values = settings.model_dump()
    for name in ("max_frames", "max_seconds"):
        value = getattr(args, name)
        if value is not None:
            values[f"vision_{name}"] = value
    if args.output is not None:
        values["vision_output_directory"] = args.output
    settings = Settings(_env_file=None, **values)

    async def execute():
        task = asyncio.current_task()
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
        try:
            return await run(settings, camera_id)
        finally:
            loop.remove_signal_handler(signal.SIGTERM)

    try:
        report = asyncio.run(execute())
        if report["status"] != "completed":
            raise SystemExit("Vision run did not complete; inspect its sanitized execution report")
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception:
        raise SystemExit("Vision worker failed; inspect its sanitized execution report") from None


if __name__ == "__main__":
    main()
