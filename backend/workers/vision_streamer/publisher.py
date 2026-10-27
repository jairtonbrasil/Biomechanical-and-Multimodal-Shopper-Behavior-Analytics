"""Sequential bounded publication, preserving observation identity across retries."""

import asyncio
import time

from app.core.config import Settings
from app.core.kafka_client import publish_observation
from app.schemas.observation import ObservationEnvelope
from workers.vision_streamer.diagnostics import Diagnostics, emit


class PublicationUnavailable(RuntimeError):
    pass


async def publish_envelope(
    producer, settings: Settings, envelope: ObservationEnvelope, diagnostics: Diagnostics
):
    observation = envelope.observation
    payload = envelope.model_dump(mode="json")
    started = time.monotonic()
    for attempt in range(settings.vision_publish_attempts):
        try:
            async with asyncio.timeout(settings.vision_publish_timeout_seconds):
                metadata = await publish_observation(
                    producer, settings, observation.camera_id, payload
                )
            milliseconds = (time.monotonic() - started) * 1000
            diagnostics.record("producer_ack", milliseconds)
            diagnostics.counts["observations_published"] += 1
            diagnostics.counts["payload_bytes"] += len(envelope.model_dump_json().encode())
            # Periodic sampling bounds diagnostic volume while preserving offset correlation.
            if (
                diagnostics.counts["observations_published"] <= 3
                or diagnostics.counts["observations_published"] % 50 == 0
            ):
                emit(
                    "producer_acknowledged",
                    camera_id=observation.camera_id,
                    observation_id=observation.id,
                    tracking_session_id=observation.tracking_session_id,
                    frame_seq=observation.frame_seq,
                    topic=metadata.topic,
                    partition=metadata.partition,
                    offset=metadata.offset,
                    payload_bytes=len(envelope.model_dump_json().encode()),
                    producer_ack_ms=round(milliseconds, 3),
                )
            return metadata
        except asyncio.CancelledError:
            raise
        except Exception as error:
            diagnostics.counts["producer_failures"] += 1
            emit(
                "producer_retry",
                attempt=attempt + 1,
                observation_id=observation.id,
                error_type=type(error).__name__,
            )
            if attempt + 1 < settings.vision_publish_attempts:
                diagnostics.counts["producer_retries"] += 1
                await asyncio.sleep(min(0.25 * 2**attempt, 2))
    raise PublicationUnavailable("Broker publication deadline exhausted") from None
