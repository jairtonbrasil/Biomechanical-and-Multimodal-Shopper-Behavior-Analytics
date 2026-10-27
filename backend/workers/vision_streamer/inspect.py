"""Read-only Kafka inspection; this is not the temporal business consumer."""

import argparse
import asyncio
import json
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from aiokafka import AIOKafkaConsumer, TopicPartition
from aiokafka.admin import AIOKafkaAdminClient

from app.core.config import Settings
from app.domain.identity import TrackScope, observation_id
from app.schemas.observation import ObservationEnvelope


def emit(event: str, **fields: Any) -> None:
    print(
        json.dumps({"event": event, **fields}, allow_nan=False, separators=(",", ":")), flush=True
    )


def decode_observation(record: Any, maximum_bytes: int) -> ObservationEnvelope:
    if not isinstance(record.value, bytes) or len(record.value) > maximum_bytes - 512:
        raise ValueError("Observation payload is outside its byte budget")
    envelope = ObservationEnvelope.model_validate_json(record.value)
    observation = envelope.observation
    scope = TrackScope(
        observation.camera_id, observation.tracking_session_id, observation.local_track_id
    )
    if observation.id != observation_id(scope, observation.frame_seq):
        raise ValueError("Observation identity is inconsistent")
    if record.key != str(observation.camera_id).encode():
        raise ValueError("Observation camera differs from its partition key")
    return envelope


async def describe_partitions(settings: Settings, topic: str) -> list[TopicPartition]:
    admin = AIOKafkaAdminClient(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        client_id="shopper-read-only-inspector-metadata",
        request_timeout_ms=settings.kafka_request_timeout_ms,
    )
    try:
        await admin.start()
        descriptions = await admin.describe_topics([topic])
        if not isinstance(descriptions, list) or len(descriptions) != 1:
            raise RuntimeError("Observation topic metadata is unavailable")
        description = descriptions[0]
        if description.get("topic") != topic or description.get("error_code", 0) != 0:
            raise RuntimeError("Observation topic metadata is unavailable")
        entries = description.get("partitions")
        if not isinstance(entries, list) or not 1 <= len(entries) <= 1024:
            raise RuntimeError("Observation topic partitions are unavailable")
        identifiers = []
        for entry in entries:
            identifier = entry.get("partition")
            if (
                type(identifier) is not int
                or identifier < 0
                or entry.get("error_code", 0) != 0
                or identifier in identifiers
            ):
                raise RuntimeError("Observation topic partitions are invalid")
            identifiers.append(identifier)
        return [TopicPartition(topic, identifier) for identifier in sorted(identifiers)]
    finally:
        try:
            async with asyncio.timeout(settings.dependency_timeout_seconds):
                await admin.close()
        except Exception:
            pass


async def group_lag(
    consumer: AIOKafkaConsumer,
    settings: Settings,
    group: str,
    partitions: list[TopicPartition],
) -> list[dict]:
    admin = AIOKafkaAdminClient(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        client_id="shopper-read-only-inspector-admin",
        request_timeout_ms=settings.kafka_request_timeout_ms,
    )
    end_offsets: dict = {}
    try:
        async with asyncio.timeout(settings.kafka_request_timeout_ms / 1000):
            end_offsets = await consumer.end_offsets(partitions)
            await admin.start()
            committed = await admin.list_consumer_group_offsets(group, partitions=partitions)
        result = []
        for partition in partitions:
            value = committed.get(partition)
            offset = value.offset if value is not None and value.offset >= 0 else None
            end = end_offsets.get(partition)
            if offset is None:
                lag, status = None, "UNCOMMITTED"
            elif end is None or offset > end:
                lag, status = None, "OUT_OF_RANGE"
            else:
                lag, status = end - offset, "AVAILABLE"
            result.append(
                {
                    "topic": partition.topic,
                    "partition": partition.partition,
                    "group": group,
                    "end_offset": end,
                    "committed_offset": offset,
                    "lag_offsets": lag,
                    "status": status,
                }
            )
        return result
    except asyncio.CancelledError:
        raise
    except Exception:
        return [
            {
                "topic": partition.topic,
                "partition": partition.partition,
                "group": group,
                "end_offset": end_offsets.get(partition),
                "committed_offset": None,
                "lag_offsets": None,
                "status": "UNAVAILABLE",
            }
            for partition in partitions
        ]
    finally:
        try:
            async with asyncio.timeout(settings.dependency_timeout_seconds):
                await admin.close()
        except Exception:
            pass


def private_output(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "w", encoding="utf-8")


async def inspect_observations(
    settings: Settings,
    *,
    limit: int = 10,
    wait_seconds: float = 30,
    camera_id: UUID | None = None,
    from_beginning: bool = False,
    output: Path | None = None,
    group: str = "retail.temporal-processor",
) -> dict:
    if not 1 <= limit <= 1000 or not 0 < wait_seconds <= 3600:
        raise ValueError("Inspection limits are outside their supported bounds")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,249}", group):
        raise ValueError("A valid diagnostic group name is required")
    topic = settings.kafka_observations_topic
    consumer = AIOKafkaConsumer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        client_id="shopper-read-only-inspector",
        group_id=None,
        enable_auto_commit=False,
        auto_offset_reset="earliest" if from_beginning else "latest",
        max_poll_records=min(limit, 100),
        max_partition_fetch_bytes=settings.kafka_max_message_bytes,
        fetch_max_bytes=settings.kafka_max_message_bytes * 2,
        request_timeout_ms=settings.kafka_request_timeout_ms,
    )
    count, invalid, filtered, examined = 0, 0, 0, 0
    partitions: list[TopicPartition] = []
    offsets: dict[str, dict] = {}
    status = "COMPLETE"
    destination = None
    failure_type = None
    lag = []
    try:
        if output is not None:
            destination = private_output(output)
        async with asyncio.timeout(wait_seconds):
            await consumer.start()
            # topics() returns a separate metadata snapshot in aiokafka; it does not
            # populate an unsubscribed consumer's local partition cache.
            partitions = await describe_partitions(settings, topic)
            consumer.assign(partitions)
            if from_beginning:
                await consumer.seek_to_beginning(*partitions)
            else:
                await consumer.seek_to_end(*partitions)
            emit(
                "inspection_started",
                topic=topic,
                camera_id=str(camera_id) if camera_id else None,
                start="BEGINNING" if from_beginning else "TAIL",
                limit=limit,
                diagnostic_only=True,
                production_offsets_modified=False,
            )
            while count < limit and examined < limit * 100:
                batches = await consumer.getmany(
                    timeout_ms=500, max_records=min(limit - count, 100)
                )
                for records in batches.values():
                    for record in records:
                        examined += 1
                        try:
                            envelope = decode_observation(record, settings.kafka_max_message_bytes)
                        except Exception:
                            invalid += 1
                            if invalid <= 10:
                                emit(
                                    "observation_rejected",
                                    topic=record.topic,
                                    partition=record.partition,
                                    offset=record.offset,
                                    reason="INVALID_OBSERVATION_CONTRACT",
                                )
                            continue
                        observation = envelope.observation
                        if camera_id is not None and observation.camera_id != camera_id:
                            filtered += 1
                            continue
                        age_ms = (
                            datetime.now(UTC) - observation.observed_at
                        ).total_seconds() * 1000
                        count += 1
                        key = f"{record.topic}:{record.partition}"
                        span = offsets.setdefault(
                            key,
                            {
                                "topic": record.topic,
                                "partition": record.partition,
                                "min_offset": record.offset,
                                "max_offset": record.offset,
                            },
                        )
                        span["min_offset"] = min(span["min_offset"], record.offset)
                        span["max_offset"] = max(span["max_offset"], record.offset)
                        emit(
                            "observation_received",
                            topic=record.topic,
                            partition=record.partition,
                            offset=record.offset,
                            observation_id=str(observation.id),
                            camera_id=str(observation.camera_id),
                            tracking_session_id=str(observation.tracking_session_id),
                            frame_seq=observation.frame_seq,
                            payload_bytes=len(record.value),
                            observed_age_ms=round(age_ms, 3) if age_ms >= 0 else None,
                            clock_assumption="SYNCHRONIZED_WALL_CLOCKS",
                            physical_exposure_latency_ms=None,
                        )
                        if destination is not None:
                            destination.write(envelope.model_dump_json() + "\n")
                        if count >= limit:
                            break
                    if count >= limit or examined >= limit * 100:
                        break
            if count < limit:
                status = "SCAN_LIMIT_REACHED"
    except TimeoutError:
        status = "DEADLINE_REACHED"
    except asyncio.CancelledError:
        status = "CANCELLED"
        raise
    except Exception as error:
        status, failure_type = "UNAVAILABLE", type(error).__name__
    finally:
        if partitions and status != "CANCELLED":
            lag = await group_lag(consumer, settings, group, partitions)
        try:
            async with asyncio.timeout(settings.dependency_timeout_seconds):
                await consumer.stop()
        except Exception:
            pass
        if destination is not None:
            destination.close()
    summary = {
        "status": status,
        "topic": topic,
        "count": count,
        "invalid_messages": invalid,
        "filtered_messages": filtered,
        "examined_messages": examined,
        "offset_ranges": list(offsets.values()),
        "consumer_group_lag": lag,
        "diagnostic_only": True,
        "production_offsets_modified": False,
        "clock_assumption": "SYNCHRONIZED_WALL_CLOCKS",
        "physical_exposure_latency_ms": None,
    }
    if failure_type:
        summary["error_type"] = failure_type
    if output is not None:
        with private_output(output.with_suffix(output.suffix + ".summary.json")) as file:
            file.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    emit("inspection_summary", **summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30, dest="wait_seconds")
    parser.add_argument("--camera-id", type=UUID)
    parser.add_argument("--from-beginning", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--group", default="retail.temporal-processor")
    arguments = parser.parse_args()
    logging.getLogger("aiokafka").setLevel(logging.CRITICAL)
    try:
        summary = asyncio.run(inspect_observations(Settings(), **vars(arguments)))
    except KeyboardInterrupt:
        return
    except Exception:
        raise SystemExit(
            "Kafka inspection could not start; verify configuration and limits"
        ) from None
    if summary["status"] == "UNAVAILABLE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
