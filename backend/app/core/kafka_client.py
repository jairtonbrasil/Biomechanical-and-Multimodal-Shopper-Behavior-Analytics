import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import UUID

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, ConsumerRebalanceListener, TopicPartition
from aiokafka.admin import AIOKafkaAdminClient
from aiokafka.structs import ConsumerRecord, RecordMetadata

from app.core.config import Settings


class KafkaHealth:
    def __init__(self, settings: Settings) -> None:
        self.topic = settings.kafka_observations_topic
        self.client = AIOKafkaAdminClient(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            client_id="shopper-health",
            request_timeout_ms=settings.kafka_request_timeout_ms,
        )

    async def start(self) -> None:
        await self.client.start()

    async def health(self) -> None:
        if self.topic not in await self.client.list_topics():
            raise RuntimeError("The observation topic is unavailable")

    async def close(self) -> None:
        await self.client.close()


@asynccontextmanager
async def observation_producer(settings: Settings) -> AsyncIterator[AIOKafkaProducer]:
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        client_id="shopper-vision",
        enable_idempotence=True,
        acks="all",
        max_request_size=settings.kafka_max_message_bytes,
        request_timeout_ms=settings.kafka_request_timeout_ms,
    )
    try:
        async with asyncio.timeout(settings.kafka_request_timeout_ms / 1000):
            await producer.start()
        yield producer
    finally:
        async with asyncio.timeout(settings.kafka_request_timeout_ms / 1000):
            await producer.stop()


async def publish_observation(
    producer: AIOKafkaProducer,
    settings: Settings,
    camera_id: UUID,
    payload: dict[str, object],
) -> RecordMetadata:
    value = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    if len(value) > settings.kafka_max_message_bytes - 512:
        raise ValueError("Observation exceeds the configured message size budget")
    async with asyncio.timeout(settings.kafka_request_timeout_ms / 1000):
        return await producer.send_and_wait(
            settings.kafka_observations_topic, key=str(camera_id).encode(), value=value
        )


@asynccontextmanager
async def observation_consumer(
    settings: Settings, listener: ConsumerRebalanceListener
) -> AsyncIterator[AIOKafkaConsumer]:
    consumer = AIOKafkaConsumer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        client_id="shopper-temporal",
        group_id=settings.kafka_consumer_group,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        max_poll_records=settings.kafka_max_poll_records,
        max_poll_interval_ms=settings.kafka_max_poll_interval_ms,
        max_partition_fetch_bytes=settings.kafka_max_message_bytes,
        fetch_max_bytes=settings.kafka_max_message_bytes * 2,
        request_timeout_ms=settings.kafka_request_timeout_ms,
    )
    consumer.subscribe([settings.kafka_observations_topic], listener=listener)
    try:
        async with asyncio.timeout(settings.kafka_request_timeout_ms / 1000):
            await consumer.start()
        yield consumer
    finally:
        async with asyncio.timeout(settings.kafka_request_timeout_ms / 1000):
            await consumer.stop()


async def process_record(
    consumer: AIOKafkaConsumer,
    record: ConsumerRecord,
    persist: Callable[[ConsumerRecord], Awaitable[None]],
) -> None:
    """Persist one record serially, then acknowledge its partition's next offset.

    The caller must stop on failure and coordinate ownership with its rebalance listener.
    The persistence callback must return only after its database transaction commits.
    """
    await persist(record)
    await consumer.commit({TopicPartition(record.topic, record.partition): record.offset + 1})
