from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from aiokafka import ConsumerRebalanceListener, TopicPartition

from app.core import kafka_client
from app.core.config import Settings
from app.core.kafka_client import (
    observation_consumer,
    observation_producer,
    process_record,
    publish_observation,
)
from app.core.redis_client import RedisClient


async def test_cache_and_state_do_not_collide(settings: Settings) -> None:
    client = RedisClient(settings)
    client.client = AsyncMock()
    await client.set_cache("camera:1", "cached")
    await client.set_state("camera:1", "active")
    assert (
        client.client.set.await_args_list[0].args[0]
        != (client.client.set.await_args_list[1].args[0])
    )
    assert client.key("channel", "camera:1") not in {
        client.key("cache", "camera:1"),
        client.key("state", "camera:1"),
    }


@pytest.mark.parametrize("ttl", [0, 59, 301])
async def test_cache_write_rejects_invalid_ttl(settings: Settings, ttl: int) -> None:
    client = RedisClient(settings)
    client.client = AsyncMock()
    with pytest.raises(ValueError):
        await client.set_cache("scope", "value", ttl=ttl)
    client.client.set.assert_not_awaited()


async def test_publisher_uses_camera_partition_key(settings: Settings) -> None:
    producer = AsyncMock()
    camera = uuid4()
    await publish_observation(producer, settings, camera, {"sequence": 1})
    producer.send_and_wait.assert_awaited_once_with(
        settings.kafka_observations_topic, key=str(camera).encode(), value=b'{"sequence":1}'
    )


async def test_publisher_rejects_oversized_or_nonfinite_payloads(settings: Settings) -> None:
    producer = AsyncMock()
    for payload in ({"data": "x" * settings.kafka_max_message_bytes}, {"value": float("nan")}):
        with pytest.raises(ValueError):
            await publish_observation(producer, settings, uuid4(), payload)
    producer.send_and_wait.assert_not_awaited()


async def test_failed_persistence_never_acknowledges() -> None:
    consumer = AsyncMock()
    record = SimpleNamespace(topic="observations", partition=2, offset=7)
    with pytest.raises(RuntimeError):
        await process_record(consumer, record, AsyncMock(side_effect=RuntimeError("rollback")))
    consumer.commit.assert_not_awaited()


async def test_acknowledgement_follows_persistence_and_uses_next_offset() -> None:
    consumer = AsyncMock()
    record = SimpleNamespace(topic="observations", partition=2, offset=7)

    async def persist(message: object) -> None:
        consumer.commit.assert_not_awaited()

    await process_record(consumer, record, persist)
    consumer.commit.assert_awaited_once_with({TopicPartition("observations", 2): 8})


@pytest.mark.parametrize("startup_failure", [False, True])
async def test_producer_cleanup_on_failure(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, startup_failure: bool
) -> None:
    producer = AsyncMock()
    if startup_failure:
        producer.start.side_effect = RuntimeError("unavailable")
    factory = Mock(return_value=producer)
    monkeypatch.setattr(kafka_client, "AIOKafkaProducer", factory)
    with pytest.raises(RuntimeError):
        async with observation_producer(settings):
            raise RuntimeError("processing failed")
    assert factory.call_args.kwargs["enable_idempotence"] is True
    producer.stop.assert_awaited_once()


async def test_consumer_requires_manual_commit_and_registers_rebalance_listener(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumer = AsyncMock()
    consumer.subscribe = Mock()
    factory = Mock(return_value=consumer)
    monkeypatch.setattr(kafka_client, "AIOKafkaConsumer", factory)
    listener = Mock(spec=ConsumerRebalanceListener)
    with pytest.raises(RuntimeError):
        async with observation_consumer(settings, listener):
            raise RuntimeError("processing failed")
    assert factory.call_args.kwargs["enable_auto_commit"] is False
    consumer.subscribe.assert_called_once_with(
        [settings.kafka_observations_topic], listener=listener
    )
    consumer.stop.assert_awaited_once()
