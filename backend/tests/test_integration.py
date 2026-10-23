import asyncio
from uuid import uuid4

import pytest
from aiokafka import ConsumerRebalanceListener, TopicPartition
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings
from app.core.database import Database
from app.core.kafka_client import observation_consumer, observation_producer, publish_observation
from app.core.redis_client import RedisClient
from app.core.resources import open_resources

pytestmark = pytest.mark.integration


async def test_real_dependency_readiness() -> None:
    async with open_resources(Settings()) as resources:
        assert await resources.ready()


async def test_postgresql_transaction_rolls_back_partial_write() -> None:
    database = Database(Settings())
    try:
        async with database.engine.connect() as connection:
            await connection.execute(text("CREATE TEMP TABLE stage1_probe (value integer)"))
            await connection.commit()
            database.session_factory = async_sessionmaker(connection, expire_on_commit=False)
            with pytest.raises(RuntimeError):
                async with database.transaction() as session:
                    await session.execute(text("INSERT INTO stage1_probe VALUES (1)"))
                    raise RuntimeError("Simulated persistence failure")
            async with database.transaction() as session:
                assert await session.scalar(text("SELECT count(*) FROM stage1_probe")) == 0
    finally:
        await database.close()


async def test_real_redis_namespace_isolation_and_expiration() -> None:
    client = RedisClient(Settings())
    scope = f"integration:{uuid4()}"
    try:
        await client.set_cache(scope, "cached")
        await client.set_state(scope, "active")
        assert await client.get_cache(scope) == "cached"
        assert await client.get_state(scope) == "active"
        assert 0 < await client.client.ttl(client.key("cache", scope)) <= 300
        assert 0 < await client.client.ttl(client.key("state", scope)) <= 60
    finally:
        await client.client.delete(client.key("cache", scope), client.key("state", scope))
        await client.close()


class TestRebalanceListener(ConsumerRebalanceListener):
    __test__ = False

    async def on_partitions_revoked(self, revoked: set[TopicPartition]) -> None:
        pass

    async def on_partitions_assigned(self, assigned: set[TopicPartition]) -> None:
        pass


async def test_kafka_uncommitted_record_is_redelivered() -> None:
    topic = f"stage1.integration.{uuid4().hex}"
    settings = Settings().model_copy(
        update={"kafka_observations_topic": topic, "kafka_consumer_group": topic}
    )
    admin = AIOKafkaAdminClient(bootstrap_servers=settings.kafka_bootstrap_servers)
    try:
        await admin.start()
        await admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)])
        camera = uuid4()
        async with observation_producer(settings) as producer:
            await publish_observation(producer, settings, camera, {"test": "redelivery"})
        async with observation_consumer(settings, TestRebalanceListener()) as consumer:
            first = await asyncio.wait_for(consumer.getone(), timeout=20)
            assert first.key == str(camera).encode()
        async with observation_consumer(settings, TestRebalanceListener()) as consumer:
            repeated = await asyncio.wait_for(consumer.getone(), timeout=20)
            assert (repeated.partition, repeated.offset, repeated.value) == (
                first.partition,
                first.offset,
                first.value,
            )
    finally:
        await admin.delete_topics([topic])
        await admin.close()
