import os
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.core.database import Database


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        app_environment="test",
        database_password="database-test-password",
        redis_password="redis-test-password",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if os.environ.get("RUN_INTEGRATION_TESTS") == "1":
        return
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(
                pytest.mark.skip(reason="Set RUN_INTEGRATION_TESTS=1 for local services")
            )


def migration_config(connection) -> Config:
    configuration = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    configuration.attributes["connection"] = connection
    return configuration


@pytest_asyncio.fixture
async def domain_database():
    settings = Settings()
    schema = f"test_{uuid4().hex}"
    admin_engine = create_async_engine(settings.database_url, hide_parameters=True)
    database = Database(settings)
    await database.engine.dispose()
    database.engine = create_async_engine(
        settings.database_url,
        hide_parameters=True,
        connect_args={"server_settings": {"search_path": schema}},
    )
    database.session_factory = async_sessionmaker(database.engine, expire_on_commit=False)
    try:
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        async with database.engine.begin() as connection:
            await connection.run_sync(lambda sync: command.upgrade(migration_config(sync), "head"))
        yield database
    finally:
        await database.close()
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin_engine.dispose()
