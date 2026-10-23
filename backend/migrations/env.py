import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection, pool, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import Settings
from app.models import Base
from app.models.base import ValidatedJSONB

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)


def render_item(kind, item, autogen_context):
    if kind == "type" and isinstance(item, ValidatedJSONB):
        autogen_context.imports.add("from sqlalchemy.dialects import postgresql")
        return "postgresql.JSONB(astext_type=sa.Text())"
    return False


def migrate(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        compare_type=True,
        compare_server_default=True,
        render_item=render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


async def migrate_online() -> None:
    engine = create_async_engine(
        Settings().database_url, poolclass=pool.NullPool, hide_parameters=True
    )
    try:
        async with engine.begin() as connection:
            # Serialize migration jobs sharing the database, including concurrent Compose starts.
            await connection.execute(text("SELECT pg_advisory_xact_lock(724191002)"))
            await connection.run_sync(migrate)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    context.configure(
        url=Settings().database_url,
        target_metadata=Base.metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()
elif connection := config.attributes.get("connection"):
    migrate(connection)
else:
    asyncio.run(migrate_online())
