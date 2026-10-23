from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings


class Database:
    def __init__(self, settings: Settings) -> None:
        self.engine = create_async_engine(
            settings.database_url,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_pre_ping=True,
            pool_timeout=settings.dependency_timeout_seconds,
            hide_parameters=True,
            connect_args={
                "timeout": settings.dependency_timeout_seconds,
                "command_timeout": settings.dependency_timeout_seconds,
            },
        )
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session, session.begin():
            # SQLAlchemy rolls back on exceptions, including task cancellation.
            yield session

    async def health(self) -> None:
        async with self.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
            revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))
            if revision != "0002_english_names":
                raise RuntimeError("Database schema is not at the required revision")

    async def close(self) -> None:
        await self.engine.dispose()


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    database: Database = request.app.state.resources.database
    async with database.transaction() as session:
        yield session
