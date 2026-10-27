import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app import main
from app.core import resources as resource_module
from app.core.config import Settings
from app.core.resources import Resources, open_resources


@pytest.mark.parametrize("available", [True, False])
async def test_health_responses_are_minimal(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, available: bool
) -> None:
    @asynccontextmanager
    async def fake_resources(configuration: Settings):
        yield SimpleNamespace(ready=AsyncMock(return_value=available))

    monkeypatch.setattr(main, "open_resources", fake_resources)
    application = main.create_app(settings)
    async with application.router.lifespan_context(application):
        async with AsyncClient(
            transport=ASGITransport(application), base_url="http://test"
        ) as client:
            assert (await client.get("/health/live")).json() == {"status": "alive"}
            response = await client.get("/health/ready")
            assert response.status_code == (200 if available else 503)
            assert set(response.json()) == {"status"}
            assert (await client.get("/docs")).status_code == 404


async def test_unhandled_errors_do_not_expose_credentials(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    @asynccontextmanager
    async def fake_resources(configuration: Settings):
        yield SimpleNamespace(ready=AsyncMock(side_effect=RuntimeError("credential-sentinel")))

    monkeypatch.setattr(main, "open_resources", fake_resources)
    application = main.create_app(settings)
    async with application.router.lifespan_context(application):
        async with AsyncClient(
            transport=ASGITransport(application, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.get("/health/ready")
    assert response.status_code == 500
    assert "credential-sentinel" not in response.text + caplog.text


async def test_readiness_times_out_without_blocking_event_loop(settings: Settings) -> None:
    resources = Resources(settings)
    resources.timeout = 0.02

    async def unavailable() -> None:
        await asyncio.Event().wait()

    resources.database.health = unavailable
    resources.redis.health = AsyncMock()
    resources.kafka.health = AsyncMock()
    try:
        assert await asyncio.wait_for(resources.ready(), timeout=0.5) is False
    finally:
        await resources.database.close()
        await resources.redis.close()
        await resources.kafka.close()


async def test_partial_startup_closes_all_resources(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    resources = SimpleNamespace(
        media=None,
        configuration=None,
        database=SimpleNamespace(close=AsyncMock()),
        redis=SimpleNamespace(close=AsyncMock()),
        kafka=SimpleNamespace(start=AsyncMock(side_effect=RuntimeError()), close=AsyncMock()),
    )
    monkeypatch.setattr(resource_module, "Resources", lambda configuration: resources)
    with pytest.raises(RuntimeError):
        async with open_resources(settings):
            pytest.fail("Failed startup must not yield resources")
    resources.database.close.assert_awaited_once()
    resources.redis.close.assert_awaited_once()
    resources.kafka.close.assert_awaited_once()
