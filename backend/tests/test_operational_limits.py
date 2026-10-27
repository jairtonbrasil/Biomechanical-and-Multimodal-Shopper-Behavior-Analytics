import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request

from app.middleware.limits import OperationalLimits


class QuotaClient:
    def __init__(self, *, unavailable: bool = False) -> None:
        self.count = 0
        self.unavailable = unavailable

    async def eval(self, *arguments):
        if self.unavailable:
            raise RuntimeError("redis://private-credential@internal.invalid")
        self.count += 1
        return self.count


def limited_app(*, quota: int = 0, unavailable: bool = False, timeout: float = 1) -> FastAPI:
    application = FastAPI()
    application.state.settings = SimpleNamespace(
        api_request_max_bytes=1024,
        api_rate_limit_per_minute=quota,
        dependency_timeout_seconds=timeout,
        app_environment="local",
    )
    application.state.resources = SimpleNamespace(
        redis=SimpleNamespace(
            client=QuotaClient(unavailable=unavailable),
            key=lambda namespace, identifier: f"test:{namespace}:{identifier}",
        )
    )
    application.add_middleware(OperationalLimits)

    @application.post("/api/v1/echo")
    async def echo(request: Request):
        return {"length": len(await request.body())}

    @application.get("/api/v1/status")
    async def status():
        return {"status": "available"}

    @application.get("/health/live")
    async def health():
        return {"status": "alive"}

    return application


async def test_chunked_bodies_are_limited_without_trusting_content_length():
    application = limited_app()

    async def chunks():
        yield b"x" * 700
        yield b"x" * 700

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        rejected = await client.post("/api/v1/echo", content=chunks())
        accepted = await client.post("/api/v1/echo", content=b"x" * 1024)
    assert rejected.status_code == 413
    assert rejected.headers["cache-control"] == "no-store"
    assert accepted.json() == {"length": 1024}
    assert accepted.headers["x-content-type-options"] == "nosniff"


async def test_stalled_request_body_has_a_deadline():
    async def stalled():
        yield b"first"
        await asyncio.sleep(1)
        yield b"last"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=limited_app(timeout=0.01)),
        base_url="http://test",
    ) as client:
        response = await client.post("/api/v1/echo", content=stalled())
    assert response.status_code == 408
    assert response.json() == {"detail": "Request body deadline exceeded"}


async def test_shared_quota_rejects_requests_and_does_not_block_health():
    application = limited_app(quota=1)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        assert (await client.post("/api/v1/echo", content=b"one")).status_code == 200
        assert (await client.post("/api/v1/echo", content=b"two")).status_code == 429
        assert (await client.get("/health/live")).status_code == 200
    assert application.state.resources.redis.client.count == 2


async def test_quota_dependency_failure_is_closed_and_redacted():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=limited_app(quota=1, unavailable=True)),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/v1/status")
    assert response.status_code == 503
    assert response.json() == {"detail": "Request quota service is unavailable"}
    assert "private-credential" not in response.text and "internal.invalid" not in response.text


@pytest.mark.parametrize("authorization", [None, "Bearer private-credential", "Basic invalid"])
async def test_routes_do_not_require_authentication_or_echo_credentials(authorization):
    headers = {} if authorization is None else {"Authorization": authorization}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=limited_app()), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/status", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"status": "available"}
    assert "private-credential" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
