from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from app.core.database import get_db
from app.repositories.errors import EntityNotFoundError
from app.routers import playback


class FakeSession:
    def __init__(self, camera):
        self.camera = camera

    async def scalar(self, statement):
        return self.camera


@pytest.fixture
def player():
    camera_id = uuid4()
    camera = SimpleNamespace(
        id=camera_id,
        is_active=True,
        stream_url="rtsp://private-user:private-password@camera.invalid",
    )
    app = FastAPI()
    app.include_router(playback.router)
    app.state.settings = SimpleNamespace(
        mediamtx_hls_url="http://mediamtx:8888",
        mediamtx_read_password=SecretStr("private-read-password"),
        dependency_timeout_seconds=1,
    )
    session = FakeSession(camera)

    async def database():
        yield session

    app.dependency_overrides[get_db] = database
    return app, camera, session


async def test_playback_descriptor_and_player_need_no_application_credentials(player):
    app, camera, _ = player
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        descriptor = await client.get(f"/api/v1/cameras/{camera.id}/playback")
        page = await client.get(f"/api/v1/cameras/{camera.id}/player")
    assert descriptor.status_code == page.status_code == 200
    assert descriptor.json()["status"] == "CONFIGURED"
    assert descriptor.json()["content"] == "SOURCE_VIDEO"
    assert descriptor.json()["hls_url"] == f"/api/v1/cameras/{camera.id}/hls/index.m3u8"
    assert descriptor.json()["player_url"] == f"/api/v1/cameras/{camera.id}/player"
    assert "/static/hls.min.js" in page.text
    assert "private-password" not in page.text + descriptor.text
    assert "camera.invalid" not in page.text + descriptor.text


async def test_inactive_camera_cannot_access_playback_routes(player):
    app, camera, _ = player
    camera.is_active = False
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for suffix in ("playback", "player", "hls/index.m3u8"):
            assert (await client.get(f"/api/v1/cameras/{camera.id}/{suffix}")).status_code == 409


async def test_missing_camera_uses_repository_not_found_contract(player):
    app, camera, session = player
    session.camera = None
    with pytest.raises(EntityNotFoundError):
        await playback.playback_descriptor(camera.id, session)


def install_upstream(monkeypatch, handler):
    original = httpx.AsyncClient
    calls = []

    def factory(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(playback.httpx, "AsyncClient", factory)
    return original, calls


async def test_hls_bridge_uses_fixed_internal_origin_and_safe_media_credentials(
    player, monkeypatch
):
    app, camera, session = player
    received = []

    def source(request):
        received.append(request)
        return httpx.Response(
            200, content=b'#EXTM3U\n#EXT-X-MAP:URI="init.mp4"\n#EXTINF:1,\nseg0.mp4\n'
        )

    client_class, calls = install_upstream(monkeypatch, source)
    async with client_class(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            f"/api/v1/cameras/{camera.id}/hls/stream.m3u8?_HLS_msn=2&_HLS_skip=YES",
            headers={"Authorization": "Bearer browser-secret", "Cookie": "browser-private-cookie"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/vnd.apple.mpegurl")
    assert response.headers["cache-control"] == "private, no-store"
    assert received[0].url.host == "mediamtx"
    assert received[0].url.path == f"/cameras/{camera.id}/stream.m3u8"
    assert received[0].url.params["_HLS_msn"] == "2"
    assert "browser-secret" not in received[0].headers["authorization"]
    assert calls[0]["follow_redirects"] is False and calls[0]["trust_env"] is False
    assert calls[0]["headers"]["Authorization"] == "Bearer private-read-password"
    assert "auth" not in calls[0]
    assert "cookie" not in received[0].headers
    assert "set-cookie" not in response.headers
    assert "private-read-password" not in response.text


@pytest.mark.parametrize(
    "playlist",
    [
        b"not-a-playlist",
        b"#EXTM3U\nhttp://private:secret@camera.invalid/live\n",
        b"#EXTM3U\n../other/stream.m3u8\n",
        b'#EXTM3U\n#EXT-X-MAP:URI="/other/init.mp4"\n',
        b"#EXTM3U\nsegment.mp4?token=secret\n",
        b"#EXTM3U\n\xff\n",
    ],
)
async def test_unsafe_or_invalid_upstream_playlists_fail_without_echoing_source(
    player, monkeypatch, playlist
):
    app, camera, _ = player
    client_class, _ = install_upstream(
        monkeypatch, lambda request: httpx.Response(200, content=playlist)
    )
    async with client_class(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/v1/cameras/{camera.id}/hls/index.m3u8")
    assert response.status_code == 503
    assert response.json() == {"detail": "Camera playback is unavailable"}
    assert "secret" not in response.text and "camera.invalid" not in response.text


@pytest.mark.parametrize("status", [301, 302, 401, 404, 500])
async def test_upstream_failures_and_redirects_are_unavailable(player, monkeypatch, status):
    app, camera, _ = player
    client_class, _ = install_upstream(
        monkeypatch,
        lambda request: httpx.Response(
            status, headers={"Location": "http://private.invalid"}, content=b"private-source"
        ),
    )
    async with client_class(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/v1/cameras/{camera.id}/hls/index.m3u8")
    assert response.status_code == 503 and "private" not in response.text


@pytest.mark.parametrize(
    "query",
    [
        "token=secret",
        "_HLS_msn=-1",
        "_HLS_msn=1&_HLS_msn=2",
        "_HLS_skip=no",
        "_HLS_part=1000000000000000",
    ],
)
async def test_playback_rejects_unbounded_or_unknown_query_parameters(player, query):
    app, camera, _ = player
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/v1/cameras/{camera.id}/hls/stream.m3u8?{query}")
    assert response.status_code == 422 and "secret" not in response.text


@pytest.mark.parametrize("asset", ["private.txt", "..mp4", "a..mp4", "index.m3u8.exe"])
async def test_playback_rejects_invalid_assets_before_media_request(player, asset):
    app, camera, _ = player
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/v1/cameras/{camera.id}/hls/{asset}")
    assert response.status_code == 422


async def test_media_segments_have_bounded_bytes_and_safe_content_type(player, monkeypatch):
    app, camera, _ = player
    monkeypatch.setattr(playback, "MAX_MEDIA_BYTES", 4)
    client_class, _ = install_upstream(
        monkeypatch, lambda request: httpx.Response(200, content=b"12345")
    )
    async with client_class(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/v1/cameras/{camera.id}/hls/seg0.mp4")
    assert response.status_code == 503
