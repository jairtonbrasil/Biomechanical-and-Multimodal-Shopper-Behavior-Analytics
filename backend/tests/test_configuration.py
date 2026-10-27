import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.media import MediaMTXControl, MediaUnavailableError
from app.domain.configuration import camera_revision, proxy_path, source_revision
from app.models import Camera
from app.services.configuration import ConfigurationSynchronizer, configuration_snapshot


@pytest.fixture
def media_settings(settings):
    return settings.model_copy(
        update={"rtsp_allowed_networks": ("192.0.2.0/24",), "rtsp_allowed_ports": (8554,)}
    )


@pytest.fixture
def configured_camera():
    return Camera(
        id=uuid4(),
        name="Configured camera",
        stream_url="rtsp://sensitive-user:sensitive-password@192.0.2.10:8554/live",
        is_active=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


async def test_media_configuration_creates_patches_and_preserves_matching_paths(
    media_settings, configured_camera
):
    recorded = []
    current = None

    def handle(request):
        recorded.append(request)
        if request.method == "GET":
            return httpx.Response(404) if current is None else httpx.Response(200, json=current)
        return httpx.Response(200)

    media = MediaMTXControl(media_settings, transport=httpx.MockTransport(handle))
    try:
        await media.apply(configured_camera)
        assert recorded[-1].method == "POST"
        desired = json.loads(recorded[-1].content)
        assert desired["source"] == configured_camera.stream_url
        current = {**desired, "source": "rtsp://192.0.2.20:8554/old"}
        await media.apply(configured_camera)
        assert recorded[-1].method == "PATCH"
        current = desired
        recorded.clear()
        await media.apply(configured_camera)
        assert [request.method for request in recorded] == ["GET"]
    finally:
        await media.close()


async def test_narrowed_stream_allowlist_removes_the_old_proxy(media_settings, configured_camera):
    recorded = []

    def handle(request):
        recorded.append(request)
        return httpx.Response(200)

    restricted = media_settings.model_copy(update={"rtsp_allowed_networks": ()})
    media = MediaMTXControl(restricted, transport=httpx.MockTransport(handle))
    try:
        with pytest.raises(MediaUnavailableError) as result:
            await media.apply(configured_camera)
        assert "sensitive" not in str(result.value)
        assert len(recorded) == 1 and recorded[0].method == "DELETE"
        assert recorded[0].url.path.endswith(proxy_path(configured_camera.id))
    finally:
        await media.close()


@pytest.mark.parametrize("document", [None, [], "sensitive-password", 12])
async def test_media_path_response_requires_a_mapping(media_settings, configured_camera, document):
    media = MediaMTXControl(
        media_settings, transport=httpx.MockTransport(lambda _: httpx.Response(200, json=document))
    )
    try:
        with pytest.raises(MediaUnavailableError) as result:
            await media.apply(configured_camera)
        assert "sensitive" not in str(result.value)
    finally:
        await media.close()


@pytest.mark.parametrize("status_code", [301, 401, 500])
async def test_media_control_errors_are_generic_and_do_not_follow_redirects(
    media_settings, configured_camera, status_code
):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            status_code, text="sensitive-password", headers={"Location": "http://other.invalid"}
        )

    media = MediaMTXControl(media_settings, transport=httpx.MockTransport(handle))
    try:
        with pytest.raises(MediaUnavailableError) as result:
            await media.apply(configured_camera)
        assert "sensitive" not in str(result.value)
        assert len(requests) == 1
    finally:
        await media.close()


async def test_media_response_size_is_bounded(media_settings):
    media = MediaMTXControl(
        media_settings,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=b"x" * (MediaMTXControl.MAX_RESPONSE_BYTES + 1))
        ),
    )
    try:
        with pytest.raises(MediaUnavailableError, match="size limit"):
            await media.remove_orphans(set())
    finally:
        await media.close()


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"items": [], "pageCount": True},
        {"items": {}, "pageCount": 1},
        {"items": [], "pageCount": -1},
        {"items": [], "pageCount": 101},
        {"items": [None], "pageCount": 1},
        {"items": [{"name": 12}], "pageCount": 1},
        {"items": [{"name": "x"}] * 101, "pageCount": 1},
    ],
)
async def test_media_list_response_has_bounded_validated_pagination(media_settings, document):
    media = MediaMTXControl(
        media_settings, transport=httpx.MockTransport(lambda _: httpx.Response(200, json=document))
    )
    try:
        with pytest.raises(MediaUnavailableError):
            await media.remove_orphans(set())
    finally:
        await media.close()


async def test_orphan_deletion_occurs_after_reading_all_pages(media_settings):
    first, second, retained = uuid4(), uuid4(), uuid4()
    calls = []

    def handle(request):
        calls.append((request.method, request.url.path))
        if request.method == "GET":
            identifiers = (first, retained) if request.url.params["page"] == "0" else (second,)
            return httpx.Response(
                200,
                json={
                    "pageCount": 2,
                    "items": [{"name": proxy_path(item)} for item in identifiers],
                },
            )
        return httpx.Response(200)

    media = MediaMTXControl(media_settings, transport=httpx.MockTransport(handle))
    try:
        await media.remove_orphans({retained})
        assert [method for method, _ in calls] == ["GET", "GET", "DELETE", "DELETE"]
        assert {path.rsplit("/", 1)[-1] for _, path in calls[2:]} == {str(first), str(second)}
    finally:
        await media.close()


def synchronizer(settings, camera):
    state = {}

    async def set_value(key, value, **kwargs):
        state[key] = value

    async def delete_values(*keys):
        for key in keys:
            state.pop(key, None)

    redis = SimpleNamespace(
        key=lambda namespace, value: f"test:{namespace}:{value}",
        client=SimpleNamespace(
            set=AsyncMock(side_effect=set_value),
            delete=AsyncMock(side_effect=delete_values),
            get=AsyncMock(side_effect=lambda key: state.get(key)),
        ),
    )
    session = SimpleNamespace(
        scalar=AsyncMock(return_value=True),
        scalars=AsyncMock(side_effect=[[camera.id], [], []]),
        get=AsyncMock(return_value=camera),
        refresh=AsyncMock(),
    )

    @asynccontextmanager
    async def transaction():
        yield session

    media = SimpleNamespace(apply=AsyncMock(), remove_orphans=AsyncMock())
    references = SimpleNamespace(current=AsyncMock(return_value=None))
    service = ConfigurationSynchronizer(
        settings, SimpleNamespace(transaction=transaction), redis, references, media
    )
    return service, session, state


async def test_successful_reconciliation_acknowledges_the_exact_committed_revision(
    settings, configured_camera
):
    service, _, state = synchronizer(settings, configured_camera)
    await service.reconcile_once()
    configuration = await configuration_snapshot(configured_camera, [], service.references)
    applied = await service.status(configuration)
    assert applied.proxy_state == "APPLIED" and applied.applied_at.tzinfo is not None
    acknowledgement = json.loads(next(iter(state.values())))
    assert acknowledgement["camera_revision"] == str(camera_revision(configured_camera))
    assert acknowledgement["source_revision"] == str(source_revision(configured_camera))
    service.redis.client.set.assert_awaited_once()
    assert service.redis.client.set.call_args.kwargs["ex"] >= 15


async def test_failed_reconciliation_revokes_the_previous_acknowledgement(
    settings, configured_camera
):
    service, _, state = synchronizer(settings, configured_camera)
    state[service.redis.key("state", f"configuration:{configured_camera.id}")] = "previous"
    service.media.apply.side_effect = MediaUnavailableError("Media control is unavailable")
    await service.reconcile_once()
    assert state == {}
    service.redis.client.set.assert_not_awaited()


async def test_superseded_camera_revision_is_not_acknowledged(settings, configured_camera):
    service, session, state = synchronizer(settings, configured_camera)

    async def concurrent_edit(camera):
        camera.updated_at += timedelta(seconds=1)

    session.refresh.side_effect = concurrent_edit
    await service.reconcile_once()
    assert state == {}
    service.redis.client.set.assert_not_awaited()


async def test_orphan_reconciliation_failure_revokes_known_camera_acknowledgements(
    settings, configured_camera
):
    service, _, state = synchronizer(settings, configured_camera)
    state[service.redis.key("state", f"configuration:{configured_camera.id}")] = "previous"
    service.media.remove_orphans.side_effect = MediaUnavailableError("Media control is unavailable")
    with pytest.raises(MediaUnavailableError):
        await service.reconcile_once()
    assert state == {}
    service.media.apply.assert_not_awaited()


@pytest.mark.parametrize("value", [None, "[]", "null", "invalid", '{"status":"pending"}'])
async def test_missing_and_invalid_acknowledgements_remain_pending(
    settings, configured_camera, value
):
    service, _, _ = synchronizer(settings, configured_camera)
    service.redis.client.get = AsyncMock(return_value=value)
    configuration = await configuration_snapshot(configured_camera, [], service.references)
    assert (await service.status(configuration)).proxy_state == "PENDING"


async def test_redis_failure_reports_pending(settings, configured_camera):
    service, _, _ = synchronizer(settings, configured_camera)
    service.redis.client.get = AsyncMock(side_effect=RedisConnectionError("Unavailable"))
    configuration = await configuration_snapshot(configured_camera, [], service.references)
    assert (await service.status(configuration)).proxy_state == "PENDING"


@pytest.mark.parametrize(
    "changed_field", ["source_revision", "configuration_revision", "applied_at"]
)
async def test_acknowledgement_scope_and_timestamp_must_match(
    settings, configured_camera, changed_field
):
    service, _, state = synchronizer(settings, configured_camera)
    await service.reconcile_once()
    key = next(iter(state))
    acknowledgement = json.loads(state[key])
    acknowledgement[changed_field] = (
        "2026-10-26T12:00:00" if changed_field == "applied_at" else str(uuid4())
    )
    state[key] = json.dumps(acknowledgement)
    configuration = await configuration_snapshot(configured_camera, [], service.references)
    assert (await service.status(configuration)).proxy_state == "PENDING"
