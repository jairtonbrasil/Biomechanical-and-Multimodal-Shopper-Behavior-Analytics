import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import func, select

from app.core.config import Settings
from app.core.database import Database
from app.domain.configuration import source_revision
from app.main import create_app
from app.models import Camera, Shelf
from app.repositories.events import InteractionRepository
from app.schemas.events import InteractionCreate
from tests.factories import POLYGON

pytestmark = pytest.mark.integration
SENSITIVE_URL = "rtsp://private-user:private-password@192.0.2.10:8554/demo"


class FakeReferences:
    def __init__(self) -> None:
        self.frames: dict[UUID, SimpleNamespace] = {}
        self.fail = False

    async def current(self, camera: Camera) -> SimpleNamespace | None:
        frame = self.frames.get(camera.id)
        return (
            frame
            if camera.is_active
            and frame is not None
            and frame.source_revision == source_revision(camera)
            else None
        )

    async def get(self, camera: Camera, *, refresh: bool = False) -> SimpleNamespace:
        from app.core.references import ReferenceUnavailableError

        if self.fail:
            raise ReferenceUnavailableError("Reference is unavailable")
        if refresh or await self.current(camera) is None:
            self.frames[camera.id] = SimpleNamespace(
                content=b"test-jpeg",
                revision=uuid4(),
                source_revision=source_revision(camera),
                width=1280,
                height=720,
                captured_at=datetime.now(UTC),
            )
        return self.frames[camera.id]


class FakeConfiguration:
    def __init__(self) -> None:
        self.applied_revision: UUID | None = None

    async def status(self, configuration):
        if configuration.configuration_revision == self.applied_revision:
            return configuration.model_copy(
                update={"proxy_state": "APPLIED", "applied_at": datetime.now(UTC)}
            )
        return configuration


@pytest_asyncio.fixture
async def api(domain_database: Database):
    settings = Settings(
        app_environment="test",
        rtsp_allowed_networks=["192.0.2.0/24"],
    )
    application = create_app(settings)
    references = FakeReferences()
    application.state.settings = settings
    application.state.resources = SimpleNamespace(
        database=domain_database, references=references, configuration=FakeConfiguration()
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://test",
    ) as client:
        yield client, references, application


async def create_camera(client: httpx.AsyncClient, **changes) -> dict:
    response = await client.post(
        "/api/v1/cameras", json={"name": "Demo camera", "stream_url": SENSITIVE_URL, **changes}
    )
    assert response.status_code == 201, response.text
    return response.json()


async def create_shelf(client: httpx.AsyncClient, camera: dict, **changes) -> dict:
    reference = await client.get(f"/api/v1/cameras/{camera['id']}/frame-zero")
    assert reference.status_code == 200, reference.text
    response = await client.post(
        "/api/v1/shelves",
        json={
            "camera_id": camera["id"],
            "name": "Demo shelf",
            "category": "Drinks",
            "product_roi": {
                **POLYGON,
                "reference_revision": reference.headers["X-Reference-Revision"],
            },
            "presence_roi": POLYGON,
            **changes,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


async def test_camera_crud_redaction_version_and_persistence(api, domain_database):
    client, _, _ = api
    camera = await create_camera(client)
    assert "stream_url" not in camera
    assert "private-password" not in str(camera)
    listing = await client.get("/api/v1/cameras?limit=1&offset=0&is_active=true")
    assert listing.json() == [camera]
    assert (await client.get("/api/v1/cameras?limit=101")).status_code == 422
    changed = await client.put(
        f"/api/v1/cameras/{camera['id']}",
        json={"expected_revision": camera["configuration_revision"], "name": "Renamed camera"},
    )
    assert changed.status_code == 200
    assert changed.json()["source_revision"] == camera["source_revision"]
    assert changed.json()["configuration_revision"] != camera["configuration_revision"]
    stale = await client.put(
        f"/api/v1/cameras/{camera['id']}",
        json={"expected_revision": camera["configuration_revision"], "name": "Stale"},
    )
    assert stale.status_code == 409
    async with domain_database.transaction() as session:
        assert (await session.get(Camera, UUID(camera["id"]))).name == "Renamed camera"


async def test_reference_headers_refresh_fencing_and_unavailable(api):
    client, references, _ = api
    camera = await create_camera(client)
    path = f"/api/v1/cameras/{camera['id']}/frame-zero"
    first = await client.get(path)
    assert first.headers["content-type"] == "image/jpeg"
    assert first.headers["X-Frame-Width"] == "1280"
    assert first.headers["X-Frame-Height"] == "720"
    assert first.headers["X-Source-Revision"] == camera["source_revision"]
    assert (await client.get(path)).headers["ETag"] == first.headers["ETag"]
    refreshed = await client.post(path)
    assert refreshed.status_code == 200
    assert refreshed.headers["ETag"] != first.headers["ETag"]
    await create_shelf(client, camera)
    assert (await client.post(path)).status_code == 409
    references.fail = True
    assert (await client.get(path)).status_code == 503


async def test_shelf_geometry_history_stale_metadata_and_invalid_polygons(api, domain_database):
    client, _, _ = api
    camera = await create_camera(client)
    shelf = await create_shelf(client, camera)
    revision = shelf["product_roi"]["reference_revision"]
    changed = await client.put(
        f"/api/v1/shelves/{shelf['id']}",
        json={
            "expected_version": 1,
            "product_roi": {**POLYGON, "reference_revision": revision},
            "presence_roi": POLYGON,
        },
    )
    assert changed.status_code == 200, changed.text
    assert changed.json()["roi_version"] == 2
    archived = changed.json()["product_roi"]["history"]
    assert len(archived) == 1 and archived[0]["roi_version"] == 1
    assert archived[0]["valid_until"] is not None
    assert (
        await client.put(
            f"/api/v1/shelves/{shelf['id']}", json={"expected_version": 1, "name": "Stale"}
        )
    ).status_code == 409
    invalid = await client.post(
        "/api/v1/shelves",
        json={
            "camera_id": camera["id"],
            "name": "Invalid shelf",
            "category": "Drinks",
            "product_roi": {**POLYGON, "reference_revision": revision},
            "presence_roi": {**POLYGON, "points": [[0, 0], [0.5, 0.5], [1, 1]]},
        },
    )
    assert invalid.status_code == 422
    async with domain_database.transaction() as session:
        assert await session.scalar(select(func.count()).select_from(Shelf)) == 1


async def test_cross_camera_reference_unknown_ids_and_null_updates(api):
    client, _, _ = api
    first = await create_camera(client)
    second = await create_camera(client)
    shelf = await create_shelf(client, first)
    await client.get(f"/api/v1/cameras/{second['id']}/frame-zero")
    cross_camera = await client.post(
        "/api/v1/shelves",
        json={
            "camera_id": second["id"],
            "name": "Wrong reference",
            "category": "Drinks",
            "product_roi": {
                **POLYGON,
                "reference_revision": shelf["product_roi"]["reference_revision"],
            },
            "presence_roi": POLYGON,
        },
    )
    assert cross_camera.status_code == 409
    for resource in ("cameras", "shelves"):
        assert (await client.get(f"/api/v1/{resource}/{uuid4()}")).status_code == 404
    assert (
        await client.put(
            f"/api/v1/cameras/{first['id']}",
            json={"expected_revision": first["configuration_revision"], "stream_url": None},
        )
    ).status_code == 422
    assert (
        await client.put(
            f"/api/v1/shelves/{shelf['id']}",
            json={"expected_version": 1, "camera_id": second["id"]},
        )
    ).status_code == 422
    assert (
        await client.put(
            f"/api/v1/shelves/{shelf['id']}",
            json={
                "expected_version": 1,
                "product_roi": {
                    **POLYGON,
                    "reference_revision": shelf["product_roi"]["reference_revision"],
                },
            },
        )
    ).status_code == 422


async def test_source_cutover_deactivates_shelves_and_requires_new_reference(api):
    client, _, _ = api
    camera = await create_camera(client)
    shelf = await create_shelf(client, camera)
    changed = await client.put(
        f"/api/v1/cameras/{camera['id']}",
        json={
            "expected_revision": camera["configuration_revision"],
            "stream_url": "rtsp://192.0.2.10:8554/other-demo",
        },
    )
    assert changed.status_code == 200
    assert changed.json()["source_revision"] != camera["source_revision"]
    current = (await client.get(f"/api/v1/shelves/{shelf['id']}")).json()
    assert not current["is_active"]
    assert current["product_roi"] == shelf["product_roi"]
    assert (
        await client.put(
            f"/api/v1/shelves/{shelf['id']}", json={"expected_version": 1, "is_active": True}
        )
    ).status_code == 409


async def test_routes_work_without_user_credentials_and_redact_errors(api):
    client, _, _ = api
    assert "Authorization" not in client.headers
    camera = await create_camera(client)
    other = await create_camera(client, name="Other camera")
    shelf = await create_shelf(client, camera)
    cameras = (await client.get("/api/v1/cameras")).json()
    assert {row["id"] for row in cameras} == {camera["id"], other["id"]}
    assert (await client.get(f"/api/v1/cameras/{other['id']}")).status_code == 200
    filtered = (await client.get(f"/api/v1/shelves?camera_id={camera['id']}")).json()
    assert [row["id"] for row in filtered] == [shelf["id"]]
    assert (await client.get(f"/api/v1/shelves?camera_id={other['id']}")).json() == []
    assert (await client.get(f"/api/v1/shelves/{shelf['id']}")).status_code == 200
    assert (await client.get(f"/api/v1/cameras/{camera['id']}/runtime")).status_code == 200
    denied = await client.post(
        "/api/v1/cameras",
        json={"name": "Unsafe", "stream_url": "rtsp://user:secret@forbidden.example/live"},
    )
    assert denied.status_code == 422
    assert "secret" not in denied.text and "forbidden.example" not in denied.text


async def test_active_interactions_block_cutovers_without_partial_writes(api, domain_database):
    client, _, _ = api
    camera = await create_camera(client)
    shelf = await create_shelf(client, camera)
    async with domain_database.transaction() as session:
        row = await session.get(Shelf, UUID(shelf["id"]))
        await InteractionRepository(session).create(
            InteractionCreate(
                id=uuid4(),
                shelf_id=row.id,
                started_at=row.created_at,
                ended_at=None,
                last_presence_at=row.created_at,
                presence_duration_ms=0,
                status="ACTIVE",
                end_reason=None,
                roi_version=row.roi_version,
            )
        )
    blocked = await client.put(
        f"/api/v1/cameras/{camera['id']}",
        json={"expected_revision": camera["configuration_revision"], "is_active": False},
    )
    assert blocked.status_code == 409
    blocked = await client.put(
        f"/api/v1/shelves/{shelf['id']}",
        json={
            "expected_version": 1,
            "name": "Must roll back",
            "product_roi": {
                **POLYGON,
                "reference_revision": shelf["product_roi"]["reference_revision"],
            },
            "presence_roi": POLYGON,
        },
    )
    assert blocked.status_code == 409
    current = (await client.get(f"/api/v1/shelves/{shelf['id']}")).json()
    assert current["name"] == shelf["name"] and current["roi_version"] == 1
    assert (await client.get(f"/api/v1/cameras/{camera['id']}")).json()["is_active"]


async def test_concurrent_api_geometry_updates_accept_one_version(api):
    client, _, _ = api
    camera = await create_camera(client)
    shelf = await create_shelf(client, camera)
    data = {
        "expected_version": 1,
        "product_roi": {
            **POLYGON,
            "reference_revision": shelf["product_roi"]["reference_revision"],
        },
        "presence_roi": POLYGON,
    }
    responses = await asyncio.gather(
        client.put(f"/api/v1/shelves/{shelf['id']}", json=data),
        client.put(f"/api/v1/shelves/{shelf['id']}", json=data),
    )
    assert sorted(response.status_code for response in responses) == [200, 409]


async def test_runtime_descriptor_revisions_acknowledgements_and_inactive_camera(api):
    client, _, application = api
    camera = await create_camera(client)
    shelf = await create_shelf(client, camera)
    path = f"/api/v1/cameras/{camera['id']}/runtime"
    pending = await client.get(path)
    assert pending.status_code == 200
    configuration = pending.json()
    assert configuration["proxy_state"] == "PENDING" and configuration["applied_at"] is None
    assert configuration["proxy_path"] == f"cameras/{camera['id']}"
    assert configuration["source_revision"] == camera["source_revision"]
    assert configuration["reference_revision"] == shelf["product_roi"]["reference_revision"]
    assert configuration["shelves"][0]["id"] == shelf["id"]
    assert "stream_url" not in pending.text and "private-password" not in pending.text
    assert (
        await client.get(path, headers={"If-None-Match": pending.headers["ETag"]})
    ).status_code == 304
    synchronizer = application.state.resources.configuration
    synchronizer.applied_revision = UUID(configuration["configuration_revision"])
    applied = await client.get(path, headers={"If-None-Match": pending.headers["ETag"]})
    assert applied.status_code == 200
    assert applied.json()["proxy_state"] == "APPLIED"
    assert applied.json()["applied_at"] is not None
    assert applied.headers["ETag"] != pending.headers["ETag"]
    changed = await client.put(
        f"/api/v1/shelves/{shelf['id']}", json={"expected_version": 1, "name": "Updated shelf"}
    )
    assert changed.status_code == 200
    updated = await client.get(path, headers={"If-None-Match": applied.headers["ETag"]})
    assert updated.status_code == 200 and updated.json()["proxy_state"] == "PENDING"
    assert updated.json()["configuration_revision"] != configuration["configuration_revision"]
    assert updated.json()["shelves"][0]["name"] == "Updated shelf"
    deactivated = await client.put(
        f"/api/v1/cameras/{camera['id']}",
        json={"expected_revision": camera["configuration_revision"], "is_active": False},
    )
    assert deactivated.status_code == 200
    inactive = (await client.get(path)).json()
    assert not inactive["is_active"] and inactive["proxy_path"] is None
    assert inactive["reference_revision"] is None and inactive["shelves"] == []
    assert (await client.get(f"/api/v1/cameras/{camera['id']}/frame-zero")).status_code == 409
