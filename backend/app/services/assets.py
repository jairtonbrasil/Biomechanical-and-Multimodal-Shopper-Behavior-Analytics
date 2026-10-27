from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.stream_policy import StreamPolicy
from app.domain.configuration import camera_revision, source_revision
from app.models import Camera, Shelf
from app.repositories.assets import CameraRepository, ShelfRepository
from app.repositories.errors import InvalidReferenceError, VersionConflictError
from app.schemas.assets import (
    CameraCreate,
    CameraUpdate,
    GeometryUpdate,
    ShelfCreate,
    ShelfUpdate,
)
from app.schemas.references import ReferenceFrame


class ReferenceStore(Protocol):
    async def current(self, camera: Camera) -> ReferenceFrame | None: ...

    async def get(self, camera: Camera, *, refresh: bool = False) -> ReferenceFrame: ...


class CameraService:
    def __init__(
        self, session: AsyncSession, policy: StreamPolicy, references: ReferenceStore
    ) -> None:
        self.session = session
        self.repository = CameraRepository(session)
        self.policy = policy
        self.references = references

    async def create(self, data: CameraCreate) -> Camera:
        self.policy.validate(data.stream_url)
        return await self.repository.create(data)

    async def get(self, identifier: UUID) -> Camera:
        return await self.repository.get(identifier)

    async def list(
        self,
        *,
        offset: int,
        limit: int,
        is_active: bool | None,
    ) -> list[Camera]:
        return await self.repository.list(
            offset=offset,
            limit=limit,
            is_active=is_active,
        )

    async def update(self, identifier: UUID, data: CameraUpdate) -> Camera:
        camera = await self.repository.get(identifier, lock=True)
        if camera_revision(camera) != data.expected_revision:
            raise VersionConflictError("Camera changed; reload its configuration revision")
        source_changed = (
            data.stream_url is not None and data.stream_url.get_secret_value() != camera.stream_url
        )
        if data.stream_url is not None:
            self.policy.validate(data.stream_url)
        if source_changed or data.is_active is False:
            if await self.repository.has_active_interactions(identifier):
                raise InvalidReferenceError("Close active interactions before changing the camera")
        if source_changed:
            for shelf in await self.repository.shelves(identifier, lock=True):
                shelf.is_active = False
        if data.name is not None:
            camera.name = data.name
        if data.stream_url is not None:
            camera.stream_url = data.stream_url.get_secret_value()
        if data.is_active is not None:
            camera.is_active = data.is_active
        camera.updated_at = datetime.now(UTC)
        await self.session.flush()
        return camera

    async def reference(self, identifier: UUID, *, refresh: bool = False) -> ReferenceFrame:
        camera = await self.get(identifier)
        if not camera.is_active:
            raise InvalidReferenceError("Activate the camera before capturing a reference")
        if refresh:
            if any(shelf.is_active for shelf in await self.repository.shelves(identifier)):
                raise InvalidReferenceError("Deactivate shelves before refreshing the reference")
            if await self.repository.has_active_interactions(identifier):
                raise InvalidReferenceError(
                    "Close active interactions before refreshing the reference"
                )
        # Capture publication rechecks source revision and shelf state independently.
        return await self.references.get(camera, refresh=refresh)


class ShelfService:
    def __init__(self, session: AsyncSession, references: ReferenceStore) -> None:
        self.session = session
        self.repository = ShelfRepository(session)
        self.cameras = CameraRepository(session)
        self.references = references

    async def _validate_reference(self, camera: Camera, revision: UUID) -> None:
        reference = await self.references.current(camera)
        if (
            reference is None
            or reference.revision != revision
            or reference.source_revision != source_revision(camera)
        ):
            raise InvalidReferenceError("Use the camera's current reference revision")

    async def create(self, data: ShelfCreate) -> Shelf:
        camera = await self.cameras.get(data.camera_id, lock=True)
        if data.is_active and not camera.is_active:
            raise InvalidReferenceError("An active shelf requires an active camera")
        await self._validate_reference(camera, data.product_roi.reference_revision)
        return await self.repository.create(data)

    async def get(self, identifier: UUID) -> Shelf:
        return await self.repository.get(identifier)

    async def list(
        self,
        *,
        offset: int,
        limit: int,
        camera_id: UUID | None,
        is_active: bool | None,
    ) -> list[Shelf]:
        return await self.repository.list(
            offset=offset,
            limit=limit,
            is_active=is_active,
            camera_ids=(camera_id,) if camera_id is not None else None,
        )

    async def update(self, identifier: UUID, data: ShelfUpdate) -> Shelf:
        initial = await self.get(identifier)
        # Every operational mutation locks camera before shelf, matching source cutovers.
        camera = await self.cameras.get(initial.camera_id, lock=True)
        shelf = await self.repository.get(identifier, lock=True)
        if shelf.roi_version != data.expected_version:
            raise VersionConflictError("Geometry changed; reload the current ROI version")
        if data.product_roi is not None and data.presence_roi is not None:
            await self._validate_reference(camera, data.product_roi.reference_revision)
            shelf = await self.repository.update_geometry(
                identifier,
                GeometryUpdate(
                    expected_version=data.expected_version,
                    product_roi=data.product_roi,
                    presence_roi=data.presence_roi,
                    cutover_at=datetime.now(UTC),
                ),
            )
        if data.is_active is True:
            if not camera.is_active:
                raise InvalidReferenceError("An active shelf requires an active camera")
            await self._validate_reference(
                camera, UUID(str(shelf.product_roi["reference_revision"]))
            )
        if data.is_active is False and await self.repository.has_active_interactions(identifier):
            raise InvalidReferenceError("Close active interactions before deactivating a shelf")
        for field in ("name", "category", "is_active"):
            if getattr(data, field) is not None:
                setattr(shelf, field, getattr(data, field))
        shelf.updated_at = datetime.now(UTC)
        await self.session.flush()
        return shelf
