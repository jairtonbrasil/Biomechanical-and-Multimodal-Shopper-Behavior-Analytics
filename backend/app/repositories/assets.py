from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Camera, Shelf, ShelfInteraction
from app.repositories.errors import EntityNotFoundError, InvalidReferenceError, VersionConflictError
from app.schemas.assets import CameraCreate, GeometryUpdate, ShelfCreate
from app.schemas.geometry import GeometrySnapshot, Polygon, StoredProductROI


class CameraRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, data: CameraCreate) -> Camera:
        camera = Camera(
            name=data.name, stream_url=data.stream_url.get_secret_value(), is_active=data.is_active
        )
        self.session.add(camera)
        await self.session.flush()
        return camera

    async def get(self, identifier: UUID) -> Camera:
        camera = await self.session.get(Camera, identifier)
        if camera is None:
            raise EntityNotFoundError("Camera does not exist")
        return camera

    async def set_active(self, identifier: UUID, active: bool) -> Camera:
        camera = await self.get(identifier)
        camera.is_active = active
        await self.session.flush()
        return camera

    async def list(self, *, offset: int = 0, limit: int = 100) -> list[Camera]:
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("Invalid pagination bounds")
        return list(
            await self.session.scalars(
                select(Camera).order_by(Camera.id).offset(offset).limit(limit)
            )
        )


class ShelfRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, identifier: UUID) -> Shelf:
        shelf = await self.session.get(Shelf, identifier)
        if shelf is None:
            raise EntityNotFoundError("Shelf does not exist")
        return shelf

    async def create(self, data: ShelfCreate) -> Shelf:
        await CameraRepository(self.session).get(data.camera_id)
        shelf = Shelf(
            camera_id=data.camera_id,
            name=data.name,
            category=data.category,
            product_roi=StoredProductROI(**data.product_roi.model_dump()).model_dump(mode="json"),
            presence_roi=data.presence_roi.model_dump(mode="json"),
            roi_version=1,
            is_active=data.is_active,
        )
        self.session.add(shelf)
        await self.session.flush()
        return shelf

    @staticmethod
    def snapshot(shelf: Shelf) -> GeometrySnapshot:
        product = StoredProductROI.model_validate(shelf.product_roi)
        return GeometrySnapshot(
            camera_id=shelf.camera_id,
            shelf_id=shelf.id,
            roi_version=shelf.roi_version,
            reference_revision=product.reference_revision,
            product_roi=Polygon.model_validate(
                product.model_dump(include={"type", "coordinate_system", "points"})
            ),
            presence_roi=Polygon.model_validate(shelf.presence_roi),
            valid_from=product.history[-1].valid_until if product.history else shelf.created_at,
            valid_until=None,
        )

    async def update_geometry(self, identifier: UUID, data: GeometryUpdate) -> Shelf:
        # Lock before reading history so concurrent editors cannot lose a snapshot.
        shelf = await self.session.scalar(
            select(Shelf)
            .where(Shelf.id == identifier)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if shelf is None:
            raise EntityNotFoundError("Shelf does not exist")
        if shelf.roi_version != data.expected_version:
            raise VersionConflictError("Geometry changed; reload the current ROI version")
        if await self.session.scalar(
            select(ShelfInteraction.id)
            .where(ShelfInteraction.shelf_id == identifier, ShelfInteraction.status == "ACTIVE")
            .limit(1)
        ):
            raise InvalidReferenceError("Close active interactions before the geometry cutover")
        previous = self.snapshot(shelf)
        if data.cutover_at <= previous.valid_from:
            raise ValueError("Cutover must follow the current geometry's effective start")
        current = StoredProductROI.model_validate(shelf.product_roi)
        archived = previous.model_copy(update={"valid_until": data.cutover_at})
        replacement = StoredProductROI(
            **data.product_roi.model_dump(), history=(*current.history, archived)
        )
        shelf.product_roi = replacement.model_dump(mode="json")
        shelf.presence_roi = data.presence_roi.model_dump(mode="json")
        shelf.roi_version += 1
        await self.session.flush()
        return shelf
