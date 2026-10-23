from datetime import datetime
from uuid import UUID

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.identity import TrackScope, observation_id
from app.models import FrameObservation, ObservationInteraction, Shelf, ShelfInteraction
from app.repositories.assets import ShelfRepository
from app.repositories.errors import InvalidReferenceError, VersionConflictError
from app.repositories.model_versions import ModelVersionRepository
from app.schemas.common import LinkType, ModelKind
from app.schemas.geometry import GeometrySnapshot, StoredProductROI
from app.schemas.observation import ObservationCreate


class ObservationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, data: ObservationCreate) -> FrameObservation:
        scope = TrackScope(data.camera_id, data.tracking_session_id, data.local_track_id)
        if data.id != observation_id(scope, data.frame_seq):
            raise InvalidReferenceError("Observation identity does not match its scoped frame")
        await self.validate_provenance(data)
        values = data.model_dump()
        values["attributes"] = data.attributes.model_dump(mode="json", by_alias=True)
        await self.session.execute(
            insert(FrameObservation).values(**values).on_conflict_do_nothing(index_elements=["id"])
        )
        observation = await self.session.get(FrameObservation, data.id)
        persisted = {name: getattr(observation, name) for name in ObservationCreate.model_fields}
        if ObservationCreate.model_validate(persisted) != data:
            raise VersionConflictError("A stable observation ID was reused with different content")
        return observation

    async def validate_provenance(self, data: ObservationCreate) -> None:
        versions = ModelVersionRepository(self.session)
        provenance = data.attributes.provenance
        await versions.require_kind(provenance.pose_model_id, ModelKind.POSE)
        await versions.require_kind(provenance.temporal_rules_id, ModelKind.TEMPORAL_RULES)
        if provenance.face_model_id:
            await versions.require_kind(provenance.face_model_id, ModelKind.FACE)
        for snapshot in data.attributes.geometry:
            shelf = await self.session.get(Shelf, snapshot.shelf_id)
            if shelf is None or shelf.camera_id != data.camera_id:
                raise InvalidReferenceError("Geometry belongs to an absent or different camera")
            await self.require_geometry(shelf, snapshot, data.observed_at)

    async def require_geometry(
        self, shelf: Shelf, supplied: GeometrySnapshot, observed_at: datetime
    ) -> None:
        product = StoredProductROI.model_validate(shelf.product_roi)
        current = ShelfRepository.snapshot(shelf)
        expected = next(
            (
                item
                for item in (*product.history, current)
                if item.roi_version == supplied.roi_version
            ),
            None,
        )
        # A previously emitted current snapshot remains valid after its interval is closed.
        if expected is None or supplied.model_dump(exclude={"valid_until"}) != expected.model_dump(
            exclude={"valid_until"}
        ):
            raise InvalidReferenceError("Observation geometry differs from the registered snapshot")
        if observed_at < expected.valid_from or (
            expected.valid_until is not None and observed_at >= expected.valid_until
        ):
            raise InvalidReferenceError("Observation is outside the registered geometry interval")

    async def link(
        self, observation_id: UUID, interaction_id: UUID, kind: LinkType
    ) -> ObservationInteraction:
        observation = await self.session.get(FrameObservation, observation_id)
        interaction = await self.session.get(ShelfInteraction, interaction_id)
        if observation is None or interaction is None:
            raise InvalidReferenceError("Observation and interaction must exist before linking")
        shelf = await self.session.get(Shelf, interaction.shelf_id)
        if shelf.camera_id != observation.camera_id:
            raise InvalidReferenceError("Observation and interaction belong to different cameras")
        attributes = ObservationCreate.model_validate(
            {name: getattr(observation, name) for name in ObservationCreate.model_fields}
        ).attributes
        if not any(
            item.shelf_id == shelf.id and item.roi_version == interaction.roi_version
            for item in attributes.geometry
        ):
            raise InvalidReferenceError("Link geometry version differs from the interaction")
        if kind == LinkType.ATTENTION and observation.attention_shelf_id != shelf.id:
            raise InvalidReferenceError("Attention link requires the observed attention target")
        if kind == LinkType.REACH and observation.reach_shelf_id != shelf.id:
            raise InvalidReferenceError("Reach link requires the observed reach target")
        if kind == LinkType.PRESENCE and not any(
            item.shelf_id == shelf.id and item.in_roi is True for item in attributes.presence
        ):
            raise InvalidReferenceError("Presence link requires observed ROI membership")
        link = ObservationInteraction(
            observation_id=observation_id, interaction_id=interaction_id, link_type=kind
        )
        self.session.add(link)
        await self.session.flush()
        return link
