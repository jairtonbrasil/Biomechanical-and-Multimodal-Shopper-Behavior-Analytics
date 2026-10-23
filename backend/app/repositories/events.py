from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ActionEvent, AttentionEpisode, ReactionSignal, Shelf, ShelfInteraction
from app.repositories.errors import EntityNotFoundError, InvalidReferenceError
from app.repositories.model_versions import ModelVersionRepository
from app.schemas.common import ModelKind, ReactionSource
from app.schemas.events import ActionCreate, AttentionCreate, InteractionCreate, ReactionCreate
from app.schemas.geometry import StoredProductROI


class InteractionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, data: InteractionCreate) -> ShelfInteraction:
        shelf = await self.session.scalar(
            select(Shelf).where(Shelf.id == data.shelf_id).with_for_update()
        )
        if shelf is None:
            raise EntityNotFoundError("Shelf does not exist")
        history = StoredProductROI.model_validate(shelf.product_roi)
        if data.roi_version != shelf.roi_version and not any(
            item.roi_version == data.roi_version for item in history.history
        ):
            raise InvalidReferenceError("Interaction references an unknown geometry version")
        interaction = ShelfInteraction(**data.model_dump())
        self.session.add(interaction)
        await self.session.flush()
        return interaction


class EventRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def _parent(self, identifier: UUID) -> ShelfInteraction:
        parent = await self.session.get(ShelfInteraction, identifier)
        if parent is None:
            raise EntityNotFoundError("Interaction does not exist")
        return parent

    async def add_attention(self, data: AttentionCreate) -> AttentionEpisode:
        parent = await self._parent(data.interaction_id)
        if data.started_at < parent.started_at or (
            parent.ended_at is not None
            and (data.ended_at is None or data.ended_at > parent.ended_at)
        ):
            raise InvalidReferenceError("Attention lies outside its parent interaction")
        await ModelVersionRepository(self.session).require_kind(data.model_id, ModelKind.ATTENTION)
        event = AttentionEpisode(**data.model_dump())
        self.session.add(event)
        await self.session.flush()
        return event

    async def add_action(self, data: ActionCreate) -> ActionEvent:
        await self._event_time(data.interaction_id, data.occurred_at)
        await ModelVersionRepository(self.session).require_kind(data.model_id, ModelKind.ACTION)
        event = ActionEvent(**data.model_dump())
        self.session.add(event)
        await self.session.flush()
        return event

    async def add_reaction(self, data: ReactionCreate) -> ReactionSignal:
        await self._event_time(data.interaction_id, data.occurred_at)
        expected = (
            ModelKind.FACE_BODY_REACTION
            if data.source == ReactionSource.FACE_AND_BODY
            else ModelKind.BODY_REACTION
        )
        await ModelVersionRepository(self.session).require_kind(data.model_id, expected)
        event = ReactionSignal(**data.model_dump())
        self.session.add(event)
        await self.session.flush()
        return event

    async def _event_time(self, identifier: UUID, instant: datetime) -> None:
        parent = await self._parent(identifier)
        if instant < parent.started_at or (
            parent.ended_at is not None and instant > parent.ended_at
        ):
            raise InvalidReferenceError("Event time lies outside its parent interaction")
