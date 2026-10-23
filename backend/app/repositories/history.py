from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, model_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import FrameObservation, Shelf, ShelfInteraction
from app.repositories.observations import ObservationRepository
from app.schemas.common import Contract, TimeWindow
from app.schemas.observation import ObservationCreate


class HistoryPurpose(StrEnum):
    REPLAY = "REPLAY"
    UNIQUE_TRACKS = "UNIQUE_TRACKS"
    REAGGREGATION = "REAGGREGATION"


class VerifiedCoverage(TimeWindow):
    camera_id: UUID
    source: Literal["RAW", "BUSINESS"]
    verified_at: AwareDatetime
    linked_observations_complete: bool = False

    @model_validator(mode="after")
    def verified_window(self) -> Self:
        if self.verified_at < self.end:
            raise ValueError("Coverage verification must follow the certified interval")
        return self


class HistoryEligibility(Contract):
    eligible: bool
    reason: str
    required_start: AwareDatetime


class HistoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def assess(
        self,
        camera_id: UUID,
        window: TimeWindow,
        purpose: HistoryPurpose,
        coverage: VerifiedCoverage | None,
    ) -> HistoryEligibility:
        required_start = window.start
        if purpose == HistoryPurpose.REPLAY:
            active_start = await self.session.scalar(
                select(func.min(ShelfInteraction.started_at))
                .join(Shelf)
                .where(
                    Shelf.camera_id == camera_id,
                    ShelfInteraction.status == "ACTIVE",
                    ShelfInteraction.started_at < window.end,
                )
            )
            if active_start is not None:
                required_start = min(required_start, active_start)
        source = "BUSINESS" if purpose == HistoryPurpose.REAGGREGATION else "RAW"
        reason = "Verified retained history covers the requested operation"
        if coverage is None:
            reason = (
                "No verified coverage manifest; the earliest remaining row "
                "cannot prove completeness"
            )
        elif coverage.camera_id != camera_id or coverage.source != source:
            reason = "Coverage belongs to a different camera or history source"
        elif coverage.start > required_start or coverage.end < window.end:
            reason = "Required history has expired or extends beyond verified coverage"
        elif purpose == HistoryPurpose.UNIQUE_TRACKS and not coverage.linked_observations_complete:
            reason = "Unique-track queries require verified observation-link coverage"
        else:
            if source == "RAW":
                observations = await self.session.stream_scalars(
                    select(FrameObservation)
                    .where(
                        FrameObservation.camera_id == camera_id,
                        FrameObservation.observed_at >= required_start,
                        FrameObservation.observed_at < window.end,
                    )
                    .order_by(FrameObservation.observed_at)
                    .execution_options(yield_per=100)
                )
                repository = ObservationRepository(self.session)
                try:
                    async for observation in observations:
                        data = ObservationCreate.model_validate(
                            {
                                name: getattr(observation, name)
                                for name in ObservationCreate.model_fields
                            }
                        )
                        await repository.validate_provenance(data)
                except ValueError:
                    return HistoryEligibility(
                        eligible=False,
                        reason="Retained provenance cannot reconstruct the requested scope",
                        required_start=required_start,
                    )
                finally:
                    await observations.close()
            return HistoryEligibility(eligible=True, reason=reason, required_start=required_start)
        return HistoryEligibility(eligible=False, reason=reason, required_start=required_start)
