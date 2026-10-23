from datetime import datetime
from uuid import UUID

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdentityMixin, UpdatedMixin


class ShelfInteraction(IdentityMixin, UpdatedMixin, Base):
    __tablename__ = "shelf_interaction"
    __table_args__ = (
        CheckConstraint(
            "roi_version >= 1 AND presence_duration_ms >= 0", name="nonnegative_presence"
        ),
        CheckConstraint("ended_at IS NULL OR ended_at >= started_at", name="ordered_interval"),
        CheckConstraint(
            "last_presence_at >= started_at AND (ended_at IS NULL OR last_presence_at <= ended_at)",
            name="valid_presence_endpoint",
        ),
        CheckConstraint(
            (
                "(status = 'ACTIVE' AND ended_at IS NULL AND end_reason IS NULL) OR "
                "(status = 'CLOSED' AND ended_at IS NOT NULL AND end_reason IS NOT NULL)"
            ),
            name="valid_lifecycle",
        ),
        CheckConstraint(
            (
                "end_reason IS NULL OR end_reason IN "
                "('LEFT_ROI','TRACK_LOST','CAMERA_OFFLINE','ROI_CHANGED','REPROCESSING_GAP')"
            ),
            name="valid_close_reason",
        ),
        Index("ix_shelf_interaction_shelf_id_started_at", "shelf_id", "started_at"),
    )

    shelf_id: Mapped[UUID] = mapped_column(ForeignKey("shelf.id", ondelete="RESTRICT"))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_presence_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    presence_duration_ms: Mapped[int] = mapped_column(BigInteger)
    status: Mapped[str] = mapped_column(String(16))
    end_reason: Mapped[str | None] = mapped_column(String(32))
    roi_version: Mapped[int] = mapped_column(Integer)
