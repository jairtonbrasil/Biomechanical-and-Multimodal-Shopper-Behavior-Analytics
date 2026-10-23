from datetime import datetime
from uuid import UUID

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class MetricKey:
    shelf_id: Mapped[UUID] = mapped_column(
        ForeignKey("shelf.id", ondelete="RESTRICT"), primary_key=True
    )
    granularity: Mapped[str] = mapped_column(String(32), primary_key=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ShelfMetric(MetricKey, Base):
    __tablename__ = "shelf_metric"
    __table_args__ = (
        CheckConstraint("window_end > window_start", name="ordered_window"),
        CheckConstraint(
            (
                "interactions_started >= 0 AND interactions_with_attention >= 0 "
                "AND attention_episodes >= 0 AND presence_duration_ms >= 0 AND "
                "attention_duration_ms >= 0 AND reach_count >= 0 AND touch_count >= 0 AND "
                "pickup_count >= 0 AND return_count >= 0"
            ),
            name="nonnegative_metrics",
        ),
        CheckConstraint("length(granularity) > 0", name="nonempty_granularity"),
    )

    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    interactions_started: Mapped[int] = mapped_column(Integer, server_default="0")
    interactions_with_attention: Mapped[int] = mapped_column(Integer, server_default="0")
    attention_episodes: Mapped[int] = mapped_column(Integer, server_default="0")
    presence_duration_ms: Mapped[int] = mapped_column(BigInteger, server_default="0")
    attention_duration_ms: Mapped[int] = mapped_column(BigInteger, server_default="0")
    reach_count: Mapped[int] = mapped_column(Integer, server_default="0")
    touch_count: Mapped[int] = mapped_column(Integer, server_default="0")
    pickup_count: Mapped[int] = mapped_column(Integer, server_default="0")
    return_count: Mapped[int] = mapped_column(Integer, server_default="0")


class ShelfSignalMetric(MetricKey, Base):
    __tablename__ = "shelf_signal_metric"
    __table_args__ = (
        CheckConstraint("signal_count >= 0", name="nonnegative_count"),
        CheckConstraint(
            "length(granularity) > 0 AND length(signal_type) > 0", name="nonempty_dimensions"
        ),
    )

    signal_type: Mapped[str] = mapped_column(String(64), primary_key=True)
    signal_count: Mapped[int] = mapped_column(Integer, server_default="0")
