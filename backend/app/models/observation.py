from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, CreatedMixin, IdentityMixin, ValidatedJSONB
from app.schemas.observation import ObservationAttributes


class FrameObservation(IdentityMixin, CreatedMixin, Base):
    __tablename__ = "frame_observation"
    __table_args__ = (
        UniqueConstraint("camera_id", "tracking_session_id", "local_track_id", "frame_seq"),
        CheckConstraint(
            "frame_seq >= 0 AND length(local_track_id) > 0", name="valid_frame_identity"
        ),
        CheckConstraint(
            (
                "jsonb_typeof(attributes) = 'object' AND attributes ?& "
                "ARRAY['bbox','pose','head_pose','attention','hands','movement','face','provenance','geometry','presence','kinematics','spatial','evidence']"
            ),
            name="required_attribute_groups",
        ),
        Index("ix_frame_observation_camera_id_observed_at", "camera_id", "observed_at"),
        Index("ix_frame_observation_attention_shelf_id", "attention_shelf_id"),
        Index("ix_frame_observation_reach_shelf_id", "reach_shelf_id"),
        Index("ix_frame_observation_attributes_gin", "attributes", postgresql_using="gin"),
    )

    camera_id: Mapped[UUID] = mapped_column(ForeignKey("camera.id", ondelete="RESTRICT"))
    attention_shelf_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("shelf.id", ondelete="RESTRICT")
    )
    reach_shelf_id: Mapped[UUID | None] = mapped_column(ForeignKey("shelf.id", ondelete="RESTRICT"))
    tracking_session_id: Mapped[UUID]
    local_track_id: Mapped[str] = mapped_column(String(128))
    frame_seq: Mapped[int] = mapped_column(BigInteger)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    attributes: Mapped[dict[str, Any]] = mapped_column(ValidatedJSONB(ObservationAttributes))


class ObservationInteraction(CreatedMixin, Base):
    __tablename__ = "observation_interaction"
    __table_args__ = (
        CheckConstraint("link_type IN ('PRESENCE','ATTENTION','REACH')", name="valid_link_type"),
        Index("ix_observation_interaction_interaction_id", "interaction_id"),
    )

    observation_id: Mapped[UUID] = mapped_column(
        ForeignKey("frame_observation.id", ondelete="RESTRICT"), primary_key=True
    )
    interaction_id: Mapped[UUID] = mapped_column(
        ForeignKey("shelf_interaction.id", ondelete="RESTRICT"), primary_key=True
    )
    link_type: Mapped[str] = mapped_column(String(16), primary_key=True)
