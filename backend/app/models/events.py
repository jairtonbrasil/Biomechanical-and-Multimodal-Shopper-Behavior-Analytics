from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import REAL, CheckConstraint, DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, CreatedMixin, IdentityMixin, ValidatedJSONB
from app.schemas.events import ActionAttributes, ReactionAttributes


class AttentionEpisode(IdentityMixin, CreatedMixin, Base):
    __tablename__ = "attention_episode"
    __table_args__ = (
        CheckConstraint("ended_at IS NULL OR ended_at >= started_at", name="ordered_interval"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="valid_confidence"),
        Index("ix_attention_episode_interaction_id_started_at", "interaction_id", "started_at"),
    )

    interaction_id: Mapped[UUID] = mapped_column(
        ForeignKey("shelf_interaction.id", ondelete="RESTRICT")
    )
    model_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_version.id", ondelete="RESTRICT"), index=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    method: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(REAL)


class ActionEvent(IdentityMixin, CreatedMixin, Base):
    __tablename__ = "action_event"
    __table_args__ = (
        CheckConstraint(
            "action_type IN ('REACH','INSPECT','PICKUP','RETURN')", name="valid_action_type"
        ),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="valid_confidence"),
        CheckConstraint(
            (
                "jsonb_typeof(attributes) = 'object' AND attributes ?& "
                "ARRAY['hand','detection_method','object','interaction','temporal_evidence']"
            ),
            name="required_attribute_groups",
        ),
        Index(
            "ix_action_event_interaction_id_action_type_occurred_at",
            "interaction_id",
            "action_type",
            "occurred_at",
        ),
        Index("ix_action_event_attributes_gin", "attributes", postgresql_using="gin"),
    )

    interaction_id: Mapped[UUID] = mapped_column(
        ForeignKey("shelf_interaction.id", ondelete="RESTRICT")
    )
    model_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_version.id", ondelete="RESTRICT"), index=True
    )
    action_type: Mapped[str] = mapped_column(String(16))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    confidence: Mapped[float] = mapped_column(REAL)
    attributes: Mapped[dict[str, Any]] = mapped_column(ValidatedJSONB(ActionAttributes))


class ReactionSignal(IdentityMixin, CreatedMixin, Base):
    __tablename__ = "reaction_signal"
    __table_args__ = (
        CheckConstraint(
            "source IN ('FACE_AND_BODY','BODY_BIOMECHANICS')", name="valid_reaction_source"
        ),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="valid_confidence"),
        CheckConstraint(
            (
                "jsonb_typeof(attributes) = 'object' AND attributes ?& "
                "ARRAY['facial_analysis','behavioral_context','temporal_evidence','association','inferred_emotion']"
            ),
            name="required_attribute_groups",
        ),
        Index(
            "ix_reaction_signal_interaction_id_signal_type_occurred_at",
            "interaction_id",
            "signal_type",
            "occurred_at",
        ),
        Index("ix_reaction_signal_attributes_gin", "attributes", postgresql_using="gin"),
    )

    interaction_id: Mapped[UUID] = mapped_column(
        ForeignKey("shelf_interaction.id", ondelete="RESTRICT")
    )
    model_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_version.id", ondelete="RESTRICT"), index=True
    )
    signal_type: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(32))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    confidence: Mapped[float] = mapped_column(REAL)
    attributes: Mapped[dict[str, Any]] = mapped_column(ValidatedJSONB(ReactionAttributes))
