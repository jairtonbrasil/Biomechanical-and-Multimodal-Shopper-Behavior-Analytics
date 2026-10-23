from typing import Any

from sqlalchemy import CheckConstraint, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, CreatedMixin, IdentityMixin, ValidatedJSONB
from app.schemas.model_version import ModelConfiguration


class ModelVersion(IdentityMixin, CreatedMixin, Base):
    __tablename__ = "model_version"
    __table_args__ = (
        UniqueConstraint("name", "kind", "version"),
        CheckConstraint(
            (
                "kind IN "
                "('POSE','FACE','TEMPORAL_RULES','ATTENTION','ACTION','BODY_REACTION','FACE_BODY_REACTION')"
            ),
            name="valid_kind",
        ),
        CheckConstraint("jsonb_typeof(configuration) = 'object'", name="configuration_object"),
        Index("ix_model_version_configuration_gin", "configuration", postgresql_using="gin"),
    )

    name: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(32))
    version: Mapped[str] = mapped_column(String(128))
    configuration: Mapped[dict[str, Any]] = mapped_column(ValidatedJSONB(ModelConfiguration))
