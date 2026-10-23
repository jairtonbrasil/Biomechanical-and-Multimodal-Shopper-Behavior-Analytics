from typing import Any
from uuid import UUID

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Index, Integer, String, true
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdentityMixin, UpdatedMixin, ValidatedJSONB
from app.schemas.geometry import Polygon, StoredProductROI


class Shelf(IdentityMixin, UpdatedMixin, Base):
    __tablename__ = "shelf"
    __table_args__ = (
        CheckConstraint("roi_version >= 1", name="positive_roi_version"),
        CheckConstraint(
            "jsonb_typeof(product_roi) = 'object' AND jsonb_typeof(presence_roi) = 'object'",
            name="roi_objects",
        ),
        Index("ix_shelf_product_roi_gin", "product_roi", postgresql_using="gin"),
        Index("ix_shelf_presence_roi_gin", "presence_roi", postgresql_using="gin"),
    )

    camera_id: Mapped[UUID] = mapped_column(
        ForeignKey("camera.id", ondelete="RESTRICT"), index=True
    )
    name: Mapped[str] = mapped_column(String(128))
    category: Mapped[str] = mapped_column(String(128))
    product_roi: Mapped[dict[str, Any]] = mapped_column(ValidatedJSONB(StoredProductROI))
    presence_roi: Mapped[dict[str, Any]] = mapped_column(ValidatedJSONB(Polygon))
    roi_version: Mapped[int] = mapped_column(Integer, server_default="1")
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=true())
