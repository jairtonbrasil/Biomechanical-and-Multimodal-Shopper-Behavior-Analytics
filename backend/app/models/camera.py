from sqlalchemy import Boolean, String, Text, true
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdentityMixin, UpdatedMixin


class Camera(IdentityMixin, UpdatedMixin, Base):
    __tablename__ = "camera"

    name: Mapped[str] = mapped_column(String(128))
    stream_url: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=true())
