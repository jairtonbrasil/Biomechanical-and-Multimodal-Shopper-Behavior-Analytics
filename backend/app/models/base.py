from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from app.schemas.common import Contract

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class ValidatedJSONB(TypeDecorator):
    impl = JSONB
    cache_ok = True

    def __init__(self, schema: type[Contract]) -> None:
        super().__init__()
        self.schema = schema

    def process_bind_param(self, value: Any, dialect: Any) -> dict[str, Any]:
        return self.schema.model_validate(value).model_dump(mode="json", by_alias=True)


class IdentityMixin:
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)


class CreatedMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UpdatedMixin(CreatedMixin):
    __mapper_args__ = {"eager_defaults": True}

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
