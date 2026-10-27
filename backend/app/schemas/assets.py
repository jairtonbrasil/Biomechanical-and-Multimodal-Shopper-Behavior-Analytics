from typing import Any, Self
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import AwareDatetime, Field, SecretStr, field_validator, model_validator

from app.schemas.common import Contract, Name
from app.schemas.geometry import Polygon, ReferencedPolygon, StoredProductROI


class CameraCreate(Contract):
    name: Name
    stream_url: SecretStr = Field(repr=False)
    is_active: bool = True

    @field_validator("stream_url")
    @classmethod
    def valid_stream(cls, value: SecretStr) -> SecretStr:
        try:
            url = urlsplit(value.get_secret_value())
            valid = url.scheme in {"rtsp", "rtsps"} and url.hostname and url.port != 0
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("A valid RTSP destination is required")
        return value


class CameraResponse(Contract):
    id: UUID
    name: str
    is_active: bool
    created_at: AwareDatetime
    updated_at: AwareDatetime
    configuration_revision: UUID
    source_revision: UUID

    @model_validator(mode="before")
    @classmethod
    def include_revision(cls, value: Any) -> Any:
        if not isinstance(value, dict) and hasattr(value, "stream_url"):
            from app.domain.configuration import camera_revision, source_revision

            return {
                **{
                    name: getattr(value, name)
                    for name in cls.model_fields
                    if name not in {"configuration_revision", "source_revision"}
                },
                "configuration_revision": camera_revision(value),
                "source_revision": source_revision(value),
            }
        return value


class CameraUpdate(Contract):
    expected_revision: UUID
    name: Name | None = None
    stream_url: SecretStr | None = Field(default=None, repr=False)
    is_active: bool | None = None

    @field_validator("stream_url")
    @classmethod
    def valid_stream(cls, value: SecretStr | None) -> SecretStr | None:
        return CameraCreate.valid_stream(value) if value is not None else None

    @model_validator(mode="after")
    def valid_update(self) -> Self:
        fields = self.model_fields_set - {"expected_revision"}
        if not fields or any(getattr(self, field) is None for field in fields):
            raise ValueError("Supply at least one non-null update")
        return self


class ShelfCreate(Contract):
    camera_id: UUID
    name: Name
    category: Name
    product_roi: ReferencedPolygon
    presence_roi: Polygon
    is_active: bool = True


class GeometryUpdate(Contract):
    expected_version: int = Field(ge=1)
    product_roi: ReferencedPolygon
    presence_roi: Polygon
    cutover_at: AwareDatetime


class ShelfUpdate(Contract):
    expected_version: int = Field(ge=1)
    name: Name | None = None
    category: Name | None = None
    product_roi: ReferencedPolygon | None = None
    presence_roi: Polygon | None = None
    is_active: bool | None = None

    @model_validator(mode="after")
    def valid_update(self) -> Self:
        fields = self.model_fields_set - {"expected_version"}
        if not fields or any(getattr(self, field) is None for field in fields):
            raise ValueError("Supply at least one non-null update")
        if ("product_roi" in fields) != ("presence_roi" in fields):
            raise ValueError("Update product and presence polygons together")
        return self


class ShelfResponse(Contract):
    id: UUID
    camera_id: UUID
    name: str
    category: str
    product_roi: StoredProductROI
    presence_roi: Polygon
    roi_version: int
    is_active: bool
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def matching_history(self) -> Self:
        if len(self.product_roi.history) != self.roi_version - 1:
            raise ValueError("Geometry history must preserve every previous version")
        return self
