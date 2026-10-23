from typing import Self
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
