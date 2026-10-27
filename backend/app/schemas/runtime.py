from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime

from app.schemas.common import Contract
from app.schemas.geometry import Polygon


class RuntimeShelf(Contract):
    id: UUID
    name: str
    category: str
    roi_version: int
    reference_revision: UUID
    product_roi: Polygon
    presence_roi: Polygon


class RuntimeConfiguration(Contract):
    camera_id: UUID
    configuration_revision: UUID
    source_revision: UUID
    is_active: bool
    proxy_path: str | None
    reference_revision: UUID | None
    shelves: tuple[RuntimeShelf, ...]
    proxy_state: Literal["PENDING", "APPLIED"] = "PENDING"
    applied_at: AwareDatetime | None = None
