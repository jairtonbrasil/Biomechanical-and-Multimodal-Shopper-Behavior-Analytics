from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from pydantic import AwareDatetime, Field

from app.schemas.common import Contract


@dataclass(frozen=True)
class ReferenceFrame:
    content: bytes
    revision: UUID
    width: int
    height: int
    captured_at: datetime
    source_revision: UUID


class ReferenceMetadata(Contract):
    camera_id: UUID
    revision: UUID
    source_revision: UUID
    captured_at: AwareDatetime
    width: int = Field(gt=0, le=1920)
    height: int = Field(gt=0, le=1920)
    content_length: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ReferenceCaptureJob(Contract):
    id: UUID
    camera_id: UUID
    configuration_revision: UUID
    source_revision: UUID
    requested_at: AwareDatetime
    refresh: bool = False


class ReferenceResponse(Contract):
    camera_id: UUID
    revision: UUID
    source_revision: UUID
    captured_at: AwareDatetime
    width: int
    height: int
    image_url: str
