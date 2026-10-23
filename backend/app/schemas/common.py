from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
NonNegative = Annotated[int, Field(ge=0)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Name = Annotated[str, Field(min_length=1, max_length=128, pattern=r".*\S.*")]


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        from_attributes=True,
        allow_inf_nan=False,
        serialize_by_alias=True,
    )


class ModelKind(StrEnum):
    POSE = "POSE"
    FACE = "FACE"
    TEMPORAL_RULES = "TEMPORAL_RULES"
    ATTENTION = "ATTENTION"
    ACTION = "ACTION"
    BODY_REACTION = "BODY_REACTION"
    FACE_BODY_REACTION = "FACE_BODY_REACTION"


class ActionType(StrEnum):
    REACH = "REACH"
    INSPECT = "INSPECT"
    PICKUP = "PICKUP"
    RETURN = "RETURN"


class LinkType(StrEnum):
    PRESENCE = "PRESENCE"
    ATTENTION = "ATTENTION"
    REACH = "REACH"


class ReactionSource(StrEnum):
    FACE_AND_BODY = "FACE_AND_BODY"
    BODY_BIOMECHANICS = "BODY_BIOMECHANICS"


class CloseReason(StrEnum):
    LEFT_ROI = "LEFT_ROI"
    TRACK_LOST = "TRACK_LOST"
    CAMERA_OFFLINE = "CAMERA_OFFLINE"
    ROI_CHANGED = "ROI_CHANGED"
    REPROCESSING_GAP = "REPROCESSING_GAP"


class TimeWindow(Contract):
    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.end <= self.start:
            raise ValueError("Window end must follow its start")
        return self


def require_ordered(start: datetime, end: datetime | None) -> None:
    if end is not None and end < start:
        raise ValueError("End cannot precede start")
