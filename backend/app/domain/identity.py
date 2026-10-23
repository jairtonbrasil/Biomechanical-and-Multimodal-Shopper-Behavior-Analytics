import json
from dataclasses import dataclass
from uuid import UUID, uuid5

from app.schemas.common import ActionType

IDENTITY_NAMESPACE = UUID("3aa4e13f-13c4-55d6-9921-bbce2165d38a")


@dataclass(frozen=True)
class TrackScope:
    camera_id: UUID
    tracking_session_id: UUID
    local_track_id: str

    def __post_init__(self) -> None:
        if not self.local_track_id.strip() or len(self.local_track_id) > 128:
            raise ValueError("A bounded camera-local track identifier is required")


def stable_id(kind: str, *parts: object) -> UUID:
    return uuid5(
        IDENTITY_NAMESPACE, json.dumps(["1.0", kind, *map(str, parts)], separators=(",", ":"))
    )


def observation_id(scope: TrackScope, frame_seq: int) -> UUID:
    if frame_seq < 0:
        raise ValueError("Frame sequence cannot be negative")
    return stable_id(
        "observation", scope.camera_id, scope.tracking_session_id, scope.local_track_id, frame_seq
    )


def interaction_id(
    scope: TrackScope, shelf_id: UUID, roi_version: int, first_frame: int, rule_id: UUID
) -> UUID:
    if roi_version < 1 or first_frame < 0:
        raise ValueError("A positive ROI version and nonnegative first frame are required")
    return stable_id(
        "interaction",
        scope.camera_id,
        scope.tracking_session_id,
        scope.local_track_id,
        shelf_id,
        roi_version,
        first_frame,
        rule_id,
    )


def business_event_id(
    parent_id: UUID, kind: ActionType | str, first_frame: int, model_id: UUID
) -> UUID:
    if first_frame < 0 or not str(kind).strip():
        raise ValueError("Event type and nonnegative first qualifying frame are required")
    return stable_id("event", parent_id, kind, first_frame, model_id)
