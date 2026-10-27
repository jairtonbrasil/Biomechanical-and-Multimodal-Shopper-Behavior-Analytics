"""Typed observations returned by model adapters before feature extraction."""

from dataclasses import dataclass

from app.schemas.observation import BoundingBox, Pose


@dataclass(frozen=True, slots=True)
class Detection:
    local_track_id: str
    bbox: BoundingBox
    pose: Pose
