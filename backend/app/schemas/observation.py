from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from app.schemas.common import Contract, FiniteFloat, NonNegative, UnitFloat
from app.schemas.geometry import GeometrySnapshot


class Keypoint(Contract):
    x: UnitFloat
    y: UnitFloat
    confidence: UnitFloat


class Pose(Contract):
    nose: Keypoint | None
    left_eye: Keypoint | None
    right_eye: Keypoint | None
    left_ear: Keypoint | None
    right_ear: Keypoint | None
    left_shoulder: Keypoint | None
    right_shoulder: Keypoint | None
    left_elbow: Keypoint | None
    right_elbow: Keypoint | None
    left_wrist: Keypoint | None
    right_wrist: Keypoint | None
    left_hip: Keypoint | None
    right_hip: Keypoint | None
    left_knee: Keypoint | None
    right_knee: Keypoint | None
    left_ankle: Keypoint | None
    right_ankle: Keypoint | None


class BoundingBox(Contract):
    x1: UnitFloat
    y1: UnitFloat
    x2: UnitFloat
    y2: UnitFloat
    confidence: UnitFloat

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Bounding box must have positive width and height")
        return self


class HeadPose(Contract):
    yaw_estimated: FiniteFloat | None
    pitch_estimated: FiniteFloat | None
    roll_estimated: FiniteFloat | None
    method: Literal["FACE", "BODY_FALLBACK", "UNAVAILABLE"]
    quality: UnitFloat | None

    @model_validator(mode="after")
    def unavailable_has_no_angles(self) -> Self:
        if self.method == "UNAVAILABLE" and any(
            value is not None
            for value in (
                self.yaw_estimated,
                self.pitch_estimated,
                self.roll_estimated,
                self.quality,
            )
        ):
            raise ValueError(
                "Unavailable head orientation cannot contain measured angles or quality"
            )
        return self


class Attention(Contract):
    direction_detected: bool | None
    focus_duration_ms: NonNegative | None
    shelf_confidence: UnitFloat | None


class Hand(Contract):
    x: UnitFloat | None
    y: UnitFloat | None
    in_product_roi: bool | None

    @model_validator(mode="after")
    def paired_coordinates(self) -> Self:
        if (self.x is None) != (self.y is None):
            raise ValueError("Hand coordinates must be supplied together")
        if self.x is None and self.in_product_roi is not None:
            raise ValueError("ROI membership is unknown when the hand is unobserved")
        return self


class Hands(Contract):
    left: Hand
    right: Hand


class Movement(Contract):
    state: Literal["STANDING", "WALKING"] | None
    velocity_estimated: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None


class Face(Contract):
    visible: bool | None
    expression_detected: str | None = Field(max_length=64)
    confidence: UnitFloat | None

    @model_validator(mode="after")
    def visible_evidence(self) -> Self:
        if self.visible is not True and (
            self.expression_detected is not None or self.confidence is not None
        ):
            raise ValueError("An unobserved face cannot have an expression or confidence")
        return self


class Provenance(Contract):
    schema_version: Literal["2.0"]
    pose_model_id: UUID
    face_model_id: UUID | None
    temporal_rules_id: UUID
    source_epoch: UUID
    timestamp_source: Literal["STREAM_PTS", "CAPTURE_CLOCK"]
    source_fps: Annotated[float, Field(gt=0, le=240, allow_inf_nan=False)] | None
    sampled_fps: Annotated[float, Field(gt=0, le=240, allow_inf_nan=False)]
    image_width: int = Field(ge=1, le=16384)
    image_height: int = Field(ge=1, le=16384)


class Presence(Contract):
    shelf_id: UUID
    in_roi: bool | None
    anchor: Keypoint | None
    quality: UnitFloat | None


class ArmKinematics(Contract):
    elbow_degrees: Annotated[float, Field(ge=0, le=180, allow_inf_nan=False)] | None
    shoulder_degrees: Annotated[float, Field(ge=0, le=180, allow_inf_nan=False)] | None
    reach_ratio: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None


class Kinematics(Contract):
    shoulder_width_pixels: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None
    left: ArmKinematics
    right: ArmKinematics
    normalized_velocity: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    bimanual_ratio: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    head_adaptor_ratio: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    yaw_std_degrees: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    window_frames: int = Field(ge=0, le=15)
    quality: UnitFloat | None
    reach_candidate: bool | None
    inspect_candidate: bool | None
    hesitation_candidate: bool | None

    @model_validator(mode="after")
    def invalid_scale_has_no_candidates(self) -> Self:
        if self.shoulder_width_pixels is None and any(
            item is not None
            for item in (self.reach_candidate, self.inspect_candidate, self.hesitation_candidate)
        ):
            raise ValueError("Kinematic candidates require a usable normalization scale")
        return self


class Spatial(Contract):
    foot: Keypoint | None
    head_origin: Keypoint | None
    gaze_target: Keypoint | None
    left_hand: Keypoint | None
    right_hand: Keypoint | None


class ProductEvidence(Contract):
    provider: str | None = Field(max_length=128)
    pickup_available: bool
    return_available: bool
    touch_available: bool
    object_detected: bool | None
    object_displacement_detected: bool | None
    object_left_shelf: bool | None

    @model_validator(mode="after")
    def provider_is_required_for_available_capabilities(self) -> Self:
        if (
            self.pickup_available or self.return_available or self.touch_available
        ) and not self.provider:
            raise ValueError("Available product capabilities require a validated evidence provider")
        return self


class ObservationAttributes(Contract):
    bbox: BoundingBox | None
    pose: Pose
    head_pose: HeadPose
    attention: Attention
    hands: Hands
    movement: Movement
    face: Face
    provenance: Provenance
    geometry: tuple[GeometrySnapshot, ...] = Field(max_length=32)
    presence: tuple[Presence, ...] = Field(max_length=32)
    kinematics: Kinematics
    spatial: Spatial
    evidence: ProductEvidence


class ObservationCreate(Contract):
    id: UUID
    camera_id: UUID
    attention_shelf_id: UUID | None
    reach_shelf_id: UUID | None
    tracking_session_id: UUID
    local_track_id: str = Field(min_length=1, max_length=128)
    frame_seq: NonNegative
    observed_at: AwareDatetime
    attributes: ObservationAttributes

    @model_validator(mode="after")
    def consistent_geometry(self) -> Self:
        snapshots = self.attributes.geometry
        identifiers = {snapshot.shelf_id for snapshot in snapshots}
        if len(identifiers) != len(snapshots):
            raise ValueError("Each evaluated shelf must have exactly one geometry snapshot")
        for snapshot in snapshots:
            if snapshot.camera_id != self.camera_id:
                raise ValueError("Observation geometry belongs to a different camera")
            if self.observed_at < snapshot.valid_from or (
                snapshot.valid_until is not None and self.observed_at >= snapshot.valid_until
            ):
                raise ValueError("Geometry snapshot is not valid at observation event time")
        targets = {self.attention_shelf_id, self.reach_shelf_id} - {None}
        if not targets.issubset(identifiers):
            raise ValueError("Observation targets require corresponding geometry snapshots")
        memberships = [presence.shelf_id for presence in self.attributes.presence]
        if len(set(memberships)) != len(memberships) or not set(memberships).issubset(identifiers):
            raise ValueError("Presence memberships must reference unique evaluated geometries")
        if (
            self.attributes.face.visible is True
            and self.attributes.face.expression_detected is not None
        ):
            if self.attributes.provenance.face_model_id is None:
                raise ValueError("Facial classification requires its immutable model version")
        return self


class ObservationEnvelope(Contract):
    schema_version: Literal["2.0"]
    observation: ObservationCreate

    @model_validator(mode="after")
    def bounded_message(self) -> Self:
        if len(self.model_dump_json().encode()) > 262144 - 512:
            raise ValueError("Observation envelope exceeds the version 2.0 message budget")
        return self
