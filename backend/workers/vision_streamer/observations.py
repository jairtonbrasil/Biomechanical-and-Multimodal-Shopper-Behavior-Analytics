"""Construct validated observations without promoting posture to object evidence."""

from datetime import datetime
from uuid import UUID

from app.domain.identity import TrackScope, observation_id
from app.schemas.common import ModelKind
from app.schemas.observation import (
    Attention,
    Face,
    Hand,
    Hands,
    ObservationAttributes,
    ObservationCreate,
    ObservationEnvelope,
    Presence,
    ProductEvidence,
    Provenance,
    Spatial,
)
from workers.vision_streamer.configuration import VisionConfiguration
from workers.vision_streamer.gaze import GazeEstimator
from workers.vision_streamer.geometry import foot_anchor, matching_shelf, point_in_polygon, usable
from workers.vision_streamer.kinematics import KinematicsEngine, reach_condition
from workers.vision_streamer.types import Detection


def unsupported_product_evidence() -> ProductEvidence:
    return ProductEvidence(
        provider=None,
        pickup_available=False,
        return_available=False,
        touch_available=False,
        object_detected=None,
        object_displacement_detected=None,
        object_left_shelf=None,
    )


def build_observation(
    detection: Detection,
    *,
    configuration: VisionConfiguration,
    model_ids: dict[ModelKind, UUID],
    session_id: UUID,
    source_epoch: UUID,
    frame_seq: int,
    observed_at: datetime,
    source_fps: float | None,
    sampled_fps: float,
    width: int,
    height: int,
    kinematics: KinematicsEngine,
    gaze: GazeEstimator,
    face: Face,
    confidence_threshold: float = 0.3,
) -> ObservationEnvelope:
    pose, shelves = detection.pose, configuration.runtime.shelves
    orientation = gaze.estimate(pose, width, height, shelves)
    features, movement = kinematics.update(
        detection.local_track_id,
        pose,
        width,
        height,
        observed_at,
        orientation.head_pose.yaw_estimated,
    )
    foot = foot_anchor(pose, confidence_threshold)
    hands = {}
    reach_targets = set()
    for side in ("left", "right"):
        wrist = getattr(pose, f"{side}_wrist")
        observed = usable(wrist, confidence_threshold)
        hands[side] = Hand(
            x=wrist.x if observed else None,
            y=wrist.y if observed else None,
            in_product_roi=any(
                point_in_polygon((wrist.x, wrist.y), shelf.product_roi) for shelf in shelves
            )
            if observed
            else None,
        )
        if reach_condition(getattr(features, side)) is True:
            target = matching_shelf(wrist, shelves, confidence_threshold=confidence_threshold)
            if target is not None:
                reach_targets.add(target)
    reach_shelf_id = next(iter(reach_targets)) if len(reach_targets) == 1 else None
    scope = TrackScope(configuration.runtime.camera_id, session_id, detection.local_track_id)
    attributes = ObservationAttributes(
        bbox=detection.bbox,
        pose=pose,
        head_pose=orientation.head_pose,
        attention=Attention(
            direction_detected=orientation.head_pose.method != "UNAVAILABLE",
            focus_duration_ms=None,
            shelf_confidence=orientation.quality if orientation.attention_shelf_id else None,
        ),
        hands=Hands(**hands),
        movement=movement,
        face=face,
        provenance=Provenance(
            schema_version="2.0",
            pose_model_id=model_ids[ModelKind.POSE],
            face_model_id=model_ids.get(ModelKind.FACE),
            temporal_rules_id=model_ids[ModelKind.TEMPORAL_RULES],
            source_epoch=source_epoch,
            timestamp_source="CAPTURE_CLOCK",
            source_fps=source_fps,
            sampled_fps=sampled_fps,
            image_width=width,
            image_height=height,
        ),
        geometry=configuration.geometry,
        presence=tuple(
            Presence(
                shelf_id=shelf.id,
                in_roi=point_in_polygon((foot.x, foot.y), shelf.presence_roi) if foot else None,
                anchor=foot,
                quality=foot.confidence if foot else None,
            )
            for shelf in shelves
        ),
        kinematics=features,
        spatial=Spatial(
            foot=foot,
            head_origin=orientation.origin,
            gaze_target=orientation.target,
            left_hand=pose.left_wrist,
            right_hand=pose.right_wrist,
        ),
        evidence=unsupported_product_evidence(),
    )
    return ObservationEnvelope(
        schema_version="2.0",
        observation=ObservationCreate(
            id=observation_id(scope, frame_seq),
            camera_id=scope.camera_id,
            attention_shelf_id=orientation.attention_shelf_id,
            reach_shelf_id=reach_shelf_id,
            tracking_session_id=session_id,
            local_track_id=detection.local_track_id,
            frame_seq=frame_seq,
            observed_at=observed_at,
            attributes=attributes,
        ),
    )
