from datetime import datetime
from uuid import UUID, uuid4

from app.domain.identity import TrackScope, observation_id
from app.schemas.common import ModelKind
from app.schemas.geometry import GeometrySnapshot
from app.schemas.model_version import ModelVersionCreate
from app.schemas.observation import ObservationCreate, ObservationEnvelope, Pose

POLYGON = {
    "type": "polygon",
    "coordinate_system": "normalized",
    "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
}


def model_version(
    kind: ModelKind, dependencies: dict[ModelKind, UUID] | None = None
) -> ModelVersionCreate:
    neural = kind in {ModelKind.POSE, ModelKind.FACE}
    return ModelVersionCreate.model_validate(
        {
            "name": f"test-{kind.value.lower()}",
            "kind": kind,
            "version": "1.0.0",
            "configuration": {
                "schema_version": "1.0",
                "artifact_name": f"{kind.value.lower()}.onnx" if neural else None,
                "artifact_sha256": "a" * 64 if neural else None,
                "parameters": {"sampled_fps": 10},
                "dependencies": dependencies or {},
            },
        }
    )


def observation(
    snapshot: GeometrySnapshot,
    models: dict[ModelKind, UUID],
    instant: datetime,
    *,
    frame: int = 0,
    track: str = "track-1",
    session_id: UUID | None = None,
) -> ObservationCreate:
    scope = TrackScope(snapshot.camera_id, session_id or uuid4(), track)
    return ObservationCreate.model_validate(
        {
            "id": observation_id(scope, frame),
            "camera_id": scope.camera_id,
            "attention_shelf_id": None,
            "reach_shelf_id": None,
            "tracking_session_id": scope.tracking_session_id,
            "local_track_id": track,
            "frame_seq": frame,
            "observed_at": instant,
            "attributes": {
                "bbox": None,
                "pose": dict.fromkeys(Pose.model_fields),
                "head_pose": {
                    "yaw_estimated": None,
                    "pitch_estimated": None,
                    "roll_estimated": None,
                    "method": "UNAVAILABLE",
                    "quality": None,
                },
                "attention": {
                    "direction_detected": None,
                    "focus_duration_ms": None,
                    "shelf_confidence": None,
                },
                "hands": {
                    side: {"x": None, "y": None, "in_product_roi": None}
                    for side in ("left", "right")
                },
                "movement": {"state": None, "velocity_estimated": None},
                "face": {"visible": False, "expression_detected": None, "confidence": None},
                "provenance": {
                    "schema_version": "2.0",
                    "pose_model_id": models[ModelKind.POSE],
                    "face_model_id": models.get(ModelKind.FACE),
                    "temporal_rules_id": models[ModelKind.TEMPORAL_RULES],
                    "source_epoch": scope.tracking_session_id,
                    "timestamp_source": "CAPTURE_CLOCK",
                    "source_fps": None,
                    "sampled_fps": 10,
                    "image_width": 1920,
                    "image_height": 1080,
                },
                "geometry": [snapshot.model_dump()],
                "presence": [
                    {
                        "shelf_id": snapshot.shelf_id,
                        "in_roi": True,
                        "anchor": {"x": 0.5, "y": 0.5, "confidence": 0.9},
                        "quality": 0.9,
                    }
                ],
                "kinematics": {
                    "shoulder_width_pixels": None,
                    "left": {"elbow_degrees": None, "shoulder_degrees": None, "reach_ratio": None},
                    "right": {"elbow_degrees": None, "shoulder_degrees": None, "reach_ratio": None},
                    "normalized_velocity": None,
                    "bimanual_ratio": None,
                    "head_adaptor_ratio": None,
                    "yaw_std_degrees": None,
                    "window_frames": 0,
                    "quality": None,
                    "reach_candidate": None,
                    "inspect_candidate": None,
                    "hesitation_candidate": None,
                },
                "spatial": {
                    "foot": None,
                    "head_origin": None,
                    "gaze_target": None,
                    "left_hand": None,
                    "right_hand": None,
                },
                "evidence": {
                    "provider": None,
                    "pickup_available": False,
                    "return_available": False,
                    "touch_available": False,
                    "object_detected": None,
                    "object_displacement_detected": None,
                    "object_left_shelf": None,
                },
            },
        }
    )


def example_envelope() -> ObservationEnvelope:
    from datetime import UTC

    identifiers = [UUID(int=index) for index in range(1, 8)]
    snapshot = GeometrySnapshot(
        camera_id=identifiers[0],
        shelf_id=identifiers[1],
        roi_version=1,
        reference_revision=identifiers[2],
        product_roi=POLYGON,
        presence_roi=POLYGON,
        valid_from=datetime(2026, 1, 1, tzinfo=UTC),
        valid_until=None,
    )
    data = observation(
        snapshot,
        {ModelKind.POSE: identifiers[3], ModelKind.TEMPORAL_RULES: identifiers[4]},
        datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        session_id=identifiers[5],
    )
    return ObservationEnvelope(schema_version="2.0", observation=data)
