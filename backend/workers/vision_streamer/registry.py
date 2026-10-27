"""Verify local model artifacts and register immutable extraction configurations."""

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from app.core.config import Settings
from app.core.database import Database
from app.repositories.model_versions import ModelVersionRepository
from app.schemas.common import ModelKind
from app.schemas.model_version import ModelVersionCreate


@dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str


def local_artifacts(manifest_path: Path) -> dict[str, Artifact]:
    if not manifest_path.is_file() or manifest_path.stat().st_size > 16384:
        raise ValueError("Run prepare_vision_models.py before starting vision")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "1.0":
        raise ValueError("Unsupported local model manifest")
    artifacts = {}
    for name, suffix in (("pose", ".pt"), ("face", ".onnx")):
        value = manifest[name]
        filename = value["filename"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("Model manifest requires plain artifact filenames")
        path = manifest_path.parent / filename
        if path.suffix != suffix or not path.is_file():
            raise ValueError("A required local model artifact is absent")
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != value["sha256"]:
            raise ValueError("A model artifact differs from its pinned checksum")
        artifacts[name] = Artifact(path, digest)
    return artifacts


def extraction_parameters(settings: Settings) -> dict:
    return {
        "rule_revision": "vision-features-1.0",
        "feature_source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("kinematics.py", "geometry.py", "gaze.py", "observations.py")
        },
        "gaze_conventions": {
            "face_torso_blend": [0.85, 0.15],
            "fallback_ear_asymmetry_weight": 0.9,
            "fallback_cervical_offset_weight": 1.4,
            "fallback_rotation_clip_radians": 1.1,
            "face_ear_minimum_separation_pixels": 2,
            "temporal_smoothing": False,
            "pitch_available": False,
            "roll": "axial_eye_line_modulo_180",
        },
        "motion_estimator": "endpoint_displacement_divided_by_event_time_interval",
        "stationary_requires_full_motion_window": True,
        "sampling_fps": settings.sampling_fps,
        "keypoint_confidence": settings.vision_confidence,
        "shoulder_convention": "angle_at_shoulder_between_hip_and_elbow",
        "distance_units": "projected_shoulder_widths",
        "velocity_units": "projected_shoulder_widths_per_second",
        "standing_threshold_per_second": 0.048 * settings.sampling_fps,
        "window_frames": 15,
        "max_gap_seconds": 1.0,
        "head_yaw_units": "image_plane_direction_degrees",
        "minimum_shoulder_width_pixels": 2.0,
        "same_arm_reach": {"elbow_min": 130, "elbow_max": 180, "ratio_gt": 1.10, "shoulder_gt": 38},
        "inspection": {"both_elbows_min": 50, "both_elbows_max": 115, "bimanual_lt": 0.85},
        "hesitation": {"head_adaptor_lt": 0.72, "yaw_std_gt": 13.5},
        "gaze_range_shoulder_widths": 2.5,
        "gaze_dimension": "heuristic_2d_projection",
        "entry_confirmation_ms": settings.entry_confirmation_ms,
        "exit_tolerance_ms": settings.exit_tolerance_ms,
        "tracking_loss_tolerance_ms": settings.tracking_loss_tolerance_ms,
        "product_evidence_provider": None,
        "scientific_validation": "heuristic_not_accuracy_validated",
    }


async def register_models(
    settings: Settings,
    database: Database,
    artifacts: dict[str, Artifact],
    pose_parameters: dict,
    face_parameters: dict,
) -> dict[ModelKind, UUID]:
    runtime = {
        name: importlib.metadata.version(name)
        for name in (
            "ultralytics",
            "torch",
            "torchvision",
            "numpy",
            "opencv-python",
            "onnxruntime",
            "lap",
        )
    }
    definitions = {
        ModelKind.POSE: {
            "artifact": artifacts["pose"],
            "parameters": {"runtime": runtime, **pose_parameters},
        },
        ModelKind.FACE: {
            "artifact": artifacts["face"],
            "parameters": {"runtime": runtime, **face_parameters},
        },
        ModelKind.TEMPORAL_RULES: {"artifact": None, "parameters": extraction_parameters(settings)},
    }
    identifiers = {}
    async with database.transaction() as session:
        repository = ModelVersionRepository(session)
        for kind, item in definitions.items():
            artifact = item["artifact"]
            configuration = {
                "schema_version": "1.0",
                "artifact_name": artifact.path.name if artifact else None,
                "artifact_sha256": artifact.sha256 if artifact else None,
                "parameters": item["parameters"],
                "dependencies": {},
            }
            fingerprint = hashlib.sha256(
                json.dumps(configuration, sort_keys=True).encode()
            ).hexdigest()[:16]
            model = await repository.register(
                ModelVersionCreate(
                    name=f"vision-{kind.value.lower()}",
                    kind=kind,
                    version=f"1.0-{fingerprint}",
                    configuration=configuration,
                )
            )
            identifiers[kind] = model.id
    return identifiers
