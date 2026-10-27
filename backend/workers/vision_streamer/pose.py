"""YOLO11-pose with explicit ByteTrack continuity and nullable keypoint evidence."""

import math
import os
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.schemas.observation import BoundingBox, Keypoint, Pose
from workers.vision_streamer.types import Detection

if TYPE_CHECKING:
    import numpy as np

KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
TRACKER_PATH = Path(__file__).with_name("bytetrack.yaml")


def _values(tensor: Any) -> list[Any]:
    if tensor is None:
        return []
    return tensor.cpu().tolist()


class PoseEstimator:
    def __init__(
        self,
        weights: Path,
        confidence: float = 0.3,
        imgsz: int = 640,
        device: str = "cpu",
        max_people: int = 32,
        *,
        keypoint_confidence: float = 0.3,
    ) -> None:
        if not weights.is_file() or weights.suffix != ".pt":
            raise ValueError("An existing YOLO pose artifact is required")
        if (
            not 0 < confidence <= 1
            or not 0 < keypoint_confidence <= 1
            or not 32 <= imgsz <= 1920
            or imgsz % 32
            or not 1 <= max_people <= 128
        ):
            raise ValueError("Invalid pose extraction configuration")
        os.environ["YOLO_AUTOINSTALL"] = "false"
        from ultralytics import YOLO

        self.weights = weights
        self.confidence, self.keypoint_confidence = confidence, keypoint_confidence
        self.imgsz, self.device, self.max_people = imgsz, device, max_people
        try:
            self.model = YOLO(str(weights), task="pose")
        except Exception:
            raise RuntimeError("Pose model initialization failed") from None

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "detector_confidence": self.confidence,
            "keypoint_confidence": self.keypoint_confidence,
            "image_size": self.imgsz,
            "device": self.device,
            "max_people": self.max_people,
            "tracker": "ByteTrack",
            "tracker_sha256": sha256(TRACKER_PATH.read_bytes()).hexdigest(),
            "keypoint_layout": "COCO_17",
        }

    def reset(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None:
            trackers = getattr(predictor, "trackers", ())
            for tracker in trackers:
                tracker.reset()
            if trackers:
                predictor.vid_path = [None] * len(trackers)

    def infer(self, frame: "np.ndarray") -> list[Detection]:
        if len(frame.shape) != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
            raise ValueError("Pose inference requires a nonempty three-channel image")
        height, width = frame.shape[:2]
        try:
            results = self.model.track(
                frame,
                persist=True,
                tracker=str(TRACKER_PATH),
                conf=self.confidence,
                imgsz=self.imgsz,
                device=self.device,
                max_det=self.max_people,
                classes=[0],
                verbose=False,
                save=False,
            )
            if not results:
                return []
            result = results[0]
            if result.boxes is None or result.keypoints is None or result.boxes.id is None:
                return []
            identifiers = _values(result.boxes.id)
            boxes = _values(result.boxes.xyxy)
            scores = _values(result.boxes.conf)
            points = _values(result.keypoints.xy)
            qualities = _values(result.keypoints.conf)
            detections = []
            seen = set()
            for identifier, box, score, joints, confidence in zip(
                identifiers, boxes, scores, points, qualities, strict=True
            ):
                if (
                    not math.isfinite(identifier)
                    or identifier < 0
                    or identifier != int(identifier)
                    or identifier in seen
                    or not math.isfinite(score)
                    or not self.confidence <= score <= 1
                    or len(box) != 4
                    or not all(math.isfinite(value) for value in box)
                    or len(joints) != 17
                    or len(confidence) != 17
                ):
                    continue
                x1, y1, x2, y2 = box
                x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
                y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                pose = {}
                for name, coordinates, quality in zip(
                    KEYPOINT_NAMES, joints, confidence, strict=True
                ):
                    valid = (
                        len(coordinates) == 2
                        and all(math.isfinite(value) for value in (*coordinates, quality))
                        and self.keypoint_confidence <= quality <= 1
                        and 0 <= coordinates[0] <= width
                        and 0 <= coordinates[1] <= height
                    )
                    pose[name] = (
                        Keypoint(
                            x=coordinates[0] / width, y=coordinates[1] / height, confidence=quality
                        )
                        if valid
                        else None
                    )
                detections.append(
                    Detection(
                        local_track_id=str(int(identifier)),
                        bbox=BoundingBox(
                            x1=x1 / width,
                            y1=y1 / height,
                            x2=x2 / width,
                            y2=y2 / height,
                            confidence=score,
                        ),
                        pose=Pose(**pose),
                    )
                )
                seen.add(identifier)
                if len(detections) >= self.max_people:
                    break
            return detections
        except Exception:
            raise RuntimeError("Pose inference failed") from None
