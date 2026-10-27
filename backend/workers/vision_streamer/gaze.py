"""Image-plane head-direction proxies, not calibrated 3D pose or eye fixation."""

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, sin
from uuid import UUID

from app.schemas.observation import HeadPose, Keypoint, Pose
from app.schemas.runtime import RuntimeShelf
from workers.vision_streamer.geometry import (
    average_keypoints,
    head_anchor,
    pixel_point,
    ray_polygon_distance,
    usable,
)
from workers.vision_streamer.kinematics import MINIMUM_SHOULDER_PIXELS


@dataclass(frozen=True)
class GazeResult:
    head_pose: HeadPose
    origin: Keypoint | None
    target: Keypoint | None
    attention_shelf_id: UUID | None

    @property
    def quality(self) -> float | None:
        return self.head_pose.quality


class GazeEstimator:
    def __init__(self, confidence_threshold: float = 0.3, range_shoulder_widths: float = 2.5):
        if not 0 <= confidence_threshold <= 1 or not 0 < range_shoulder_widths <= 10:
            raise ValueError("Invalid head-direction confidence or projection range")
        self.confidence_threshold = confidence_threshold
        self.range_shoulder_widths = range_shoulder_widths

    def reset(self) -> None:
        """The estimator intentionally keeps no history between camera-local tracks."""

    @staticmethod
    def unavailable() -> GazeResult:
        return GazeResult(
            HeadPose(
                yaw_estimated=None,
                pitch_estimated=None,
                roll_estimated=None,
                method="UNAVAILABLE",
                quality=None,
            ),
            None,
            None,
            None,
        )

    def estimate(
        self,
        pose: Pose,
        width: int,
        height: int,
        shelves: tuple[RuntimeShelf, ...],
    ) -> GazeResult:
        if width < 1 or height < 1:
            raise ValueError("Image dimensions must be positive")
        threshold = self.confidence_threshold
        shoulders = pose.left_shoulder, pose.right_shoulder
        if not all(usable(point, threshold) for point in shoulders):
            return self.unavailable()
        left, right = (pixel_point(point, width, height) for point in shoulders)
        shoulder_width = hypot(right[0] - left[0], right[1] - left[1])
        if shoulder_width < MINIMUM_SHOULDER_PIXELS:
            return self.unavailable()
        chest_normal = (
            (right[1] - left[1]) / shoulder_width,
            -(right[0] - left[0]) / shoulder_width,
        )
        ears = [
            point
            for point in (pose.left_ear, pose.right_ear)
            if usable(point, threshold) and point is not None
        ]
        facial = [
            point
            for point in (pose.nose, pose.left_eye, pose.right_eye)
            if usable(point, threshold) and point is not None
        ]
        if not ears:
            return self.unavailable()
        origin = head_anchor(pose, threshold)
        if origin is None:
            return self.unavailable()
        if len(facial) >= 2:
            ear_center = average_keypoints(ears)
            face_center = average_keypoints(facial)
            ex, ey = pixel_point(ear_center, width, height)
            fx, fy = pixel_point(face_center, width, height)
            separation = hypot(fx - ex, fy - ey)
            if separation < MINIMUM_SHOULDER_PIXELS:
                return self.unavailable()
            raw = (
                0.85 * (fx - ex) / separation + 0.15 * chest_normal[0],
                0.85 * (fy - ey) / separation + 0.15 * chest_normal[1],
            )
            method = "FACE"
            evidence = [*ears, *facial, *shoulders]
        else:
            ear_center = average_keypoints(ears)
            ox, _ = pixel_point(ear_center, width, height)
            cervical_offset = (ox - (left[0] + right[0]) / 2) / shoulder_width
            left_conf = pose.left_ear.confidence if usable(pose.left_ear, threshold) else 0.0
            right_conf = pose.right_ear.confidence if usable(pose.right_ear, threshold) else 0.0
            rotation = max(-1.1, min(1.1, (left_conf - right_conf) * 0.9 + cervical_offset * 1.4))
            raw = (
                chest_normal[0] * cos(rotation) - chest_normal[1] * sin(rotation),
                chest_normal[0] * sin(rotation) + chest_normal[1] * cos(rotation),
            )
            origin = ear_center
            method = "BODY_FALLBACK"
            evidence = [*ears, *shoulders]
        norm = hypot(*raw)
        if norm < 1e-9:
            return self.unavailable()
        direction = raw[0] / norm, raw[1] / norm
        quality = min(point.confidence for point in evidence)
        roll = None
        if (
            method == "FACE"
            and usable(pose.left_eye, threshold)
            and usable(pose.right_eye, threshold)
        ):
            lx, ly = pixel_point(pose.left_eye, width, height)
            rx, ry = pixel_point(pose.right_eye, width, height)
            if hypot(rx - lx, ry - ly) >= MINIMUM_SHOULDER_PIXELS:
                # Eye-line tilt is axial: swapping anatomical eye order must not add 180 degrees.
                roll = (degrees(atan2(ry - ly, rx - lx)) + 90) % 180 - 90
        head_pose = HeadPose(
            yaw_estimated=degrees(atan2(direction[0], -direction[1])),
            pitch_estimated=None,
            roll_estimated=roll,
            method=method,
            quality=quality,
        )
        ox, oy = pixel_point(origin, width, height)
        distance = self.range_shoulder_widths * shoulder_width
        # Clip the displayed ray to the observable image plane, never clamp an off-image sample.
        for coordinate, delta, boundary in ((ox, direction[0], width), (oy, direction[1], height)):
            if delta > 1e-9:
                distance = min(distance, (boundary - coordinate) / delta)
            elif delta < -1e-9:
                distance = min(distance, -coordinate / delta)
        if distance <= 1e-9:
            return GazeResult(head_pose, origin, None, None)
        tx, ty = ox + direction[0] * distance, oy + direction[1] * distance
        target = Keypoint(
            x=max(0.0, min(1.0, tx / width)),
            y=max(0.0, min(1.0, ty / height)),
            confidence=quality,
        )
        intersections = []
        for shelf in shelves:
            polygon = [(x * width, y * height) for x, y in shelf.product_roi.points]
            intersection = ray_polygon_distance((ox, oy), direction, distance, polygon)
            if intersection is not None:
                intersections.append((intersection, shelf.id))
        intersections.sort(key=lambda item: item[0])
        selected = None
        if intersections and (
            len(intersections) == 1 or abs(intersections[1][0] - intersections[0][0]) > 1e-6
        ):
            selected = intersections[0][1]
        return GazeResult(head_pose, origin, target, selected)
