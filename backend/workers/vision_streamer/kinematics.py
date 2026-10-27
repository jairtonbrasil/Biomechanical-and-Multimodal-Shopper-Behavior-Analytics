"""Quality-qualified 2D kinematics with explicit sampling and missing-data semantics."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from math import acos, degrees, hypot, isfinite, sqrt

from app.schemas.observation import ArmKinematics, Keypoint, Kinematics, Movement, Pose
from workers.vision_streamer.geometry import foot_anchor, head_anchor, pixel_point, usable

WINDOW_FRAMES = 15
MINIMUM_SHOULDER_PIXELS = 2.0
THRESHOLD_EPSILON = 1e-9


def joint_angle(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> float | None:
    left, right = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    denominator = hypot(*left) * hypot(*right)
    if denominator <= THRESHOLD_EPSILON:
        return None
    cosine = max(-1.0, min(1.0, (left[0] * right[0] + left[1] * right[1]) / denominator))
    return degrees(acos(cosine))


def reach_condition(arm: ArmKinematics) -> bool | None:
    if arm.elbow_degrees is None or arm.shoulder_degrees is None or arm.reach_ratio is None:
        return None
    return (
        130 - THRESHOLD_EPSILON <= arm.elbow_degrees <= 180 + THRESHOLD_EPSILON
        and arm.reach_ratio > 1.10 + THRESHOLD_EPSILON
        and arm.shoulder_degrees > 38 + THRESHOLD_EPSILON
    )


def _any_supported(values: tuple[bool | None, ...]) -> bool | None:
    if any(value is True for value in values):
        return True
    return False if all(value is False for value in values) else None


@dataclass
class TrackWindow:
    dimensions: tuple[int, int]
    observed_at: datetime
    positions: deque = field(default_factory=lambda: deque(maxlen=WINDOW_FRAMES))
    yaw: deque = field(default_factory=lambda: deque(maxlen=WINDOW_FRAMES))


class KinematicsEngine:
    def __init__(
        self,
        confidence_threshold: float = 0.3,
        sampling_fps: float = 10,
        max_gap_seconds: float = 1.0,
        max_tracks: int = 1024,
    ) -> None:
        if not 0 <= confidence_threshold <= 1 or not 0 < sampling_fps <= 240:
            raise ValueError("Invalid landmark confidence or sampling cadence")
        if not isfinite(max_gap_seconds) or max_gap_seconds <= 0 or not 1 <= max_tracks <= 10000:
            raise ValueError("Feature history bounds must be positive")
        self.confidence_threshold = confidence_threshold
        self.sampling_fps = sampling_fps
        self.max_gap_seconds = max_gap_seconds
        self.max_tracks = max_tracks
        self._tracks: dict[str, TrackWindow] = {}

    def clear(self, track_id: str | None = None) -> None:
        if track_id is None:
            self._tracks.clear()
        else:
            self._tracks.pop(track_id, None)

    reset = clear

    def _point(self, point: Keypoint | None) -> Keypoint | None:
        return point if usable(point, self.confidence_threshold) else None

    def _arm(self, pose: Pose, side: str, width: int, height: int, scale: float | None):
        shoulder, elbow, wrist, hip = (
            self._point(getattr(pose, f"{side}_{part}"))
            for part in ("shoulder", "elbow", "wrist", "hip")
        )
        elbow_angle = shoulder_angle = reach = None
        if shoulder is not None and elbow is not None and wrist is not None:
            elbow_angle = joint_angle(
                pixel_point(shoulder, width, height),
                pixel_point(elbow, width, height),
                pixel_point(wrist, width, height),
            )
        if hip is not None and shoulder is not None and elbow is not None:
            # This is the angle at the shoulder: shoulder-to-hip versus shoulder-to-elbow.
            shoulder_angle = joint_angle(
                pixel_point(hip, width, height),
                pixel_point(shoulder, width, height),
                pixel_point(elbow, width, height),
            )
        if shoulder is not None and wrist is not None and scale is not None:
            sx, sy = pixel_point(shoulder, width, height)
            wx, wy = pixel_point(wrist, width, height)
            reach = hypot(wx - sx, wy - sy) / scale
        return ArmKinematics(
            elbow_degrees=elbow_angle, shoulder_degrees=shoulder_angle, reach_ratio=reach
        )

    def update(
        self,
        track_id: str,
        pose: Pose,
        width: int,
        height: int,
        observed_at: datetime,
        yaw_degrees: float | None = None,
    ) -> tuple[Kinematics, Movement]:
        if not track_id.strip() or len(track_id) > 128:
            raise ValueError("A bounded camera-local track identifier is required")
        if width < 1 or height < 1 or observed_at.utcoffset() is None:
            raise ValueError("Positive image dimensions and timezone-aware event time are required")
        for stale_id in list(self._tracks):
            age = (observed_at - self._tracks[stale_id].observed_at).total_seconds()
            if age > max(5.0, self.max_gap_seconds * WINDOW_FRAMES):
                self.clear(stale_id)
        history = self._tracks.get(track_id)
        if history is not None:
            elapsed = (observed_at - history.observed_at).total_seconds()
            if (
                elapsed <= 0
                or elapsed > self.max_gap_seconds
                or history.dimensions != (width, height)
            ):
                self.clear(track_id)
                history = None
        if history is None:
            if len(self._tracks) >= self.max_tracks:
                oldest = min(self._tracks, key=lambda item: self._tracks[item].observed_at)
                self.clear(oldest)
            history = TrackWindow((width, height), observed_at)
            self._tracks[track_id] = history
        history.observed_at = observed_at
        shoulders = self._point(pose.left_shoulder), self._point(pose.right_shoulder)
        scale = None
        if all(point is not None for point in shoulders):
            left, right = (pixel_point(point, width, height) for point in shoulders)
            distance = hypot(right[0] - left[0], right[1] - left[1])
            if distance >= MINIMUM_SHOULDER_PIXELS:
                scale = distance
        left = self._arm(pose, "left", width, height, scale)
        right = self._arm(pose, "right", width, height, scale)
        anchor = foot_anchor(pose, self.confidence_threshold)
        velocity = None
        if anchor is None or scale is None:
            history.positions.clear()
        else:
            history.positions.append((observed_at, pixel_point(anchor, width, height)))
            if len(history.positions) == WINDOW_FRAMES:
                first_time, first_point = history.positions[0]
                last_time, last_point = history.positions[-1]
                elapsed = (last_time - first_time).total_seconds()
                if elapsed > 0:
                    velocity = hypot(
                        last_point[0] - first_point[0], last_point[1] - first_point[1]
                    ) / (elapsed * scale)
        yaw_std = None
        if yaw_degrees is None or not isfinite(yaw_degrees) or scale is None:
            history.yaw.clear()
        else:
            previous_yaw = history.yaw[-1] if history.yaw else yaw_degrees
            unwrapped = previous_yaw + ((yaw_degrees - previous_yaw + 180) % 360 - 180)
            history.yaw.append(unwrapped)
            if len(history.yaw) == WINDOW_FRAMES:
                mean = sum(history.yaw) / WINDOW_FRAMES
                yaw_std = sqrt(sum((value - mean) ** 2 for value in history.yaw) / WINDOW_FRAMES)
        wrists = self._point(pose.left_wrist), self._point(pose.right_wrist)
        bimanual = head_adaptor = None
        if scale is not None:
            if all(point is not None for point in wrists):
                first, second = (pixel_point(point, width, height) for point in wrists)
                bimanual = hypot(first[0] - second[0], first[1] - second[1]) / scale
            head = head_anchor(pose, self.confidence_threshold)
            if head is not None:
                hx, hy = pixel_point(head, width, height)
                distances = [
                    hypot(
                        pixel_point(point, width, height)[0] - hx,
                        pixel_point(point, width, height)[1] - hy,
                    )
                    / scale
                    for point in wrists
                    if point is not None
                ]
                head_adaptor = min(distances) if distances else None
        stationary = None if velocity is None else velocity < 0.048 * self.sampling_fps
        inspect = hesitation = None
        if scale is not None and stationary is not None:
            if (
                bimanual is not None
                and left.elbow_degrees is not None
                and right.elbow_degrees is not None
            ):
                inspect = bool(
                    stationary
                    and bimanual < 0.85
                    and 50 <= left.elbow_degrees <= 115
                    and 50 <= right.elbow_degrees <= 115
                )
            head_pattern = None if head_adaptor is None else head_adaptor < 0.72
            yaw_pattern = None if yaw_std is None else yaw_std > 13.5
            candidate = _any_supported((head_pattern, yaw_pattern))
            hesitation = None if candidate is None else bool(stationary and candidate)
        quality = min(point.confidence for point in shoulders) if scale is not None else None
        features = Kinematics(
            shoulder_width_pixels=scale,
            left=left,
            right=right,
            normalized_velocity=velocity,
            bimanual_ratio=bimanual,
            head_adaptor_ratio=head_adaptor,
            yaw_std_degrees=yaw_std,
            window_frames=len(history.positions),
            quality=quality,
            reach_candidate=_any_supported((reach_condition(left), reach_condition(right)))
            if scale is not None
            else None,
            inspect_candidate=inspect,
            hesitation_candidate=hesitation,
        )
        return features, Movement(
            state=None if stationary is None else "STANDING" if stationary else "WALKING",
            velocity_estimated=velocity,
        )
