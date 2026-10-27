"""Observed landmarks and bounded image-plane geometry for vision features."""

from collections.abc import Sequence
from math import hypot
from uuid import UUID

from app.schemas.geometry import Polygon
from app.schemas.observation import Keypoint, Pose
from app.schemas.runtime import RuntimeShelf

Point = tuple[float, float]
EPSILON = 1e-9


def usable(point: Keypoint | None, confidence_threshold: float = 0.3) -> bool:
    return point is not None and point.confidence >= confidence_threshold


def pixel_point(point: Keypoint, width: int, height: int) -> Point:
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")
    return point.x * width, point.y * height


def average_keypoints(points: Sequence[Keypoint]) -> Keypoint | None:
    if not points:
        return None
    return Keypoint(
        x=sum(point.x for point in points) / len(points),
        y=sum(point.y for point in points) / len(points),
        confidence=min(point.confidence for point in points),
    )


def foot_anchor(pose: Pose, confidence_threshold: float = 0.3) -> Keypoint | None:
    ankles = (pose.left_ankle, pose.right_ankle)
    if not all(usable(point, confidence_threshold) for point in ankles):
        return None
    return average_keypoints([point for point in ankles if point is not None])


def head_anchor(pose: Pose, confidence_threshold: float = 0.3) -> Keypoint | None:
    return average_keypoints(
        [
            point
            for point in (pose.nose, pose.left_eye, pose.right_eye, pose.left_ear, pose.right_ear)
            if usable(point, confidence_threshold) and point is not None
        ]
    )


def point_in_polygon(point: Point, polygon: Polygon | Sequence[Point]) -> bool:
    """Include boundary points; coordinates must share the polygon's image plane."""
    points = polygon.points if isinstance(polygon, Polygon) else polygon
    if len(points) < 3:
        return False
    inside = False
    px, py = point
    for index, (ax, ay) in enumerate(points):
        bx, by = points[(index + 1) % len(points)]
        cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
        tolerance = EPSILON * max(1, hypot(bx - ax, by - ay))
        if (
            abs(cross) <= tolerance
            and min(ax, bx) - EPSILON <= px <= max(ax, bx) + EPSILON
            and min(ay, by) - EPSILON <= py <= max(ay, by) + EPSILON
        ):
            return True
        if (ay > py) != (by > py):
            intersection_x = ax + (py - ay) * (bx - ax) / (by - ay)
            if px < intersection_x:
                inside = not inside
    return inside


def matching_shelf(
    point: Keypoint | None,
    shelves: tuple[RuntimeShelf, ...],
    *,
    field: str = "product_roi",
    confidence_threshold: float = 0.3,
) -> UUID | None:
    if field not in {"product_roi", "presence_roi"}:
        raise ValueError("Unknown spatial region")
    if not usable(point, confidence_threshold):
        return None
    matches = [
        shelf.id for shelf in shelves if point_in_polygon((point.x, point.y), getattr(shelf, field))
    ]
    return matches[0] if len(matches) == 1 else None


def ray_polygon_distance(
    origin: Point, direction: Point, maximum: float, polygon: Sequence[Point]
) -> float | None:
    """Return the first segment/polygon intersection distance, including tangencies."""
    if point_in_polygon(origin, polygon):
        return 0.0
    ox, oy = origin
    dx, dy = direction
    intersections = []
    for index, (ax, ay) in enumerate(polygon):
        bx, by = polygon[(index + 1) % len(polygon)]
        ex, ey = bx - ax, by - ay
        denominator = dx * ey - dy * ex
        offset_x, offset_y = ax - ox, ay - oy
        if abs(denominator) <= EPSILON:
            if abs(offset_x * dy - offset_y * dx) <= EPSILON:
                for x, y in ((ax, ay), (bx, by)):
                    distance = (x - ox) * dx + (y - oy) * dy
                    if -EPSILON <= distance <= maximum + EPSILON:
                        intersections.append(max(0.0, distance))
            continue
        distance = (offset_x * ey - offset_y * ex) / denominator
        fraction = (offset_x * dy - offset_y * dx) / denominator
        if -EPSILON <= distance <= maximum + EPSILON and -EPSILON <= fraction <= 1 + EPSILON:
            intersections.append(max(0.0, distance))
    return min(intersections) if intersections else None
