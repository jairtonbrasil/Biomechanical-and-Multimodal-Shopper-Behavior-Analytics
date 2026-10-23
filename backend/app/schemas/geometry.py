from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from app.schemas.common import Contract, UnitFloat, require_ordered

Point = tuple[UnitFloat, UnitFloat]


def cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def intersects(a: Point, b: Point, c: Point, d: Point) -> bool:
    def on_segment(p: Point, q: Point, r: Point) -> bool:
        return (
            cross(p, q, r) == 0
            and min(p[0], q[0]) <= r[0] <= max(p[0], q[0])
            and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])
        )

    return (cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0) or any(
        (on_segment(a, b, c), on_segment(a, b, d), on_segment(c, d, a), on_segment(c, d, b))
    )


class Polygon(Contract):
    type: Literal["polygon"]
    coordinate_system: Literal["normalized"]
    points: Annotated[tuple[Point, ...], Field(min_length=3, max_length=64)]

    @model_validator(mode="after")
    def simple_nonzero_polygon(self) -> Self:
        points = self.points
        if len(set(points)) != len(points):
            raise ValueError("Vertices must be distinct; do not repeat the closing vertex")
        size = len(points)
        area = sum(
            points[i][0] * points[(i + 1) % size][1] - points[(i + 1) % size][0] * points[i][1]
            for i in range(size)
        )
        if abs(area) <= 1e-12:
            raise ValueError("Polygon area is zero or numerically degenerate")
        for i in range(size):
            if cross(points[i - 1], points[i], points[(i + 1) % size]) == 0:
                raise ValueError("Adjacent edges must not overlap or contain redundant vertices")
            for j in range(i + 1, size):
                if j == i + 1 or (i == 0 and j == size - 1):
                    continue
                if intersects(points[i], points[(i + 1) % size], points[j], points[(j + 1) % size]):
                    raise ValueError("Polygon edges intersect")
        return self


class ReferencedPolygon(Polygon):
    reference_revision: UUID


class GeometrySnapshot(Contract):
    camera_id: UUID
    shelf_id: UUID
    roi_version: int = Field(ge=1)
    reference_revision: UUID
    product_roi: Polygon
    presence_roi: Polygon
    valid_from: AwareDatetime
    valid_until: AwareDatetime | None

    @model_validator(mode="after")
    def ordered(self) -> Self:
        require_ordered(self.valid_from, self.valid_until)
        return self


class StoredProductROI(ReferencedPolygon):
    history: tuple[GeometrySnapshot, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def history_is_ordered(self) -> Self:
        for index, snapshot in enumerate(self.history):
            if snapshot.roi_version != index + 1 or snapshot.valid_until is None:
                raise ValueError("History must contain closed, consecutive ROI versions")
            if index and self.history[index - 1].valid_until != snapshot.valid_from:
                raise ValueError("Historical geometry intervals must be contiguous")
        return self
