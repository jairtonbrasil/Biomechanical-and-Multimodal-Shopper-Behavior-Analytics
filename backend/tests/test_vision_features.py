from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.schemas.geometry import Polygon
from app.schemas.observation import ArmKinematics, Keypoint, Pose
from app.schemas.runtime import RuntimeShelf
from workers.vision_streamer.gaze import GazeEstimator
from workers.vision_streamer.geometry import (
    foot_anchor,
    matching_shelf,
    point_in_polygon,
    ray_polygon_distance,
)
from workers.vision_streamer.kinematics import KinematicsEngine, joint_angle, reach_condition

NOW = datetime(2026, 10, 27, tzinfo=UTC)


def point(x, y, confidence=0.9):
    return Keypoint(x=x, y=y, confidence=confidence)


def pose(**changes):
    return Pose(
        **{
            **dict.fromkeys(Pose.model_fields),
            "nose": point(0.5, 0.2),
            "left_eye": point(0.48, 0.19),
            "right_eye": point(0.52, 0.19),
            "left_ear": point(0.46, 0.22),
            "right_ear": point(0.54, 0.22),
            "left_shoulder": point(0.4, 0.4),
            "right_shoulder": point(0.6, 0.4),
            "left_elbow": point(0.3, 0.45),
            "right_elbow": point(0.7, 0.45),
            "left_wrist": point(0.2, 0.4),
            "right_wrist": point(0.8, 0.4),
            "left_hip": point(0.4, 0.6),
            "right_hip": point(0.6, 0.6),
            "left_ankle": point(0.4, 0.8),
            "right_ankle": point(0.6, 0.8),
            **changes,
        }
    )


def shelf(points, **changes):
    polygon = Polygon(type="polygon", coordinate_system="normalized", points=points)
    return RuntimeShelf(
        **{
            "id": uuid4(),
            "name": "Test shelf",
            "category": "test",
            "roi_version": 1,
            "reference_revision": uuid4(),
            "product_roi": polygon,
            "presence_roi": polygon,
            **changes,
        }
    )


@pytest.mark.parametrize("elbow,expected", [(129.999, False), (130, True), (180, True)])
def test_reach_elbow_boundaries(elbow, expected):
    assert (
        reach_condition(ArmKinematics(elbow_degrees=elbow, shoulder_degrees=60, reach_ratio=1.5))
        is expected
    )


@pytest.mark.parametrize(
    "shoulder,reach,expected",
    [(38, 1.5, False), (38.001, 1.5, True), (60, 1.1, False), (60, 1.10001, True)],
)
def test_reach_strict_elevation_and_distance_boundaries(shoulder, reach, expected):
    assert (
        reach_condition(
            ArmKinematics(elbow_degrees=150, shoulder_degrees=shoulder, reach_ratio=reach)
        )
        is expected
    )


def test_degenerate_joint_vectors_remain_unknown():
    assert joint_angle((0, 0), (0, 0), (1, 1)) is None
    assert joint_angle((1, 0), (0, 0), (-1, 0)) == pytest.approx(180)


def test_angle_geometry_uses_pixels_and_declared_shoulder_vertex():
    source = pose(left_elbow=point(0.3, 0.5), left_wrist=point(0.2, 0.6))
    features, _ = KinematicsEngine().update("one", source, 1000, 500, NOW)
    assert features.shoulder_width_pixels == pytest.approx(200)
    assert features.left.shoulder_degrees == pytest.approx(63.43494882)
    assert features.left.elbow_degrees == pytest.approx(180)


def test_reach_cannot_combine_indicators_from_different_arms():
    source = pose(
        left_elbow=point(0.4, 0.5),
        left_wrist=point(0.4, 0.7),
        right_elbow=point(0.8, 0.4),
        right_wrist=point(0.6, 0.45),
    )
    features, _ = KinematicsEngine().update("one", source, 1000, 1000, NOW)
    assert features.left.elbow_degrees == pytest.approx(180)
    assert features.left.reach_ratio > 1.1
    assert features.right.shoulder_degrees > 38
    assert features.reach_candidate is False


def test_observed_reaching_arm_supports_candidate_with_other_arm_missing():
    source = pose(left_wrist=point(0.1, 0.4), right_wrist=None)
    features, _ = KinematicsEngine().update("one", source, 1000, 1000, NOW)
    assert features.reach_candidate is True


@pytest.mark.parametrize(
    "changes",
    [
        {"right_shoulder": None},
        {"right_shoulder": point(0.6, 0.4, 0.1)},
        {"right_shoulder": point(0.4001, 0.4)},
    ],
)
def test_invalid_shoulder_scale_suppresses_normalized_features(changes):
    features, movement = KinematicsEngine().update("one", pose(**changes), 1000, 1000, NOW)
    assert features.shoulder_width_pixels is None
    assert features.normalized_velocity is None
    assert features.reach_candidate is None
    assert features.inspect_candidate is None
    assert features.hesitation_candidate is None
    assert movement.state is None


def warmup(engine, source, *, step=0.1, yaw=None):
    result = None
    for sequence in range(15):
        result = engine.update(
            "one", source, 1000, 1000, NOW + timedelta(seconds=sequence * step), yaw
        )
    return result


def test_motion_requires_full_warmup_and_resets_after_missing_anchor():
    engine = KinematicsEngine()
    features, movement = engine.update("one", pose(), 1000, 1000, NOW)
    assert features.window_frames == 1 and movement.state is None
    features, movement = warmup(engine, pose())
    assert features.window_frames == 15 and features.normalized_velocity == 0
    assert movement.state == "STANDING"
    features, movement = engine.update(
        "one", pose(left_ankle=None), 1000, 1000, NOW + timedelta(seconds=1.5)
    )
    assert features.window_frames == 0 and movement.state is None
    features, _ = engine.update("one", pose(), 1000, 1000, NOW + timedelta(seconds=1.6))
    assert features.window_frames == 1


def test_motion_uses_event_time_seconds_instead_of_received_frame_count():
    engine = KinematicsEngine()
    for sequence in range(15):
        elapsed = sequence * 0.2
        source = pose(
            left_ankle=point(0.4 + 0.1 * elapsed, 0.8),
            right_ankle=point(0.6 + 0.1 * elapsed, 0.8),
        )
        features, movement = engine.update(
            "one", source, 1000, 1000, NOW + timedelta(seconds=elapsed)
        )
    assert features.normalized_velocity == pytest.approx(0.5)
    assert movement.velocity_estimated == pytest.approx(0.5)
    assert movement.state == "WALKING"


@pytest.mark.parametrize("sampling_fps,expected", [(5, "WALKING"), (10, "STANDING")])
def test_nominal_sampling_cadence_is_explicit_in_stationary_threshold(sampling_fps, expected):
    engine = KinematicsEngine(sampling_fps=sampling_fps)
    for sequence in range(15):
        elapsed = sequence * 0.2
        source = pose(
            left_ankle=point(0.4 + 0.06 * elapsed, 0.8),
            right_ankle=point(0.6 + 0.06 * elapsed, 0.8),
        )
        _, movement = engine.update("one", source, 1000, 1000, NOW + timedelta(seconds=elapsed))
    assert movement.velocity_estimated == pytest.approx(0.3)
    assert movement.state == expected


@pytest.mark.parametrize("event_time", [NOW, NOW + timedelta(seconds=3)])
def test_backward_time_and_support_gaps_reset_track_windows(event_time):
    engine = KinematicsEngine()
    warmup(engine, pose())
    features, _ = engine.update("one", pose(), 1000, 1000, event_time)
    assert features.window_frames == 1
    assert features.normalized_velocity is None


def test_dimension_change_and_explicit_cutover_reset_history():
    engine = KinematicsEngine()
    warmup(engine, pose())
    features, _ = engine.update("one", pose(), 500, 500, NOW + timedelta(seconds=1.5))
    assert features.window_frames == 1
    engine.clear()
    features, _ = engine.update("one", pose(), 500, 500, NOW + timedelta(seconds=1.6))
    assert features.window_frames == 1


def test_bimanual_inspection_requires_both_flexed_arms_and_observed_stationarity():
    source = pose(
        left_elbow=point(0.4, 0.55),
        left_wrist=point(0.49, 0.55),
        right_elbow=point(0.6, 0.55),
        right_wrist=point(0.51, 0.55),
    )
    engine = KinematicsEngine()
    assert engine.update("one", source, 1000, 1000, NOW)[0].inspect_candidate is None
    assert warmup(engine, source)[0].inspect_candidate is True
    straight_arm = source.model_copy(update={"right_wrist": point(0.6, 0.65)})
    features, _ = engine.update("one", straight_arm, 1000, 1000, NOW + timedelta(seconds=1.5))
    assert features.bimanual_ratio < 0.85
    assert features.inspect_candidate is False


def test_head_adaptor_hesitation_does_not_invent_facial_or_product_evidence():
    source = pose(left_wrist=point(0.5, 0.21))
    features, _ = warmup(KinematicsEngine(), source)
    assert features.head_adaptor_ratio < 0.72
    assert features.hesitation_candidate is True


def test_yaw_warmup_and_wraparound_do_not_create_spurious_variance():
    engine = KinematicsEngine()
    for sequence in range(15):
        features, _ = engine.update(
            "one",
            pose(),
            1000,
            1000,
            NOW + timedelta(seconds=sequence * 0.1),
            179 if sequence % 2 else -179,
        )
        if sequence < 14:
            assert features.yaw_std_degrees is None
    assert features.yaw_std_degrees < 1.1
    assert features.hesitation_candidate is False


def test_yaw_variation_requires_complete_valid_window():
    engine = KinematicsEngine()
    for sequence in range(15):
        features, _ = engine.update(
            "one",
            pose(),
            1000,
            1000,
            NOW + timedelta(seconds=sequence * 0.1),
            20 if sequence % 2 else -20,
        )
    assert features.yaw_std_degrees > 13.5
    assert features.hesitation_candidate is True
    features, _ = engine.update("one", pose(), 1000, 1000, NOW + timedelta(seconds=1.5), None)
    assert features.yaw_std_degrees is None


def test_track_history_is_bounded_and_does_not_mix_people():
    engine = KinematicsEngine(max_tracks=2)
    warmup(engine, pose())
    assert (
        engine.update("two", pose(), 1000, 1000, NOW + timedelta(seconds=1.5))[0].window_frames == 1
    )
    engine.update("three", pose(), 1000, 1000, NOW + timedelta(seconds=1.6))
    assert len(engine._tracks) == 2
    assert "one" not in engine._tracks


@pytest.mark.parametrize(
    "xy,expected", [((0, 0), True), ((1, 0.5), True), ((0.5, 0.5), True), ((1.01, 0.5), False)]
)
def test_polygon_membership_includes_valid_image_boundaries(xy, expected):
    polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert point_in_polygon(xy, polygon) is expected


def test_foot_anchor_needs_two_observed_ankles_and_preserves_zero_coordinates():
    source = pose(left_ankle=point(0, 0.5), right_ankle=point(0.2, 0.5))
    assert foot_anchor(source).x == pytest.approx(0.1)
    assert foot_anchor(source.model_copy(update={"right_ankle": None})) is None


def test_hand_shelf_assignment_rejects_ambiguous_overlaps():
    first = shelf([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)])
    second = first.model_copy(update={"id": uuid4()})
    assert matching_shelf(point(0.1, 0.1), (first,)) == first.id
    assert matching_shelf(point(0.1, 0.1), (first, second)) is None
    assert matching_shelf(None, (first,)) is None
    assert matching_shelf(point(0.1, 0.1, 0.1), (first,)) is None


def test_ray_intersection_handles_collinear_boundary_and_nearest_entry():
    polygon = [(1, 0), (2, 0), (2, 1), (1, 1)]
    assert ray_polygon_distance((0, 0), (1, 0), 3, polygon) == pytest.approx(1)
    assert ray_polygon_distance((0, 0.5), (1, 0), 3, polygon) == pytest.approx(1)
    assert ray_polygon_distance((0, 2), (1, 0), 3, polygon) is None


def test_face_head_direction_uses_observed_confidence_and_unknown_pitch():
    result = GazeEstimator().estimate(pose(left_eye=point(0.48, 0.19, 0.4)), 1000, 500, ())
    assert result.head_pose.method == "FACE"
    assert result.head_pose.quality == 0.4
    assert result.head_pose.pitch_estimated is None
    assert result.origin is not None and result.target is not None
    assert result.head_pose.yaw_estimated == pytest.approx(0)


def test_back_turned_direction_is_qualified_and_has_no_fabricated_pitch_or_roll():
    result = GazeEstimator().estimate(pose(nose=None, left_eye=None, right_eye=None), 1000, 500, ())
    assert result.head_pose.method == "BODY_FALLBACK"
    assert result.head_pose.pitch_estimated is None and result.head_pose.roll_estimated is None
    assert result.head_pose.quality == 0.9


@pytest.mark.parametrize(
    "changes",
    [
        {"left_ear": None, "right_ear": None},
        {"left_shoulder": None},
        {"right_shoulder": point(0.4, 0.4)},
    ],
)
def test_missing_head_geometry_is_unavailable_instead_of_default_orientation(changes):
    result = GazeEstimator().estimate(pose(**changes), 1000, 1000, ())
    assert result.head_pose.method == "UNAVAILABLE"
    assert result.head_pose.quality is None and result.head_pose.yaw_estimated is None
    assert result.origin is None and result.target is None


def test_gaze_chooses_nearest_intersection_and_suppresses_ties():
    near = shelf([(0.45, 0.1), (0.55, 0.1), (0.55, 0.15), (0.45, 0.15)])
    far = shelf([(0.45, 0), (0.55, 0), (0.55, 0.05), (0.45, 0.05)])
    estimator = GazeEstimator()
    assert estimator.estimate(pose(), 1000, 1000, (far, near)).attention_shelf_id == near.id
    duplicate = near.model_copy(update={"id": uuid4()})
    assert estimator.estimate(pose(), 1000, 1000, (near, duplicate)).attention_shelf_id is None


def test_gaze_projection_remains_in_observable_image_plane():
    result = GazeEstimator(range_shoulder_widths=10).estimate(pose(), 1000, 1000, ())
    assert result.target.y == pytest.approx(0)
    assert 0 <= result.target.x <= 1


def test_gaze_has_no_cross_person_smoothing_state():
    estimator = GazeEstimator()
    original = estimator.estimate(pose(), 1000, 1000, ())
    estimator.estimate(pose(nose=point(0.6, 0.2)), 1000, 1000, ())
    assert estimator.estimate(pose(), 1000, 1000, ()) == original


def test_eye_line_roll_does_not_flip_when_anatomical_eye_order_is_mirrored():
    source = pose(left_eye=point(0.52, 0.19), right_eye=point(0.48, 0.19))
    result = GazeEstimator().estimate(source, 1000, 1000, ())
    assert result.head_pose.roll_estimated == pytest.approx(0)
