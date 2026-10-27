import importlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.schemas.observation import Keypoint, Pose
from workers.vision_streamer.face import EXPRESSION_MAPPING, FaceEstimator
from workers.vision_streamer.pose import KEYPOINT_NAMES, TRACKER_PATH, PoseEstimator


class Tensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class Frame:
    shape = (100, 200, 3)

    def __getitem__(self, indexes):
        rows, columns = indexes
        return SimpleNamespace(shape=(rows.stop - rows.start, columns.stop - columns.start, 3))


def pose(**points):
    return Pose(**{**dict.fromkeys(Pose.model_fields), **points})


@pytest.fixture
def tracked_result():
    joints = [[50.0, 50.0] for _ in range(17)]
    joints[5] = [40, 30]
    qualities = [0.9] * 17
    return SimpleNamespace(
        boxes=SimpleNamespace(
            id=Tensor([7.0]), xyxy=Tensor([[20, 10, 180, 90]]), conf=Tensor([0.8])
        ),
        keypoints=SimpleNamespace(xy=Tensor([joints]), conf=Tensor([qualities])),
    )


@pytest.fixture
def estimator(tmp_path, monkeypatch, tracked_result):
    weights = tmp_path / "pose.pt"
    weights.write_bytes(b"model-test-fixture")
    model = SimpleNamespace(track=Mock(return_value=[tracked_result]), predictor=None)
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=factory))
    instance = PoseEstimator(weights)
    assert factory.call_args.kwargs == {"task": "pose"}
    return instance


def test_pose_normalizes_each_axis_and_preserves_coco_landmark_names(estimator):
    detection = estimator.infer(Frame())[0]
    assert tuple(Pose.model_fields) == KEYPOINT_NAMES
    assert detection.local_track_id == "7"
    assert detection.pose.left_shoulder == Keypoint(x=0.2, y=0.3, confidence=0.9)
    assert detection.bbox.x1 == detection.bbox.y1 == 0.1
    assert detection.bbox.x2 == detection.bbox.y2 == 0.9
    assert detection.bbox.confidence == 0.8
    arguments = estimator.model.track.call_args.kwargs
    assert arguments["tracker"] == str(TRACKER_PATH)
    assert "tracker_type: bytetrack" in TRACKER_PATH.read_text()
    assert arguments["persist"] is True and arguments["classes"] == [0]
    assert arguments["save"] is False and arguments["verbose"] is False


@pytest.mark.parametrize("invalid", [0.1, -0.1, math.nan, math.inf])
def test_invalid_or_low_confidence_keypoints_remain_null(estimator, tracked_result, invalid):
    tracked_result.keypoints.conf.values[0][5] = invalid
    detection = estimator.infer(Frame())[0]
    assert detection.pose.left_shoulder is None
    assert detection.pose.right_shoulder is not None


@pytest.mark.parametrize("coordinate", [[201, 20], [-1, 20], [20, 101], [math.nan, 20]])
def test_outside_or_nonfinite_points_do_not_become_clamped_measurements(
    estimator, tracked_result, coordinate
):
    tracked_result.keypoints.xy.values[0][9] = coordinate
    assert estimator.infer(Frame())[0].pose.left_wrist is None


def test_valid_boundary_keypoints_are_not_discarded(estimator, tracked_result):
    tracked_result.keypoints.xy.values[0][9] = [0, 100]
    assert estimator.infer(Frame())[0].pose.left_wrist == Keypoint(x=0, y=1, confidence=0.9)


@pytest.mark.parametrize("identifiers", [None, [math.nan], [-1], [1.5]])
def test_untracked_or_invalid_identifiers_do_not_fabricate_local_tracks(
    estimator, tracked_result, identifiers
):
    tracked_result.boxes.id = Tensor(identifiers) if identifiers is not None else None
    assert estimator.infer(Frame()) == []


def test_tracking_reset_reuses_loaded_model_and_resets_all_trackers(estimator):
    reset = Mock()
    estimator.model.predictor = SimpleNamespace(
        trackers=[SimpleNamespace(reset=reset)], vid_path=[1]
    )
    previous = estimator.model
    estimator.reset()
    reset.assert_called_once()
    assert estimator.model is previous
    assert estimator.model.predictor.vid_path == [None]


def test_model_failure_does_not_echo_vendor_errors(estimator):
    estimator.model.track.side_effect = RuntimeError("rtsp://private:password@camera")
    with pytest.raises(RuntimeError, match="^Pose inference failed$") as result:
        estimator.infer(Frame())
    assert "password" not in str(result.value)


def test_missing_pose_artifact_does_not_trigger_a_download(tmp_path):
    with pytest.raises(ValueError, match="existing YOLO"):
        PoseEstimator(tmp_path / "missing.pt")


@pytest.fixture
def face_estimator(tmp_path, monkeypatch):
    weights = tmp_path / "enet_b0_8_best_vgaf.onnx"
    weights.write_bytes(b"model-test-fixture")
    session = SimpleNamespace(
        get_inputs=lambda: [
            SimpleNamespace(name="input", type="tensor(float)", shape=[1, 3, 224, 224])
        ],
        get_outputs=lambda: [SimpleNamespace(shape=[1, 8])],
        run=Mock(return_value=[[[-3, -3, -3, -3, 8, -3, -3, -3]]]),
    )
    factory = Mock(return_value=session)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=factory),
    )
    instance = FaceEstimator(weights, min_sharpness=0)
    assert factory.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
    return instance


def frontal_pose():
    return pose(
        nose=Keypoint(x=0.5, y=0.5, confidence=0.9),
        left_eye=Keypoint(x=0.45, y=0.4, confidence=0.9),
        right_eye=Keypoint(x=0.55, y=0.4, confidence=0.9),
    )


def test_face_crop_requires_observed_nose_and_both_eyes(face_estimator):
    assert face_estimator.crop(Frame(), pose()) is None
    frontal = frontal_pose()
    assert face_estimator.crop(Frame(), frontal.model_copy(update={"left_eye": None})) is None
    assert min(face_estimator.crop(Frame(), frontal).shape[:2]) >= 64


def test_face_crop_rejects_small_truncated_and_sideways_faces(face_estimator):
    frontal = frontal_pose()
    assert (
        face_estimator.crop(
            Frame(),
            frontal.model_copy(update={"right_eye": Keypoint(x=0.46, y=0.4, confidence=0.9)}),
        )
        is None
    )
    assert (
        face_estimator.crop(
            Frame(), frontal.model_copy(update={"nose": Keypoint(x=0.8, y=0.5, confidence=0.9)})
        )
        is None
    )
    assert (
        face_estimator.crop(
            Frame(),
            frontal.model_copy(update={"left_eye": Keypoint(x=0.45, y=0.01, confidence=0.9)}),
        )
        is None
    )


def test_no_face_measurement_is_neutral_by_default(face_estimator):
    unavailable = face_estimator.unavailable()
    assert unavailable.visible is None
    assert unavailable.expression_detected is None and unavailable.confidence is None
    assert EXPRESSION_MAPPING["Fear"] is None and EXPRESSION_MAPPING["Contempt"] is None


def test_face_rgb_preprocessing_and_real_softmax_score(face_estimator):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :, 2] = 255
    face = face_estimator.infer(frame, frontal_pose())
    assert face.visible is True and face.expression_detected == "smile"
    assert 0.99 < face.confidence < 1
    batch = face_estimator.session.run.call_args.args[1]["input"]
    assert batch.shape == (1, 3, 224, 224) and batch.dtype == np.float32
    assert float(batch[0, 0, 0, 0]) == pytest.approx((1 - 0.485) / 0.229)
    assert float(batch[0, 2, 0, 0]) == pytest.approx(-0.406 / 0.225)


def test_unmapped_low_confidence_or_failed_face_inference_stays_unknown(face_estimator):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    for scores in ([[0] * 8], [[0, 9, 0, 0, 0, 0, 0, 0]], [[math.nan] * 8]):
        face_estimator.session.run.return_value = [scores]
        output = face_estimator.infer(frame, frontal_pose())
        assert output.visible is True and output.expression_detected is None
        assert output.confidence is None
    face_estimator.session.run.side_effect = RuntimeError("private model failure")
    assert face_estimator.infer(frame, frontal_pose()).expression_detected is None
    face_estimator.session.run.reset_mock()
    face_estimator.min_sharpness = 10
    assert face_estimator.infer(frame, frontal_pose()).visible is None
    face_estimator.session.run.assert_not_called()


def test_model_preparation_rejects_wrong_checksum_without_replacing_target(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "prepare_vision_models",
        Path(__file__).resolve().parents[2] / "scripts/prepare_vision_models.py",
    )
    if spec is None or spec.loader is None or not Path(spec.origin).is_file():
        pytest.skip("Model preparation script is outside the core test container context")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source.pt"
    source.write_bytes(b"invalid model")
    target = tmp_path / "model.pt"
    target.write_bytes(b"previous verified model")
    with pytest.raises(ValueError, match="checksum"):
        module.prepare(
            source, tmp_path, {"filename": "model.pt", "sha256": "a" * 64}, download=False
        )
    assert target.read_bytes() == b"previous verified model"
    assert not list(tmp_path.glob(".model-*"))
