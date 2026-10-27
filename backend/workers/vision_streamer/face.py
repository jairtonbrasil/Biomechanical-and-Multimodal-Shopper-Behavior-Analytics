"""Opportunistic HSEmotion ONNX classification without neutral fallbacks or downloads."""

import math
from pathlib import Path
from typing import TYPE_CHECKING

from app.schemas.observation import Face, Pose

if TYPE_CHECKING:
    import numpy as np

EXPRESSION_MAPPING = {
    "Anger": "frown",
    "Contempt": None,
    "Disgust": None,
    "Fear": None,
    "Happiness": "smile",
    "Neutral": "neutral",
    "Sadness": "frown",
    "Surprise": "surprise",
}
EMOTION_CLASSES = tuple(EXPRESSION_MAPPING)


class FaceEstimator:
    def __init__(
        self,
        weights: Path,
        confidence: float = 0.5,
        *,
        keypoint_confidence: float = 0.4,
        min_crop_pixels: int = 32,
        min_sharpness: float = 10.0,
        threads: int = 1,
    ) -> None:
        if not weights.is_file() or weights.suffix != ".onnx":
            raise ValueError("An existing HSEmotion ONNX artifact is required")
        if (
            not 0 < confidence <= 1
            or not 0 < keypoint_confidence <= 1
            or not 16 <= min_crop_pixels <= 256
            or not math.isfinite(min_sharpness)
            or min_sharpness < 0
            or not 1 <= threads <= 32
        ):
            raise ValueError("Invalid face extraction configuration")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.enable_mem_pattern = True
        options.log_severity_level = 3
        try:
            self.session = ort.InferenceSession(
                str(weights), sess_options=options, providers=["CPUExecutionProvider"]
            )
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            if (
                len(inputs) != 1
                or inputs[0].type != "tensor(float)"
                or list(inputs[0].shape)[1:] != [3, 224, 224]
                or len(outputs) != 1
                or list(outputs[0].shape)[-1] != 8
            ):
                raise ValueError
            self.input_name = inputs[0].name
        except Exception:
            raise RuntimeError("Face model initialization failed") from None
        self.weights, self.confidence = weights, confidence
        self.keypoint_confidence = keypoint_confidence
        self.min_crop_pixels, self.min_sharpness = min_crop_pixels, min_sharpness
        self.threads = threads

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "classification_threshold": self.confidence,
            "keypoint_confidence": self.keypoint_confidence,
            "min_crop_pixels": self.min_crop_pixels,
            "min_laplacian_variance": self.min_sharpness,
            "min_retained_crop_area": 0.85,
            "max_eye_vertical_ratio": 0.6,
            "max_nose_horizontal_ratio": 0.8,
            "nose_vertical_ratio_interval": [-0.5, 2.0],
            "crop_width_eye_distance_ratio": 3.2,
            "crop_height_eye_distance_ratio": 4.0,
            "color_space": "RGB",
            "input_size": 224,
            "normalization": "ImageNet",
            "expression_mapping": dict(EXPRESSION_MAPPING),
            "provider": "CPUExecutionProvider",
            "intra_op_threads": self.threads,
        }

    @staticmethod
    def unavailable() -> Face:
        return Face(visible=None, expression_detected=None, confidence=None)

    def crop(self, frame: "np.ndarray", pose: Pose) -> "np.ndarray | None":
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return None
        points = (pose.nose, pose.left_eye, pose.right_eye)
        if any(point is None or point.confidence < self.keypoint_confidence for point in points):
            return None
        height, width = frame.shape[:2]
        nose, left, right = [(point.x * width, point.y * height) for point in points]
        separation = math.dist(left, right)
        if separation < self.min_crop_pixels / 3.2:
            return None
        center_x, eye_y = (left[0] + right[0]) / 2, (left[1] + right[1]) / 2
        if (
            abs(left[1] - right[1]) > separation * 0.6
            or abs(nose[0] - center_x) > separation * 0.8
            or not -0.5 * separation <= nose[1] - eye_y <= 2 * separation
        ):
            return None
        half_width, half_height = separation * 1.6, separation * 2
        raw = (
            center_x - half_width,
            eye_y - separation,
            center_x + half_width,
            eye_y - separation + 2 * half_height,
        )
        x1, y1, x2, y2 = (
            max(0, int(raw[0])),
            max(0, int(raw[1])),
            min(width, int(raw[2])),
            min(height, int(raw[3])),
        )
        if min(x2 - x1, y2 - y1) < self.min_crop_pixels or (x2 - x1) * (y2 - y1) < 0.85 * (
            raw[2] - raw[0]
        ) * (raw[3] - raw[1]):
            return None
        return frame[y1:y2, x1:x2]

    def infer(self, frame: "np.ndarray", pose: Pose) -> Face:
        import cv2
        import numpy as np

        crop = self.crop(frame, pose)
        if crop is None:
            return self.unavailable()
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if float(cv2.Laplacian(gray, cv2.CV_32F).var()) < self.min_sharpness:
            return self.unavailable()
        try:
            # HSEmotion expects RGB/ImageNet normalization, rather than OpenCV's BGR input.
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            normalized = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
            normalized = (
                normalized - np.array([0.485, 0.456, 0.406], dtype=np.float32)
            ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
            batch = normalized.transpose(2, 0, 1)[np.newaxis, ...]
            logits = np.asarray(self.session.run(None, {self.input_name: batch})[0])
            if logits.shape != (1, 8) or not np.isfinite(logits).all():
                return Face(visible=True, expression_detected=None, confidence=None)
            scores = np.exp(logits[0] - logits[0].max())
            scores /= scores.sum()
            index = int(scores.argmax())
            probability = float(scores[index])
            expression = EXPRESSION_MAPPING[EMOTION_CLASSES[index]]
            if probability < self.confidence or expression is None:
                return Face(visible=True, expression_detected=None, confidence=None)
            return Face(visible=True, expression_detected=expression, confidence=probability)
        except Exception:
            return Face(visible=True, expression_detected=None, confidence=None)
