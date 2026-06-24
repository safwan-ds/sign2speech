"""Tests for core/inference/gesture_predictor.py — real-time prediction logic."""

from __future__ import annotations

import torch

from core.inference.gesture_predictor import _canonical_sensor_name
from core.inference.gesture_predictor import _normalize_class_thresholds

# Module-level constants from gesture_predictor for testing
_SENSOR_NAME_ALIASES = {
    "accelx": "accelX", "accel_x": "accelX", "accelerometer_x": "accelX",
    "accely": "accelY", "accel_y": "accelY", "accelerometer_y": "accelY",
    "accelz": "accelZ", "accel_z": "accelZ", "accelerometer_z": "accelZ",
    "gyrox": "gyroX", "gyro_x": "gyroX", "gyroscope_x": "gyroX",
    "gyroy": "gyroY", "gyro_y": "gyroY", "gyroscope_y": "gyroY",
    "gyroz": "gyroZ", "gyro_z": "gyroZ", "gyroscope_z": "gyroZ",
}


class TestCanonicalSensorName:
    """Tests for _canonical_sensor_name()."""

    def test_accel_alias(self) -> None:
        assert _canonical_sensor_name("accel_x") == "accelX"
        assert _canonical_sensor_name("accelx") == "accelX"
        assert _canonical_sensor_name("accelerometer_x") == "accelX"

    def test_gyro_alias(self) -> None:
        assert _canonical_sensor_name("gyro_y") == "gyroY"
        assert _canonical_sensor_name("gyroy") == "gyroY"

    def test_flex_snake_case(self) -> None:
        assert _canonical_sensor_name("flex_0") == "flex0"
        assert _canonical_sensor_name("flex_3") == "flex3"

    def test_sensor_prefix(self) -> None:
        assert _canonical_sensor_name("sensor_2") == "flex2"

    def test_already_canonical(self) -> None:
        assert _canonical_sensor_name("flex0") == "flex0"
        assert _canonical_sensor_name("accelX") == "accelX"
        assert _canonical_sensor_name("gyroZ") == "gyroZ"

    def test_dash_separator(self) -> None:
        assert _canonical_sensor_name("accel-x") == "accelX"

    def test_unknown_name_passes_through(self) -> None:
        result = _canonical_sensor_name("temperature")
        assert result == "temperature"

    def test_strips_whitespace(self) -> None:
        assert _canonical_sensor_name("  flex0  ") == "flex0"


class TestNormalizeClassThresholds:
    """Tests for _normalize_class_thresholds()."""

    def test_normalizes_string_keys(self) -> None:
        thresholds = {"rest": 0.8, "A": 0.9}
        result = _normalize_class_thresholds(thresholds)
        assert result["REST"] == 0.8
        assert result["A"] == 0.9

    def test_handles_invalid_values(self) -> None:
        thresholds = {"A": "not_a_number", "B": 0.9}
        result = _normalize_class_thresholds(thresholds)
        assert "A" not in result
        assert result["B"] == 0.9

    def test_non_dict_input_returns_empty(self) -> None:
        assert _normalize_class_thresholds(None) == {}
        assert _normalize_class_thresholds([1, 2, 3]) == {}
        assert _normalize_class_thresholds("string") == {}

    def test_empty_dict(self) -> None:
        assert _normalize_class_thresholds({}) == {}


class TestLSTMGesturePredictor:
    """Tests for LSTMGesturePredictor class with monkeypatched paths."""

    def test_init_with_synthetic_checkpoint(
        self, monkeypatch, synthetic_model_checkpoint: str, synthetic_encoder: str
    ) -> None:
        import os
        # Monkeypatch paths before import
        monkeypatch.setattr(
            "core.inference.gesture_predictor.MODEL_PATH",
            os.path.join(synthetic_model_checkpoint, "model.pth"),
        )
        monkeypatch.setattr(
            "core.inference.gesture_predictor.MODELS_DIR",
            os.path.dirname(synthetic_model_checkpoint),
        )

        from core.inference.gesture_predictor import LSTMGesturePredictor

        predictor = LSTMGesturePredictor(device=torch.device("cpu"))
        assert predictor is not None
        assert hasattr(predictor, "model")
        assert predictor.use_onnx is False

    def test_sanitize_sensor_dict(self, monkeypatch, synthetic_model_checkpoint: str) -> None:
        import os
        monkeypatch.setattr(
            "core.inference.gesture_predictor.MODEL_PATH",
            os.path.join(synthetic_model_checkpoint, "model.pth"),
        )
        monkeypatch.setattr(
            "core.inference.gesture_predictor.MODELS_DIR",
            os.path.dirname(synthetic_model_checkpoint),
        )
        from core.inference.gesture_predictor import LSTMGesturePredictor

        predictor = LSTMGesturePredictor(device=torch.device("cpu"))
        dirty: dict[str, float] = {"accel_x": 1.0, "gyro_Y": 2.0, "flex_0": 100.0}
        clean = predictor._sanitize_sensor_dict(dirty)
        assert "accelX" in clean
        assert "gyroY" in clean
        assert clean["accelX"] == 1.0
        assert clean["gyroY"] == 2.0

    def test_add_sensor_dict_buffers(self, monkeypatch, synthetic_model_checkpoint: str) -> None:
        import os
        monkeypatch.setattr(
            "core.inference.gesture_predictor.MODEL_PATH",
            os.path.join(synthetic_model_checkpoint, "model.pth"),
        )
        monkeypatch.setattr(
            "core.inference.gesture_predictor.MODELS_DIR",
            os.path.dirname(synthetic_model_checkpoint),
        )
        from core.inference.gesture_predictor import LSTMGesturePredictor

        predictor = LSTMGesturePredictor(device=torch.device("cpu"))
        sample: dict[str, float] = {
            f"flex{i}": float(150 + i) for i in range(5)
        }
        for axis, val in zip(["X", "Y", "Z"], [100.0, 200.0, 300.0], strict=True):
            sample[f"accel{axis}"] = val
        for axis, val in zip(["X", "Y", "Z"], [10.0, 20.0, 30.0], strict=True):
            sample[f"gyro{axis}"] = val

        predictor.add_sensor_dict(sample)
        assert len(predictor.buffer) == 1

    def test_can_predict_requires_full_buffer(
        self, monkeypatch, synthetic_model_checkpoint: str
    ) -> None:
        import os
        monkeypatch.setattr(
            "core.inference.gesture_predictor.MODEL_PATH",
            os.path.join(synthetic_model_checkpoint, "model.pth"),
        )
        monkeypatch.setattr(
            "core.inference.gesture_predictor.MODELS_DIR",
            os.path.dirname(synthetic_model_checkpoint),
        )
        from core.inference.gesture_predictor import LSTMGesturePredictor

        predictor = LSTMGesturePredictor(device=torch.device("cpu"))
        assert predictor.can_predict() is False

    def test_confidence_threshold_for_known_class(
        self, monkeypatch, synthetic_model_checkpoint: str
    ) -> None:
        import os
        monkeypatch.setattr(
            "core.inference.gesture_predictor.MODEL_PATH",
            os.path.join(synthetic_model_checkpoint, "model.pth"),
        )
        monkeypatch.setattr(
            "core.inference.gesture_predictor.MODELS_DIR",
            os.path.dirname(synthetic_model_checkpoint),
        )
        from core.inference.gesture_predictor import LSTMGesturePredictor

        predictor = LSTMGesturePredictor(device=torch.device("cpu"))
        threshold = predictor._confidence_threshold_for("REST")
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0

    def test_confidence_threshold_unknown_class(
        self, monkeypatch, synthetic_model_checkpoint: str
    ) -> None:
        import os
        monkeypatch.setattr(
            "core.inference.gesture_predictor.MODEL_PATH",
            os.path.join(synthetic_model_checkpoint, "model.pth"),
        )
        monkeypatch.setattr(
            "core.inference.gesture_predictor.MODELS_DIR",
            os.path.dirname(synthetic_model_checkpoint),
        )
        from core.inference.gesture_predictor import LSTMGesturePredictor

        predictor = LSTMGesturePredictor(device=torch.device("cpu"))
        threshold = predictor._confidence_threshold_for("NONEXISTENT_GESTURE")
        assert isinstance(threshold, float)

    def test_probability_result_shape(
        self, monkeypatch, synthetic_model_checkpoint: str
    ) -> None:
        import os
        monkeypatch.setattr(
            "core.inference.gesture_predictor.MODEL_PATH",
            os.path.join(synthetic_model_checkpoint, "model.pth"),
        )
        monkeypatch.setattr(
            "core.inference.gesture_predictor.MODELS_DIR",
            os.path.dirname(synthetic_model_checkpoint),
        )
        from core.inference.gesture_predictor import LSTMGesturePredictor

        predictor = LSTMGesturePredictor(device=torch.device("cpu"))
        n_classes = len(predictor.classes)
        probs = torch.rand(1, n_classes, device=torch.device("cpu"))
        probs = probs / probs.sum(dim=1, keepdim=True)
        idx, conf, gap, all_probs = predictor._probability_result(probs)
        assert isinstance(idx, int)
        assert isinstance(conf, float)
        assert isinstance(gap, float)
        assert len(all_probs) == n_classes
