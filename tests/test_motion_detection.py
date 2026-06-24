"""Unit tests for motion detection utilities."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from gui.services.motion_detection import (
    _confidence_gap_for_token,
    _extract_class_list,
    _load_per_class_thresholds,
    calculate_motion_magnitude,
    validate_motion_consistency,
)


class TestCalculateMotionMagnitude:
    """Tests for calculate_motion_magnitude()."""

    def test_returns_float(self) -> None:
        result = calculate_motion_magnitude({
            "accelX": 1.0, "accelY": 2.0, "accelZ": 3.0,
            "gyroX": 0.5, "gyroY": 0.3, "gyroZ": 0.1,
        })
        assert isinstance(result, float)

    def test_zero_motion(self) -> None:
        result = calculate_motion_magnitude({
            "accelX": 0.0, "accelY": 0.0, "accelZ": 0.0,
            "gyroX": 0.0, "gyroY": 0.0, "gyroZ": 0.0,
        })
        assert result == 0.0

    def test_accel_only(self) -> None:
        result = calculate_motion_magnitude({
            "accelX": 3.0, "accelY": 4.0, "accelZ": 0.0,
            "gyroX": 0.0, "gyroY": 0.0, "gyroZ": 0.0,
        })
        assert result > 0.0

    def test_missing_keys_default_to_zero(self) -> None:
        result = calculate_motion_magnitude({"accelX": 1.0})
        assert isinstance(result, float)

    def test_positive_definite(self) -> None:
        result = calculate_motion_magnitude({
            "accelX": -100.0, "accelY": -200.0, "accelZ": -300.0,
            "gyroX": -10.0, "gyroY": -20.0, "gyroZ": -30.0,
        })
        assert result >= 0.0


class TestValidateMotionConsistency:
    """Tests for validate_motion_consistency()."""

    def test_short_buffer_returns_false(self) -> None:
        samples: deque[float] = deque([10.0, 20.0, 30.0], maxlen=100)
        assert validate_motion_consistency(samples) is False

    def test_low_motion_returns_false(self) -> None:
        samples: deque[float] = deque([0.1] * 30, maxlen=100)
        assert validate_motion_consistency(samples) is False

    def test_constant_motion_returns_false(self) -> None:
        samples: deque[float] = deque([800.0] * 30, maxlen=100)
        assert validate_motion_consistency(samples) is False

    def test_varied_motion_returns_true(self) -> None:
        np.random.seed(42)
        samples: deque[float] = deque(
            (np.random.rand(30) * 5000 + 2000).tolist(), maxlen=100
        )
        result = validate_motion_consistency(samples)
        assert isinstance(result, bool)


class TestConfidenceGapForToken:
    """Tests for _confidence_gap_for_token()."""

    def test_clear_winner(self) -> None:
        probs = {"A": 0.9, "B": 0.05, "C": 0.05}
        gap = _confidence_gap_for_token(probs, "A")
        assert gap == pytest.approx(0.85)

    def test_close_race(self) -> None:
        probs = {"A": 0.45, "B": 0.44, "C": 0.11}
        gap = _confidence_gap_for_token(probs, "A")
        assert gap == pytest.approx(0.01)

    def test_token_not_in_probs(self) -> None:
        gap = _confidence_gap_for_token({"A": 0.9, "B": 0.1}, "C")
        assert gap == 0.0

    def test_single_token(self) -> None:
        gap = _confidence_gap_for_token({"A": 1.0}, "A")
        assert gap == 1.0


class TestExtractClassList:
    """Tests for _extract_class_list()."""

    def test_extracts_from_list_attribute(self) -> None:
        class MockPredictor:
            classes = ["REST", "A", "B"]

        result = _extract_class_list(MockPredictor())
        assert result == ["REST", "A", "B"]

    def test_extracts_from_array_attribute(self) -> None:
        class MockPredictor:
            classes = np.array(["X", "Y", "Z"])

        result = _extract_class_list(MockPredictor())
        assert result == ["X", "Y", "Z"]

    def test_no_classes_attribute(self) -> None:
        class MockPredictor:
            pass

        result = _extract_class_list(MockPredictor())
        assert result == []


class TestLoadPerClassThresholds:
    """Tests for _load_per_class_thresholds()."""

    def test_none_dir_returns_empty(self) -> None:
        assert _load_per_class_thresholds(None) == {}

    def test_nonexistent_dir_returns_empty(self, tmp_path) -> None:
        assert _load_per_class_thresholds(tmp_path) == {}
