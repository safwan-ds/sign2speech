"""Unit tests for data_utils module"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import (
    normalize_value,
    compute_velocity,
    compute_acceleration,
    detect_motion_boundaries,
    get_feature_names,
    compute_rolling_statistics,
)


class TestNormalizeValue:
    def test_flex_sensor_mid_range(self):
        result = normalize_value("flex0", 126)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_flex_sensor_min(self):
        result = normalize_value("flex0", 28)
        assert result == pytest.approx(0.0)

    def test_flex_sensor_max(self):
        result = normalize_value("flex0", 224)
        assert result == pytest.approx(1.0)

    def test_accel_mid_range(self):
        result = normalize_value("accelX", 0)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_gyro_mid_range(self):
        result = normalize_value("gyroY", 0)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_unknown_sensor_returns_none(self):
        assert normalize_value("unknown", 100) is None

    def test_clipping_above_max(self):
        result = normalize_value("flex0", 9999)
        assert result == pytest.approx(1.0)

    def test_clipping_below_min(self):
        result = normalize_value("flex0", -9999)
        assert result == pytest.approx(0.0)


class TestComputeVelocity:
    def test_constant_sequence_zero_velocity(self):
        seq = np.ones((10, 3), dtype=np.float32)
        vel = compute_velocity(seq)
        assert vel.shape == seq.shape
        assert np.allclose(vel, 0.0)

    def test_linear_sequence_constant_velocity(self):
        seq = np.arange(30, dtype=np.float32).reshape(10, 3)
        vel = compute_velocity(seq)
        assert vel.shape == seq.shape
        # After the first element (padded), velocity should be constant
        assert np.allclose(vel[1:], 3.0)

    def test_output_dtype(self):
        seq = np.ones((5, 2), dtype=np.float64)
        vel = compute_velocity(seq)
        assert vel.dtype == np.float32


class TestComputeAcceleration:
    def test_constant_velocity_zero_acceleration(self):
        seq = np.arange(30, dtype=np.float32).reshape(10, 3)
        acc = compute_acceleration(seq)
        assert acc.shape == seq.shape
        # Constant velocity -> zero acceleration (except boundaries)
        assert np.allclose(acc[2:], 0.0)

    def test_output_shape(self):
        seq = np.random.randn(15, 5).astype(np.float32)
        acc = compute_acceleration(seq)
        assert acc.shape == seq.shape


class TestDetectMotionBoundaries:
    def test_no_motion(self):
        seq = np.ones((50, 11), dtype=np.float32) * 0.5
        result = detect_motion_boundaries(seq, threshold=0.01)
        assert result is None

    def test_clear_motion(self):
        seq = np.ones((50, 11), dtype=np.float32) * 0.5
        # Add motion in the middle
        rng = np.random.RandomState(42)
        seq[15:35] += rng.randn(20, 11) * 0.1
        result = detect_motion_boundaries(seq, threshold=0.01, min_duration=3)
        assert result is not None
        start, end = result
        assert start >= 0
        assert end <= 50
        assert end > start


class TestGetFeatureNames:
    def test_correct_feature_count(self):
        names = get_feature_names()
        assert len(names) == 11  # 5 flex + 6 IMU

    def test_flex_names_correct(self):
        names = get_feature_names()
        for i in range(5):
            assert names[i] == f"flex{i}"

    def test_imu_names_present(self):
        names = get_feature_names()
        assert "accelX" in names
        assert "gyroZ" in names


class TestComputeRollingStatistics:
    def test_output_shapes(self):
        seq = np.random.randn(20, 5).astype(np.float32)
        stats = compute_rolling_statistics(seq, window_size=3)
        assert stats["mean"].shape == seq.shape
        assert stats["std"].shape == seq.shape
        assert stats["min"].shape == seq.shape
        assert stats["max"].shape == seq.shape

    def test_mean_of_constant(self):
        seq = np.ones((10, 2), dtype=np.float32) * 3.0
        stats = compute_rolling_statistics(seq, window_size=5)
        assert np.allclose(stats["mean"], 3.0)
        assert np.allclose(stats["std"], 0.0)
