"""Unit tests for data_utils module"""

import numpy as np
import pytest

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


class TestNormalizeYawRotation:
    def test_output_shape_unchanged(self):
        from utils.data_utils import normalize_yaw_rotation

        seq = np.random.randn(30, 11).astype(np.float32)
        result = normalize_yaw_rotation(seq)
        assert result.shape == seq.shape

    def test_output_dtype_is_float32(self):
        from utils.data_utils import normalize_yaw_rotation

        seq = np.random.randn(30, 11).astype(np.float64)
        result = normalize_yaw_rotation(seq)
        assert result.dtype == np.float32

    def test_flex_sensors_unchanged(self):
        from utils.data_utils import normalize_yaw_rotation

        rng = np.random.RandomState(0)
        seq = rng.randn(30, 11).astype(np.float32)
        result = normalize_yaw_rotation(seq)
        # Flex sensors (indices 0–4) and accelZ/gyroZ (indices 7, 10) must not change
        assert np.allclose(result[:, :5], seq[:, :5])
        assert np.allclose(result[:, 7], seq[:, 7])   # accelZ
        assert np.allclose(result[:, 10], seq[:, 10])  # gyroZ

    def test_yaw_invariance(self):
        """Same gesture at two different yaw angles must produce the same output."""
        from utils.data_utils import normalize_yaw_rotation

        rng = np.random.RandomState(42)
        seq = rng.randn(30, 11).astype(np.float32)

        # Rotate accelX/Y and gyroX/Y by 90 degrees (pi/2) to simulate
        # the user facing a different direction.  All frames are rotated
        # uniformly, including the leading reference frames used for yaw
        # estimation, so the estimate correctly captures the offset.
        angle = np.pi / 2
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        rotated = seq.copy()
        ax, ay = seq[:, 5].copy(), seq[:, 6].copy()
        rotated[:, 5] = cos_a * ax - sin_a * ay
        rotated[:, 6] = sin_a * ax + cos_a * ay
        gx, gy = seq[:, 8].copy(), seq[:, 9].copy()
        rotated[:, 8] = cos_a * gx - sin_a * gy
        rotated[:, 9] = sin_a * gx + cos_a * gy

        norm_orig = normalize_yaw_rotation(seq)
        norm_rotated = normalize_yaw_rotation(rotated)

        assert np.allclose(norm_orig, norm_rotated, atol=1e-5)

    def test_reference_frames_larger_than_sequence_uses_all_frames(self):
        """When reference_frames exceeds sequence length, all frames are used (no error)."""
        from utils.data_utils import normalize_yaw_rotation

        seq = np.random.RandomState(7).randn(3, 11).astype(np.float32)
        # reference_frames=100 >> sequence length 3 — must not raise
        result = normalize_yaw_rotation(seq, reference_frames=100)
        assert result.shape == seq.shape

    def test_single_reference_frame(self):
        """A single rest frame is sufficient to estimate yaw without error."""
        from utils.data_utils import normalize_yaw_rotation

        seq = np.random.RandomState(99).randn(10, 11).astype(np.float32)
        result = normalize_yaw_rotation(seq, reference_frames=1)
        assert result.shape == seq.shape
        assert result.dtype == np.float32

    def test_passthrough_when_sequence_has_too_few_features(self):
        from utils.data_utils import normalize_yaw_rotation

        # Feature vector too short to contain all IMU indices — returned as-is
        seq = np.ones((10, 3), dtype=np.float32)
        result = normalize_yaw_rotation(seq)
        assert np.array_equal(result, seq)

    def test_passthrough_on_empty_sequence(self):
        from utils.data_utils import normalize_yaw_rotation

        seq = np.empty((0, 11), dtype=np.float32)
        result = normalize_yaw_rotation(seq)
        assert result.shape == (0, 11)


class TestConvertToSnakeCase:
    def test_spaces_become_underscores(self):
        from utils.data_utils import convert_to_snake_case

        assert convert_to_snake_case("hello world") == "hello_world"

    def test_no_spaces_unchanged(self):
        from utils.data_utils import convert_to_snake_case

        assert convert_to_snake_case("hello") == "hello"

    def test_multiple_spaces(self):
        from utils.data_utils import convert_to_snake_case

        assert convert_to_snake_case("a b c") == "a_b_c"


class TestNormalizeDataframe:
    def _make_df(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "flex0": [28, 126, 224],
                "flex1": [28, 126, 224],
                "flex2": [28, 126, 224],
                "flex3": [28, 126, 224],
                "flex4": [28, 126, 224],
                "accelX": [-32768, 0, 32767],
                "accelY": [-32768, 0, 32767],
                "accelZ": [-32768, 0, 32767],
                "gyroX": [-32768, 0, 32767],
                "gyroY": [-32768, 0, 32767],
                "gyroZ": [-32768, 0, 32767],
            }
        )

    def test_adds_normalized_columns(self):
        from utils.data_utils import normalize_dataframe

        df = self._make_df()
        result = normalize_dataframe(df)
        assert "flex0_norm" in result.columns
        assert "accelX_norm" in result.columns
        assert "gyroZ_norm" in result.columns

    def test_original_columns_unchanged(self):
        from utils.data_utils import normalize_dataframe

        df = self._make_df()
        result = normalize_dataframe(df)
        assert "flex0" in result.columns

    def test_normalized_values_in_range(self):
        from utils.data_utils import normalize_dataframe

        df = self._make_df()
        result = normalize_dataframe(df)
        assert result["flex0_norm"].between(0.0, 1.0).all()
        assert result["accelX_norm"].between(0.0, 1.0).all()


class TestExtractNormalizedFeatures:
    def test_returns_array_with_correct_shape(self):
        import pandas as pd
        from utils.data_utils import normalize_dataframe, extract_normalized_features

        df = pd.DataFrame(
            {
                "flex0": [100, 200],
                "flex1": [100, 200],
                "flex2": [100, 200],
                "flex3": [100, 200],
                "flex4": [100, 200],
                "accelX": [0, 100],
                "accelY": [0, 100],
                "accelZ": [0, 100],
                "gyroX": [0, 100],
                "gyroY": [0, 100],
                "gyroZ": [0, 100],
            }
        )
        df = normalize_dataframe(df)
        features = extract_normalized_features(df)
        assert features.shape == (2, 11)
        assert features.dtype == np.float32

    def test_raises_on_missing_columns(self):
        import pandas as pd
        from utils.data_utils import extract_normalized_features

        df = pd.DataFrame({"flex0_norm": [0.5, 0.6]})
        with pytest.raises(ValueError, match="Missing expected normalized columns"):
            extract_normalized_features(df)


class TestComputeMagnitude:
    def test_output_shape_is_column_vector(self):
        from utils.data_utils import compute_magnitude

        seq = np.ones((10, 6), dtype=np.float32)
        mag = compute_magnitude(seq, [3, 4, 5])
        assert mag.shape == (10, 1)

    def test_magnitude_of_unit_vector(self):
        from utils.data_utils import compute_magnitude

        seq = np.zeros((5, 3), dtype=np.float32)
        seq[:, 0] = 1.0  # only x component
        mag = compute_magnitude(seq, [0, 1, 2])
        assert np.allclose(mag, 1.0)

    def test_raises_for_single_axis(self):
        from utils.data_utils import compute_magnitude

        with pytest.raises(ValueError, match="at least 2 axes"):
            compute_magnitude(np.ones((5, 3), dtype=np.float32), [0])


class TestExtractEnhancedFeatures:
    def test_with_derivatives_doubles_features(self):
        from utils.data_utils import extract_enhanced_features

        seq = np.random.randn(20, 11).astype(np.float32)
        result = extract_enhanced_features(
            seq, include_derivatives=True, include_stats=False
        )
        # base + vel + accel = 3 * 11 = 33
        assert result.shape == (20, 33)

    def test_with_stats_only_adds_mean_and_std(self):
        from utils.data_utils import extract_enhanced_features

        seq = np.random.randn(20, 11).astype(np.float32)
        result = extract_enhanced_features(
            seq, include_derivatives=False, include_stats=True
        )
        # base + mean + std = 3 * 11 = 33
        assert result.shape == (20, 33)

    def test_output_dtype_is_float32(self):
        from utils.data_utils import extract_enhanced_features

        seq = np.ones((10, 5), dtype=np.float64)
        result = extract_enhanced_features(seq)
        assert result.dtype == np.float32


class TestExtractFrequencyFeatures:
    def test_returns_dict_with_expected_keys(self):
        from utils.data_utils import extract_frequency_features

        seq = np.random.randn(50, 3).astype(np.float32)
        freq = extract_frequency_features(seq, sampling_rate=100)
        assert "feature_0_dominant_freq" in freq
        assert "feature_0_spectral_energy" in freq
        assert "feature_0_spectral_entropy" in freq

    def test_key_count_matches_feature_count(self):
        from utils.data_utils import extract_frequency_features

        n_features = 4
        seq = np.random.randn(50, n_features).astype(np.float32)
        freq = extract_frequency_features(seq, sampling_rate=100)
        # 3 metrics per feature
        assert len(freq) == 3 * n_features
