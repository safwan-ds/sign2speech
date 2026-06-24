"""Unit tests for utils/normalization.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.normalization import extract_normalized_features
from utils.normalization import get_feature_names
from utils.normalization import normalize_dataframe
from utils.normalization import normalize_value


class TestNormalizeValue:
    """Tests for normalize_value()."""

    def test_flex_sensor_returns_float(self) -> None:
        result = normalize_value("flex0", 150)
        assert isinstance(result, float)

    def test_flex_sensor_clips_min(self) -> None:
        result = normalize_value("flex0", -999)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_flex_sensor_clips_max(self) -> None:
        result = normalize_value("flex0", 9999)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_flex_sensor_mid_range(self) -> None:
        result = normalize_value("flex0", 162.5)
        assert result is not None
        assert 0.0 < result < 1.0

    def test_accel_sensor_normalizes(self) -> None:
        result = normalize_value("accelX", 0)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_accel_sensor_extremes(self) -> None:
        lo = normalize_value("accelY", -32768)
        hi = normalize_value("accelY", 32767)
        assert lo == 0.0
        assert hi == 1.0

    def test_gyro_sensor_normalizes(self) -> None:
        result = normalize_value("gyroZ", -1000)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_unknown_sensor_returns_none(self) -> None:
        assert normalize_value("temperature", 25.0) is None
        assert normalize_value("", 0.0) is None
        assert normalize_value("random123", 42.0) is None

    def test_flex_index_out_of_range_uses_default(self) -> None:
        result = normalize_value("flex99", 500)
        assert result is not None
        assert 0.0 <= result <= 1.0


class TestNormalizeDataFrame:
    """Tests for normalize_dataframe()."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        data: dict[str, list[float]] = {
            "flex0": [150.0, 200.0, 100.0],
            "flex1": [160.0, 180.0, 140.0],
            "flex2": [130.0, 170.0, 150.0],
            "flex3": [140.0, 160.0, 120.0],
            "flex4": [155.0, 165.0, 145.0],
            "accelX": [0.0, 1000.0, -500.0],
            "accelY": [0.0, 500.0, -200.0],
            "accelZ": [16384.0, 0.0, -16384.0],
            "gyroX": [0.0, 100.0, -50.0],
            "gyroY": [10.0, -10.0, 0.0],
            "gyroZ": [0.0, 0.0, 0.0],
        }
        return pd.DataFrame(data)

    def test_adds_normalized_columns(self, sample_df: pd.DataFrame) -> None:
        result = normalize_dataframe(sample_df)
        assert "flex0_norm" in result.columns
        assert "accelX_norm" in result.columns
        assert "gyroZ_norm" in result.columns

    def test_normalized_values_in_range(self, sample_df: pd.DataFrame) -> None:
        result = normalize_dataframe(sample_df)
        for col in result.columns:
            if col.endswith("_norm"):
                assert result[col].between(0.0, 1.0).all(), f"{col} out of range"

    def test_does_not_mutate_original(self, sample_df: pd.DataFrame) -> None:
        original_columns = set(sample_df.columns)
        normalize_dataframe(sample_df)
        assert set(sample_df.columns) == original_columns

    def test_handles_missing_sensor_columns(self) -> None:
        df = pd.DataFrame({"flex0": [100.0, 200.0]})
        result = normalize_dataframe(df)
        assert "flex0_norm" in result.columns
        assert "flex1_norm" not in result.columns


class TestExtractNormalizedFeatures:
    """Tests for extract_normalized_features()."""

    def test_returns_array_with_correct_shape(self) -> None:
        normalized_df = normalize_dataframe(
            pd.DataFrame({
                "flex0": [150.0] * 10,
                "flex1": [160.0] * 10,
                "flex2": [130.0] * 10,
                "flex3": [140.0] * 10,
                "flex4": [155.0] * 10,
                "accelX": [0.0] * 10,
                "accelY": [0.0] * 10,
                "accelZ": [0.0] * 10,
                "gyroX": [0.0] * 10,
                "gyroY": [0.0] * 10,
                "gyroZ": [0.0] * 10,
            })
        )
        features = extract_normalized_features(normalized_df)
        assert features.shape == (10, 11)
        assert features.dtype == np.float32

    def test_raises_on_missing_columns(self) -> None:
        df = pd.DataFrame({"flex0_norm": [0.5, 0.6]})
        with pytest.raises(ValueError, match="Missing expected normalized columns"):
            extract_normalized_features(df)


class TestGetFeatureNames:
    """Tests for get_feature_names()."""

    def test_returns_correct_count(self) -> None:
        names = get_feature_names()
        assert len(names) == 11

    def test_contains_flex_sensors(self) -> None:
        names = get_feature_names()
        for i in range(5):
            assert f"flex{i}" in names

    def test_contains_imu_sensors(self) -> None:
        names = get_feature_names()
        for axis in ["X", "Y", "Z"]:
            assert f"accel{axis}" in names
            assert f"gyro{axis}" in names

    def test_order_is_flex_first_then_imu(self) -> None:
        names = get_feature_names()
        flex_indices = [i for i, n in enumerate(names) if n.startswith("flex")]
        accel_indices = [i for i, n in enumerate(names) if n.startswith("accel")]
        assert max(flex_indices) < min(accel_indices)
